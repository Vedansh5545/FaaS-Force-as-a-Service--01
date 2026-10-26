import gc
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Union

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "Config.yaml"

_REGISTERED_EXPERTS: list[type] = []


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


class ExpertBase(ABC):
    """
    Shared lifecycle for every expert: config, model load, profiling, and cache.

    Subclasses should set `task_type` (for example "face", "object", "text")
    so ExpertManager can route work. New experts are picked up automatically
    when their module lives beside this file and subclasses ExpertBase.
    """

    task_type: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not ExpertBase and cls not in _REGISTERED_EXPERTS:
            _REGISTERED_EXPERTS.append(cls)

    def __init__(self, section_name: str):
        self.section_name = section_name
        self.config = get_config().get(self.section_name, {}) or {}
        self.model = None
        self.logging_enabled = self.config.get("logging_enabled", True)
        self.logger = get_logger(self.__class__.__name__)
        self.cache = {}

    @classmethod
    def registered_experts(cls) -> list[type]:
        return list(_REGISTERED_EXPERTS)

    def is_enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def is_loaded(self) -> bool:
        return self.model is not None

    @abstractmethod
    def load_model(self):
        """Child implements specific model loading."""
        pass

    def unload_model(self):
        """Drop model weights and extra handles so they can be loaded again later."""
        if not self.is_loaded():
            self.logger.info("%s is already unloaded.", self.__class__.__name__)
            return False
        self.logger.info("Unloading %s model...", self.__class__.__name__)
        self._release_model()
        self.model = None
        self.cache.clear()
        gc.collect()
        self.logger.info("Unloaded %s model.", self.__class__.__name__)
        return True

    def _release_model(self):
        """Subclasses override to clear extra handles beyond `self.model`."""
        pass

    @abstractmethod
    def run_inference(self, input_data: Any, **kwargs) -> Union[Any, Generator]:
        """Child implements core logic (can be return or yield)."""
        pass

    def execute(self, input_key: str, input_data: Any, **kwargs) -> Union[Any, Generator]:
        """Handles profiling, lazy loading, and streaming detection."""
        start_time = time.perf_counter()
        result = self.run_inference(input_data, **kwargs)

        if isinstance(result, Generator):
            return self._stream_wrapper(result, start_time, input_key)

        end_time = time.perf_counter()
        self._log_profile(start_time, end_time)
        self.cache[input_key] = {"output": result, "time": end_time}
        return result

    def _stream_wrapper(self, gen, start_time, input_key):
        full_response = []
        try:
            for chunk in gen:
                full_response.append(str(chunk))
                yield chunk
        finally:
            closer = getattr(gen, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
            end_time = time.perf_counter()
            self._log_profile(start_time, end_time, is_stream=True)
            self.cache[input_key] = {"output": "".join(full_response), "time": end_time}

    def _log_profile(self, start, end, is_stream=False):
        duration = end - start
        stream_tag = "[STREAM]" if is_stream else "[STATIC]"
        if self.logging_enabled:
            self.logger.info("%s Latency: %.4fs", stream_tag, duration)

    def _ensure_loaded(self) -> bool:
        if self.model is not None:
            return True
        self.logger.warning("%s is not loaded. Attempting to load it now.", self.__class__.__name__)
        self.load_model()
        return self.model is not None
