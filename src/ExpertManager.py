import importlib
import logging
import sys
from pathlib import Path

try:
    from .ExpertBase import ExpertBase, get_logger
except ImportError:
    from ExpertBase import ExpertBase, get_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

TASK_ALIASES = {
    "face": "face",
    "facial": "face",
    "facial_recognition": "face",
    "object": "object",
    "objects": "object",
    "yolo": "object",
    "object_detection": "object",
    "text": "text",
    "ocr": "text",
}


class ExpertManager:
    """
    Instantiates and routes work to experts.

    Add a new expert by creating a sibling module that subclasses ExpertBase
    and sets `task_type`. ExpertManager discovers it on the next construction.
    You can also pass an explicit expert list or call `register`.
    """

    def __init__(self, experts=None):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing ExpertManager and instantiating experts...")
        self._discover_expert_modules()
        self.experts = list(experts) if experts is not None else self._build_default_experts()

    def _discover_expert_modules(self):
        src_dir = Path(__file__).resolve().parent
        skip = {"ExpertBase", "ExpertManager"}
        package = __package__ or ""
        if not package and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        for path in sorted(src_dir.glob("*.py")):
            name = path.stem
            if name in skip or name.startswith("_"):
                continue
            try:
                if package:
                    importlib.import_module(f".{name}", package=package)
                else:
                    importlib.import_module(name)
            except Exception as exc:
                self.logger.error("Failed to import expert module %s: %s", name, exc)

    def _build_default_experts(self):
        experts = []
        for expert_cls in ExpertBase.registered_experts():
            if getattr(expert_cls, "__abstractmethods__", None):
                continue
            try:
                experts.append(expert_cls())
            except Exception as exc:
                self.logger.error("Failed to instantiate %s: %s", expert_cls.__name__, exc)
        return experts

    def register(self, expert):
        """Register an already-constructed expert at runtime."""
        self.experts.append(expert)
        self.logger.info("Registered expert %s for task '%s'.", expert.__class__.__name__, expert.task_type)
        return expert

    def resolve_task(self, task: str) -> str:
        key = (task or "").strip().lower()
        return TASK_ALIASES.get(key, key)

    def get_expert(self, task=None, name=None, condition=None):
        """
        Retrieve an expert by task type, class/section name, or a predicate.

        `condition` matches the Reference AIManager.get_module API.
        """
        if condition is not None:
            return next((expert for expert in self.experts if condition(expert)), None)
        if name:
            needle = name.lower()
            return next(
                (
                    expert
                    for expert in self.experts
                    if expert.__class__.__name__.lower() == needle
                    or expert.section_name.lower() == needle
                ),
                None,
            )
        if task:
            resolved = self.resolve_task(task)
            return next((expert for expert in self.experts if expert.task_type == resolved), None)
        return None

    def execute_module(self, condition, input_key, *args, **kwargs):
        """Finds an expert with `condition` and executes it (Reference-compatible)."""
        expert = self.get_expert(condition=condition)
        if expert:
            return expert.execute(input_key, *args, **kwargs)
        self.logger.error("No matching expert found for execution.")
        return None

    def execute(self, task, input_key, input_data, **kwargs):
        """Run the expert responsible for a face, object, or text task."""
        expert = self.get_expert(task=task)
        if expert is None:
            self.logger.error("No expert registered for task '%s'.", task)
            return None
        return expert.execute(input_key, input_data, **kwargs)

    def recognize_face(self, input_key, input_data, **kwargs):
        return self.execute("face", input_key, input_data, **kwargs)

    def detect_objects(self, input_key, input_data, **kwargs):
        return self.execute("object", input_key, input_data, **kwargs)

    def read_text(self, input_key, input_data, **kwargs):
        return self.execute("text", input_key, input_data, **kwargs)

    def load_all_models(self):
        """Load weights for every instantiated expert."""
        self.logger.info("Loading models for all initialized experts...")
        for expert in self.experts:
            try:
                expert.load_model()
            except Exception as exc:
                self.logger.error("Failed to load %s model: %s", expert.__class__.__name__, exc)
        self.logger.info("Finished loading all expert models.")

    def unload_all_models(self):
        """Release weights for every instantiated expert."""
        self.logger.info("Unloading models for all initialized experts...")
        for expert in self.experts:
            try:
                expert.unload_model()
            except Exception as exc:
                self.logger.error("Failed to unload %s model: %s", expert.__class__.__name__, exc)
        self.logger.info("Finished unloading all expert models.")

    def unload_model(self, task=None, name=None):
        """Unload a single expert by task type or class/section name."""
        expert = self.get_expert(task=task, name=name)
        if expert is None:
            self.logger.error("No expert found to unload for task=%s name=%s.", task, name)
            return False
        return expert.unload_model()

    def models_loaded(self):
        """Map of task_type -> whether that expert currently holds a model."""
        return {expert.task_type: expert.is_loaded() for expert in self.experts}


_manager = None


def get_expert_manager():
    """Return a process-wide ExpertManager, created on first use."""
    global _manager
    if _manager is None:
        _manager = ExpertManager()
    return _manager


def __getattr__(name: str):
    if name == "expert_manager":
        return get_expert_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
