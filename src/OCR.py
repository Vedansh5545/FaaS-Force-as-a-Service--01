from typing import Any

import cv2 as cv
import indra_ocr
import numpy as np

try:
    from .ExpertBase import ExpertBase, resolve_path
except ImportError:
    from ExpertBase import ExpertBase, resolve_path


class OCR(ExpertBase):
    task_type = "text"

    def __init__(self):
        super().__init__("ocr")

    def _release_model(self):
        """Shut down Indra's worker pool and drop ONNX sessions."""
        converter = self.model
        engine = getattr(converter, "ocr", None)
        if engine is None:
            return
        try:
            pool = getattr(engine, "pool", None)
            if pool is not None:
                shutdown = getattr(pool, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        shutdown(wait=False)
                    except Exception as exc:
                        self.logger.warning("Failed to shut down OCR worker pool: %s", exc)
                engine.pool = None

            for name in ("det", "rec", "mask", "order", "orient", "script"):
                if hasattr(engine, name):
                    setattr(engine, name, None)

            groups = getattr(engine, "groups", None)
            if isinstance(groups, dict):
                groups.clear()

            lms = getattr(engine, "lms", None)
            if isinstance(lms, dict):
                lms.clear()

            for name in ("math", "lm", "_pic"):
                if hasattr(engine, name):
                    setattr(engine, name, None)
        finally:
            if hasattr(converter, "ocr"):
                converter.ocr = None

    def load_model(self):
        """Loads the Indra OCR converter."""
        if self.is_loaded():
            return True
        if not self.is_enabled():
            self.logger.warning("OCR is not enabled in the configuration.")
            return False

        load_kwargs = {
            "threads": int(self.config.get("threads", 4)),
            "mode": self.config.get("mode", "ocr"),
        }
        model_dir = self.config.get("model_dir")
        if model_dir:
            load_kwargs["path"] = resolve_path(model_dir)

        self.logger.info("Loading Indra OCR model...")
        self.model = indra_ocr.load(**load_kwargs)
        self.logger.info("Loading Indra OCR model complete.")
        return True

    def run_inference(self, input_data: Any, **kwargs) -> str:
        """Reads text from an image/PDF path or an in-memory image."""
        try:
            if not self._ensure_loaded():
                return "OCR model is not loaded."

            output_format = kwargs.get("output_format", self.config.get("output_format", "text"))
            pages = self._run_ocr(input_data)
            texts = [self._page_text(page, output_format) for page in pages]
            combined = "\n\n".join(text for text in texts if text).strip()
            if not combined:
                return "No text detected."
            return combined
        except Exception as exc:
            self.logger.error("Error during OCR inference: %s", exc)
            return "An unexpected error occurred during OCR."

    def _run_ocr(self, input_data: Any):
        if isinstance(input_data, str):
            return self._collect_pages(self.model.convert(input_data))

        frame = input_data
        if frame is None or getattr(frame, "size", 0) == 0:
            self.logger.warning("Empty frame received or could not load image.")
            return []

        gray = self._to_gray(frame)
        return self._collect_pages(self.model.ocr(gray))

    def _to_gray(self, frame):
        array = np.asarray(frame)
        if array.ndim == 2:
            return array.astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 3:
            return cv.cvtColor(array, cv.COLOR_BGR2GRAY)
        if array.ndim == 3 and array.shape[2] == 4:
            return cv.cvtColor(array, cv.COLOR_BGRA2GRAY)
        raise ValueError("Unsupported image shape for OCR.")

    def _collect_pages(self, result):
        if result is None:
            return []
        if hasattr(result, "text") or hasattr(result, "markdown"):
            return [result]

        pages = []
        for item in result:
            if isinstance(item, tuple) and len(item) >= 2:
                pages.append(item[1])
            else:
                pages.append(item)
        return pages

    def _page_text(self, page, output_format: str) -> str:
        if output_format == "markdown":
            return (getattr(page, "markdown", None) or getattr(page, "text", "") or "").strip()
        return (getattr(page, "text", None) or getattr(page, "markdown", "") or "").strip()
