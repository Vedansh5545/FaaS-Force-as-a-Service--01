import os
from collections import Counter
from typing import Any

import cv2 as cv
from ultralytics import YOLO

try:
    from .ExpertBase import ExpertBase, resolve_path
except ImportError:
    from ExpertBase import ExpertBase, resolve_path


class ObjectDetection(ExpertBase):
    task_type = "object"

    def __init__(self):
        super().__init__("yolo_vision")

    def _release_model(self):
        """Drop the YOLO network, predictor, and checkpoint, then free CUDA cache."""
        model = self.model
        if model is not None:
            if getattr(model, "predictor", None) is not None:
                model.predictor = None
            inner = getattr(model, "model", None)
            if inner is not None and not isinstance(inner, (str, bytes)):
                move_to_cpu = getattr(inner, "cpu", None)
                if callable(move_to_cpu):
                    try:
                        move_to_cpu()
                    except Exception:
                        pass
                model.model = None
            if getattr(model, "ckpt", None) is not None:
                model.ckpt = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def load_model(self):
        """Initializes the YOLO model using the path from config."""
        if self.is_loaded():
            return True
        if not self.is_enabled():
            self.logger.warning("YOLO vision is not enabled in the configuration.")
            return False

        model_path = self.config.get("model_path", "yolov8s.pt")
        if not model_path.startswith(("http://", "https://")):
            resolved = resolve_path(model_path)
            if os.path.exists(resolved):
                model_path = resolved

        self.logger.info("Loading YOLO vision model %s...", model_path)
        self.model = YOLO(model_path)
        self.logger.info("Loading YOLO model complete.")
        return True

    def run_inference(self, input_data: Any, **kwargs) -> str:
        """Runs object detection on an image path or array and returns a summary."""
        try:
            frame = cv.imread(input_data) if isinstance(input_data, str) else input_data
            if frame is None or getattr(frame, "size", 0) == 0:
                self.logger.warning("Empty frame received or could not load image.")
                return "No objects detected."

            if self.model is None:
                self.load_model()
            if self.model is None:
                return "Object detection model is not loaded."

            yolo_results = self.model(frame, verbose=False)
            object_counts = Counter()
            min_conf = float(self.config.get("min_conf", 0.5))

            if yolo_results:
                for box in yolo_results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    conf = box.conf.item()
                    if conf >= min_conf:
                        object_counts[class_name] += 1

            if len(object_counts) < 1:
                return "No objects detected."

            output_parts = []
            for name, count in object_counts.items():
                if count > 1:
                    output_parts.append(f"{count} {name}")
                else:
                    output_parts.append(name)

            return " " + ", ".join(output_parts)
        except Exception as exc:
            self.logger.error("Error during inference: %s", exc)
            return "An unexpected error occurred during object detection."
