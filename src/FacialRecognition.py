import os
import pickle
from typing import Any

import cv2 as cv

try:
    from .ExpertBase import ExpertBase, resolve_path
except ImportError:
    from ExpertBase import ExpertBase, resolve_path


class FacialRecognition(ExpertBase):
    task_type = "face"

    def __init__(self):
        super().__init__("face_recognition")
        self.detector = None
        self.data = None

    def is_loaded(self) -> bool:
        return self.model is not None and self.detector is not None and self.data is not None

    def _release_model(self):
        self.detector = None
        self.data = None

    def load_model(self):
        """Loads the face detection cascade and recognition encodings."""
        if self.is_loaded():
            return True
        if not self.is_enabled():
            self.logger.warning("Face recognition is not enabled in the configuration.")
            return False

        self.logger.info("Loading facial recognition model...")
        try:
            cascade_path = resolve_path(
                self.config.get(
                    "cascade_path",
                    "models/facial_recognition/haarcascade_frontalface_default.xml",
                )
            )
            encodings_path = resolve_path(
                self.config.get(
                    "encodings_path",
                    "models/facial_recognition/encodings.pickle",
                )
            )

            self.logger.info("Loading Face Cascade from %s", cascade_path)
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Cascade file not found at {cascade_path}")

            self.detector = cv.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise IOError("Failed to load CascadeClassifier")

            self.logger.info("Loading encodings from %s", encodings_path)
            if not os.path.exists(encodings_path):
                raise FileNotFoundError(f"Encodings file not found at {encodings_path}")

            with open(encodings_path, "rb") as handle:
                self.data = pickle.load(handle)

            try:
                import face_recognition  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "face_recognition is not installed. Install dlib-bin, then "
                    "face_recognition with --no-deps (see requirements.txt)."
                ) from exc

            self.model = {"detector": self.detector, "encodings": self.data}
        except Exception as exc:
            self.logger.error("Error loading models: %s", exc)
            self.detector = None
            self.data = None
            self.model = None
            return False

        self.logger.info("Loading facial recognition model complete.")
        return True

    def run_inference(self, input_data: Any, **kwargs) -> str:
        """Recognizes faces in the input image and returns a summary string."""
        if not self.is_loaded():
            self.logger.warning(
                "Facial recognition models are not loaded. Attempting to load them now."
            )
            self.load_model()
            if not self.is_loaded():
                return "Facial recognition models are not loaded."

        try:
            frame = cv.imread(input_data) if isinstance(input_data, str) else input_data
            if frame is None or getattr(frame, "size", 0) == 0:
                self.logger.warning("Empty frame received.")
                return "No image data."

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            rects = self.detector.detectMultiScale(
                gray,
                scaleFactor=float(self.config.get("scale_factor", 1.3)),
                minNeighbors=int(self.config.get("min_neighbors", 6)),
                minSize=tuple(self.config.get("min_size", [40, 40])),
                flags=cv.CASCADE_SCALE_IMAGE,
            )

            if len(rects) == 0:
                return "No faces detected."

            import face_recognition

            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            encodings = face_recognition.face_encodings(rgb, boxes)

            names = []
            tolerance = float(self.config.get("tolerance", 0.5))

            for encoding in encodings:
                distances = face_recognition.face_distance(self.data["encodings"], encoding)
                matches = [distance <= tolerance for distance in distances]
                name = "Unknown"
                if True in matches:
                    matched_idxs = [i for (i, matched) in enumerate(matches) if matched]
                    counts = {}
                    for idx in matched_idxs:
                        matched_name = self.data["names"][idx]
                        counts[matched_name] = counts.get(matched_name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)

            known_names = [name for name in names if name != "Unknown"]
            if not known_names:
                return "No one was recognized."

            return "Detected " + ", ".join(known_names)
        except Exception as exc:
            self.logger.error("Error during face recognition inference: %s", exc)
            return "An error occurred during facial recognition."
