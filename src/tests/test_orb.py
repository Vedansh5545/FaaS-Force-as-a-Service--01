import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OrbFeatureDetection.Orb import OrbDetection, comparison  # noqa: E402


class OrbImageTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_dir = Path(self.temp_dir.name)

        self.reference = self._make_image()
        self.similar = self._make_image(shift=8)
        self.different = self._make_different_image()
        self.blank = np.zeros((400, 400), dtype=np.uint8)

        self.reference_path = self._write_image("reference.png", self.reference)
        self.similar_path = self._write_image("similar.png", self.similar)
        self.different_path = self._write_image("different.png", self.different)
        self.blank_path = self._write_image("blank.png", self.blank)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_image(self, shift=0):
        image = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(image, (35 + shift, 35), (180 + shift, 180), 255, 4)
        cv2.circle(image, (285 + shift, 115), 60, 255, 4)
        cv2.line(image, (45 + shift, 295), (350 + shift, 295), 255, 5)
        cv2.putText(
            image,
            "ORB",
            (100 + shift, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            255,
            3,
        )
        return image

    def _make_different_image(self):
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (400, 400), dtype=np.uint8)

    def _write_image(self, name, image):
        path = self.image_dir / name
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def test_detection_returns_serialized_keypoints_and_descriptors(self):
        keypoints, descriptors = OrbDetection(str(self.reference_path))

        self.assertGreater(len(keypoints), 0)
        self.assertEqual(len(keypoints), len(descriptors))
        self.assertTrue(all(isinstance(keypoint, dict) for keypoint in keypoints))
        self.assertTrue(all(isinstance(descriptor, bytes) for descriptor in descriptors))
        self.assertTrue(all(len(descriptor) == 32 for descriptor in descriptors))
        self.assertEqual(
            {"x", "y", "size", "angle", "response", "octave"},
            set(keypoints[0]),
        )

    def test_detection_returns_empty_outputs_for_image_without_features(self):
        self.assertEqual(OrbDetection(str(self.blank_path)), ([], []))

    def test_detection_raises_for_missing_image(self):
        missing_path = self.image_dir / "missing.png"

        with self.assertRaisesRegex(FileNotFoundError, "Unable to read image"):
            OrbDetection(str(missing_path))

    def test_comparison_is_high_for_the_same_image(self):
        score = comparison(str(self.reference_path), str(self.reference_path))
        print(f"\nSame image score: {score:.3f}")

        self.assertGreaterEqual(score, 0.9)
        self.assertLessEqual(score, 1.0)

    def test_comparison_recognizes_a_small_image_shift(self):
        score = comparison(str(self.reference_path), str(self.similar_path))
        print(f"Similar image score: {score:.3f}")

        self.assertGreaterEqual(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_comparison_is_low_for_different_images(self):
        score = comparison(str(self.reference_path), str(self.different_path))
        print(f"Different image score: {score:.3f}")

        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.5)

    def test_comparison_returns_zero_when_an_image_has_no_descriptors(self):
        score = comparison(str(self.reference_path), str(self.blank_path))
        print(f"No-feature image score: {score:.3f}")

        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
