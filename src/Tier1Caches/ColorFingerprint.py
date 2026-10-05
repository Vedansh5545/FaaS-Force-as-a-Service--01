import cv2 as cv
import numpy as np

"""
idea: Cut the frame into 12 squares like a chocolate bar. In each square, count how much red, green, blue, and so on. Write those count on a row and that row is the fingerprint of the frame

"""

Color_Buckets = ["black", "gray", "red", "yellow", "green", "cyan", "blue", "magenta"]
# calibration knobs: below these, hue is meaningless (too dark / too washed out)
Dark_Brightness_Cutoff = 50
Gray_Saturation_Cutoff = 50


class ColorFingerprint:
    def __init__(self, image1, image2):
        frame1 = cv.imread(image1)
        frame2 = cv.imread(image2)
        if frame1 is None or frame2 is None:
            raise FileNotFoundError(
                f"could not read image: {image1 if frame1 is None else image2}"
            )
        self.frame1 = self.convertHsv(self.resize(frame1))
        self.frame2 = self.convertHsv(self.resize(frame2))
        self.fingerprint1 = self.fingerprint(self.frame1)
        self.fingerprint2 = self.fingerprint(self.frame2)
        self.similarity = self.compare(self.fingerprint1, self.fingerprint2)

    @staticmethod
    def resize(frame):
        # resize image to 160 x 90; INTER_AREA averages pixels, best for shrinking
        # the camera thread shrinks once and reuses the result for OutlineComparison too
        return cv.resize(frame, (160, 90), interpolation=cv.INTER_AREA)

    @staticmethod
    def convertHsv(frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    @staticmethod
    def slicing(frame):
        # 3 rows down x 4 columns across = 12 pieces, left-to-right, top-to-bottom
        return [cell for row in np.vsplit(frame, 3) for cell in np.hsplit(row, 4)]

    @staticmethod
    def label(hsv):
        # each pixel -> index into Color_Buckets
        h = hsv[..., 0].astype(np.int32)
        # OpenCV hue is 0-179: 6 bins of 30, shifted by 15 so red wraps around 0
        labels = 2 + ((h + 15) // 30) % 6
        labels[hsv[..., 1] < Gray_Saturation_Cutoff] = 1
        labels[hsv[..., 2] < Dark_Brightness_Cutoff] = 0
        return labels

    @staticmethod
    def fingerprint(hsv):
        # 12 cells x 8 buckets = 96 float32 = 384 bytes, unit length
        cells = ColorFingerprint.slicing(ColorFingerprint.label(hsv))
        fp = np.concatenate(
            [np.bincount(c.ravel(), minlength=len(Color_Buckets)) for c in cells]
        ).astype(np.float32)
        return fp / np.linalg.norm(fp)

    @staticmethod
    def compare(fp1, fp2):
        # cosine similarity: ~1 same scene, ~0 different
        return float(np.dot(fp1, fp2))

    @staticmethod
    def similar(frame1, frame2, threshold=0.9):
        # two raw BGR frames in -> (score, similar enough?)
        fp1, fp2 = (
            ColorFingerprint.fingerprint(ColorFingerprint.convertHsv(ColorFingerprint.resize(f)))
            for f in (frame1, frame2)
        )
        score = ColorFingerprint.compare(fp1, fp2)
        return score, score >= threshold


if __name__ == "__main__":
    def fp(bgr):
        return ColorFingerprint.fingerprint(ColorFingerprint.convertHsv(ColorFingerprint.resize(bgr)))

    red = np.zeros((720, 1280, 3), np.uint8); red[:] = (0, 0, 200)
    blue = np.zeros((720, 1280, 3), np.uint8); blue[:] = (200, 0, 0)
    left = np.zeros((720, 1280, 3), np.uint8); left[:, :640] = (0, 0, 200)
    right = np.zeros((720, 1280, 3), np.uint8); right[:, 640:] = (0, 0, 200)

    assert fp(red).shape == (96,) and fp(red).nbytes == 384
    assert abs(ColorFingerprint.compare(fp(red), fp(red)) - 1) < 1e-5
    assert ColorFingerprint.compare(fp(red), fp(blue)) < 0.01
    assert ColorFingerprint.compare(fp(red), fp(red // 2)) > 0.99  # darker, same hue
    assert ColorFingerprint.compare(fp(left), fp(right)) < 0.5  # position matters
    print("ColorFingerprint ok")
