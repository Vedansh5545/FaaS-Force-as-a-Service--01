import cv2 as cv
import numpy as np

"""
idea: a hue x saturation histogram of the whole frame ignores where pixels are, so shake and sway
barely move it, but a new object entering shifts the colour distribution.
Unlike ColorFingerprint there are no grid cells: pure global distribution.
"""


def HistogramCompare(frame1, frame2, threshold=0.9):
    """Two raw BGR frames in -> (score, is the scene similar enough to reuse the cached answer?)."""

    def histogram(frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv.normalize(hist, hist).astype(np.float32)

    # correlation is in -1..1; negative just means "very different", so clip to 0-1
    score = cv.compareHist(histogram(frame1), histogram(frame2), cv.HISTCMP_CORREL)
    score = min(max(float(score), 0.0), 1.0)
    return score, score >= threshold
