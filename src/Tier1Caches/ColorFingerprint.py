import cv2 as cv
import numpy as np

"""
idea: Cut the frame into 12 squares like a chocolate bar. In each square, count how much red, green, blue, and so on. Write those count on a row and that row is the fingerprint of the frame

"""

Color_Buckets = ["black", "gray", "red", "yellow", "green", "cyan", "blue", "magenta"]


def ColorFingerprint(frame1, frame2, threshold=0.9):
    """Two raw BGR frames in -> (score, is the scene similar enough to reuse the cached answer?)."""

    def fingerprint(frame):
        # shrink: INTER_AREA averages the pixels it throws away, the right choice for shrinking
        small = cv.resize(frame, (160, 90), interpolation=cv.INTER_AREA)

        # HSV puts colour in one channel instead of spread across three
        hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
        hue = hsv[..., 0].astype(np.int32)
        saturation = hsv[..., 1]
        brightness = hsv[..., 2]

        labels = 2 + ((hue + 15) // 30) % 6

        # washed-out and dark pixels have no trustworthy hue, so they get their own buckets
        labels[saturation < 50] = Color_Buckets.index("gray")
        labels[brightness < 50] = Color_Buckets.index("black")

        # cut into cells, left to right, top to bottom
        cells = [cell for row in np.vsplit(labels, 3) for cell in np.hsplit(row, 4)]

        # count each cell's colours and lay the counts end to end: 12 cells x 8 buckets = 96 floats
        counts = [np.bincount(c.ravel(), minlength=len(Color_Buckets)) for c in cells]
        row = np.concatenate(counts).astype(np.float32)

        # unit length, so two frames of different brightness or detail still compare fairly
        return row / np.linalg.norm(row)

    # cosine similarity: ~1 for the same scene, ~0 for different ones
    score = float(np.dot(fingerprint(frame1), fingerprint(frame2)))
    return score, score >= threshold
