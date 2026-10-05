import cv2 as cv
import numpy as np

"""
idea: Trace each frame into a colouring-book outline (Canny edges), lay the new outline over the
last one and count how many lines don't match. Outlines ignore lighting, so a cloud passing
over doesn't look like a new scene.
"""

# how far the edge thresholds sit below/above the frame's median brightness (0.33 = -33% / +33%)
Edge_Threshold_Spread = 0.33
# how many pixels an edge can shift and still count as "didn't move"
Edge_Shift_Tolerance_Px = 1
# square used to thicken edges by Edge_Shift_Tolerance_Px on every side; derived, don't edit
Edge_Thicken_Kernel = np.ones((2 * Edge_Shift_Tolerance_Px + 1,) * 2, np.uint8)


class OutlineComparison:
    def __init__(self):
        self.last = None

    @staticmethod
    def edges(grey):
        medianBrightness = np.median(grey)
        lowThreshold = int((1 - Edge_Threshold_Spread) * medianBrightness)
        highThreshold = int(min(255, (1 + Edge_Threshold_Spread) * medianBrightness))
        return cv.Canny(grey, lowThreshold, highThreshold) > 0

    @staticmethod
    def diff(edges1, edges2):
        # fraction of the outline that moved: edge pixels with no edge nearby in the other frame / all edge pixels
        total = np.count_nonzero(edges1) + np.count_nonzero(edges2)
        if not total:
            return 0.0
        near1 = cv.dilate(edges1.view(np.uint8), Edge_Thicken_Kernel) > 0
        near2 = cv.dilate(edges2.view(np.uint8), Edge_Thicken_Kernel) > 0
        return (
            np.count_nonzero(edges1 & ~near2) + np.count_nonzero(edges2 & ~near1)
        ) / total

    @staticmethod
    def shrinkGrey(frame):
        # same 160x90 shrink as ColorFingerprint.resize; in the camera thread reuse that one instead
        return cv.cvtColor(
            cv.resize(frame, (160, 90), interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY
        )

    @staticmethod
    def similar(frame1, frame2, threshold=0.9):
        # two raw BGR frames in -> (score, similar enough?); score = fraction of outline that stayed put
        e1, e2 = (
            OutlineComparison.edges(OutlineComparison.shrinkGrey(f))
            for f in (frame1, frame2)
        )
        score = 1 - OutlineComparison.diff(e1, e2)
        return score, score >= threshold

    def update(self, grey):
        # grey = the already-shrunk 160x90 frame from ColorFingerprint.resize, don't shrink twice
        edges = self.edges(grey)
        d = 1.0 if self.last is None else self.diff(edges, self.last)
        self.last = edges
        return d


if __name__ == "__main__":
    frame = np.full((90, 160), 20, np.uint8)
    frame[30:60, 40:80] = 200
    moved = np.full((90, 160), 20, np.uint8)
    moved[30:60, 70:110] = 200

    oc = OutlineComparison()
    assert oc.update(frame) == 1.0  # nothing to compare yet
    assert oc.update(frame) == 0.0
    assert oc.update((frame * 0.6).astype(np.uint8)) < 0.05  # darker, same shape
    assert oc.update(moved) > 0.5
    assert OutlineComparison.diff(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 0.0
    print("OutlineComparison ok")
