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


def OutlineComparison(frame1, frame2, threshold=0.9):
    """Two raw BGR frames in -> (score, is the scene similar enough to reuse the cached answer?)."""

    def outline(frame):
        # shrink: INTER_AREA averages the pixels it throws away, the right choice for shrinking
        small = cv.resize(frame, (160, 90), interpolation=cv.INTER_AREA)
        grey = cv.cvtColor(small, cv.COLOR_BGR2GRAY)

        # Canny needs two brightness thresholds; hanging them off this frame's median keeps a
        # dim room and a bright one from tracing wildly different amounts of edge
        medianBrightness = np.median(grey)
        lowThreshold = int((1 - Edge_Threshold_Spread) * medianBrightness)
        highThreshold = int(min(255, (1 + Edge_Threshold_Spread) * medianBrightness))
        return cv.Canny(grey, lowThreshold, highThreshold) > 0

    edges1 = outline(frame1)
    edges2 = outline(frame2)

    # an edgeless pair (blank wall, pitch dark) has nothing to disagree about, so call it identical
    totalEdgePixels = np.count_nonzero(edges1) + np.count_nonzero(edges2)
    if not totalEdgePixels:
        return 1.0, True

    # thicken each outline by the shift tolerance, so an edge that moved a pixel still counts as put
    near1 = cv.dilate(edges1.view(np.uint8), Edge_Thicken_Kernel) > 0
    near2 = cv.dilate(edges2.view(np.uint8), Edge_Thicken_Kernel) > 0

    # an edge "moved" if the other frame has no edge anywhere near it
    moved = np.count_nonzero(edges1 & ~near2) + np.count_nonzero(edges2 & ~near1)

    # score = the fraction of the outline that stayed put
    score = 1 - moved / totalEdgePixels
    return score, score >= threshold


if __name__ == "__main__":
    frame = np.full((90, 160, 3), 20, np.uint8)
    frame[30:60, 40:80] = 200
    moved = np.full((90, 160, 3), 20, np.uint8)
    moved[30:60, 70:110] = 200

    assert OutlineComparison(frame, frame) == (1.0, True)
    assert OutlineComparison(frame, (frame * 0.6).astype(np.uint8))[1]  # darker, same shape
    assert OutlineComparison(frame, moved)[0] < 0.5
    assert OutlineComparison(np.zeros_like(frame), np.zeros_like(frame)) == (1.0, True)
    print("OutlineComparison ok")
