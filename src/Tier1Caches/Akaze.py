import cv2 as cv

"""
idea: ORB's keypoint matching, but with AKAZE keypoints. AKAZE builds its scale space with nonlinear
diffusion instead of Gaussian blur, so its keypoints survive motion blur, sensor noise and shake better.
Score = share of keypoints that find a clear match (Lowe's ratio test) in the other frame.
"""


def Akaze(frame1, frame2, threshold=0.9, ratio=0.75):
    """Two raw BGR frames in -> (score, is the scene similar enough to reuse the cached answer?)."""
    akaze = cv.AKAZE_create()
    _, des1 = akaze.detectAndCompute(cv.cvtColor(frame1, cv.COLOR_BGR2GRAY), None)
    _, des2 = akaze.detectAndCompute(cv.cvtColor(frame2, cv.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0, False

    # match from the smaller set: score stays in 0-1 and order stops mattering
    if len(des1) > len(des2):
        des1, des2 = des2, des1

    # AKAZE's default MLDB descriptor is binary, so Hamming distance like ORB
    pairs = cv.BFMatcher(cv.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [p for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    score = len(good) / len(des1)
    return score, score >= threshold
