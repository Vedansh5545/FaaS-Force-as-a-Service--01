import cv2 as cv
import numpy as np

"""
idea: blur hard (15x15) and shrink to 128x72 so micro-movement and edge jitter wash out, then SSIM
compares what is left: the overall layout of the scene.

SSIM is written out (same formula as cv2.quality.QualitySSIM_compute) because cv2.quality needs
opencv-contrib, which clashes with the opencv-python already installed.
"""


def ssim(a, b):
    # standard SSIM: 11x11 Gaussian window, sigma 1.5, constants for 8-bit images
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def window(x):
        return cv.GaussianBlur(x, (11, 11), 1.5)

    muA, muB = window(a), window(b)
    varA = window(a * a) - muA**2
    varB = window(b * b) - muB**2
    covariance = window(a * b) - muA * muB
    ssimMap = ((2 * muA * muB + c1) * (2 * covariance + c2)) / (
        (muA**2 + muB**2 + c1) * (varA + varB + c2)
    )
    return float(ssimMap.mean())


def BlurredSsim(frame1, frame2, threshold=0.9):
    """Two raw BGR frames in -> (score, is the scene similar enough to reuse the cached answer?)."""

    def prepare(frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (15, 15), 0)
        return cv.resize(blurred, (128, 72), interpolation=cv.INTER_AREA).astype(np.float64)

    # SSIM can dip below 0 for anti-correlated frames; that is just "different", so clip to 0-1
    score = min(max(ssim(prepare(frame1), prepare(frame2)), 0.0), 1.0)
    return score, score >= threshold
