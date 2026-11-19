import cv2 as cv
import numpy as np

"""
idea: shrink the frame until only its low-frequency structure is left, turn that into a row of bits,
and count how many bits differ. Noise, vibration and compression artefacts vanish in the shrink.

Same algorithms as cv2.img_hash.PHash / BlockMeanHash, written out because img_hash needs
opencv-contrib, which clashes with the opencv-python already installed.
"""


def pHashBits(frame):
    # 32x32 grey -> DCT -> keep the 8x8 lowest frequencies -> 1 bit each: above the median or not
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    small = cv.resize(gray, (32, 32), interpolation=cv.INTER_AREA).astype(np.float32)
    low = cv.dct(small)[:8, :8].ravel()
    return low > np.median(low)


def blockMeanBits(frame):
    # 256x256 grey cut into 16x16 blocks -> 1 bit per block: brighter than the median block or not
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    small = cv.resize(gray, (256, 256), interpolation=cv.INTER_AREA).astype(np.float32)
    means = small.reshape(16, 16, 16, 16).mean(axis=(1, 3)).ravel()
    return means > np.median(means)


Hashes = {"phash": pHashBits, "blockmean": blockMeanBits}


def PerceptualHash(frame1, frame2, threshold=0.9, method="phash"):
    """Two raw BGR frames in -> (score, is the scene similar enough to reuse the cached answer?)."""
    bits1, bits2 = Hashes[method](frame1), Hashes[method](frame2)
    # 1 - normalised Hamming distance: 1 = identical hash, ~0.5 = unrelated frames
    score = 1.0 - float(np.count_nonzero(bits1 != bits2)) / bits1.size
    return score, score >= threshold
