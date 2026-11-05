import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def OrbDetection(img: str) -> tuple[list, list]:
    # 1. Initialize outputs as native empty lists
    all_keypoints: list[dict[str, Any]] = []
    all_descriptors: list[bytes] = []

    orb = cv2.ORB_create(nfeatures=500)
    # process both images
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is not None:
        for kp in keypoints:
            all_keypoints.append(  # convert C++ keypoints object into Python dict
                {
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                    "response": float(kp.response),
                    "octave": int(kp.octave),
                }
            )

        # convert 2D NumpyArray rows into a list of native python bytes objects
        for row in descriptors:
            all_descriptors.append(bytes(row.tolist()))
    return all_keypoints, all_descriptors


def comparison(img1: str, img2: str, ratio: float = 0.75) -> float:
    # return a score from 0 -> 1 with 0 means they don't look the same and 1 means they look alike
    _, des1 = OrbDetection(img1)
    _, des2 = OrbDetection(img2)
    if not des1 or not des2:
        return 0.0

    # rebuild the (N, 32) uint8 arrays OpenCV expects from the bytes lists
    d1 = np.frombuffer(b"".join(des1), dtype=np.uint8).reshape(len(des1), -1)
    d2 = np.frombuffer(b"".join(des2), dtype=np.uint8).reshape(len(des2), -1)

    # match from the smaller set: score stays in 0-1 and order stops mattering
    if len(d1) > len(d2):
        d1, d2 = d2, d1

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )  # bruteforce matching using Hamming distance
    pairs = matcher.knnMatch(d1, d2, k=2)  # K-nearest-neighbors search
    good = [
        p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance
    ] #Lowe's ratio test
    return len(good) / min(len(d1), len(d2))
