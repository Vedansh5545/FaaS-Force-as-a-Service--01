import cv2
import matplotlib.pyplot as plt
import type


def OrbDetection(img: str) -> tuple[list, list]:
    # 1. Initialize outputs as native empty lists
    all_keypoints: list[dict[str, Any]] = []
    all_descriptors: list[bytes] = []

    orb = cv2.ORB_create(nfeatures=500)
    # process both images
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors:
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


def comparision():
    return
