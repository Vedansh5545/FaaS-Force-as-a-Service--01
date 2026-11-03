import cv2
import numpy as np
from pathlib import Path
from typing import Any

def OrbDetection(img: str) -> tuple[list, list]:
    # 1. Initialize outputs as native empty lists
    all_keypoints: list[dict[str, Any]] = []
    all_descriptors: list[bytes] = []

    orb = cv2.ORB_create(nfeatures=500)
    # process both images
    image_path = Path(img)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path.resolve()}")

    keypoints, descriptors = orb.detectAndCompute(image, None)

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

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [
        p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance
    ]
    return len(good) / min(len(d1), len(d2))


def main():
    folder = Path(__file__).resolve().parents[2] / "testing" / "dataset" / "images"
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(
        image for image in folder.iterdir()
        if image.is_file() and image.suffix.lower() in image_extensions
    )

    if len(images) < 2:
        raise RuntimeError(f"At least two images are required in {folder}")

    scores = []
    for image1, image2 in zip(images, images[1:]):
        score = comparison(str(image1), str(image2))
        scores.append(score)
        print(
            f"image1: {image1.name} | image2: {image2.name} | score: {score:.3f}"
        )

    print(f"\nCompared {len(images)} images ({len(scores)} pairs)")


if __name__ == "__main__":
    main()