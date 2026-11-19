"""
Finds the BlurredSsim threshold on dataset/, same split and report as tuneOrbThreshold.py.

Usage: python src/tests/tuneBlurredSsim.py [--recompute]
Rerun with --recompute after changing BlurredSsim.py.
"""
import sys

import cv2 as cv

from tuneOrbThreshold import Repo_Root, main

sys.path.insert(0, str(Repo_Root / "src" / "Tier1Caches"))
from BlurredSsim import BlurredSsim  # noqa: E402


def score(pathA, pathB):
    return BlurredSsim(cv.imread(pathA), cv.imread(pathB))[0]


if __name__ == "__main__":
    main(score, "blurredSsimScores.csv", __doc__)
