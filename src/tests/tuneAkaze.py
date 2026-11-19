"""
Finds the Akaze threshold on dataset/, same split and report as tuneOrbThreshold.py.

Usage: python src/tests/tuneAkaze.py [--recompute]
Rerun with --recompute after changing Akaze.py.
"""
import sys

import cv2 as cv

from tuneOrbThreshold import Repo_Root, main

sys.path.insert(0, str(Repo_Root / "src" / "Tier1Caches"))
from Akaze import Akaze  # noqa: E402


def score(pathA, pathB):
    return Akaze(cv.imread(pathA), cv.imread(pathB))[0]


if __name__ == "__main__":
    main(score, "akazeScores.csv", __doc__)
