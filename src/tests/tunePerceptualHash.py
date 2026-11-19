"""
Finds the PerceptualHash thresholds (PHash and BlockMeanHash) on dataset/, same split and report as
tuneOrbThreshold.py.

Usage: python src/tests/tunePerceptualHash.py [--recompute]
Rerun with --recompute after changing PerceptualHash.py.
"""
import sys

import cv2 as cv

from tuneOrbThreshold import Repo_Root, main

sys.path.insert(0, str(Repo_Root / "src" / "Tier1Caches"))
from PerceptualHash import PerceptualHash  # noqa: E402


def pHashScore(pathA, pathB):
    return PerceptualHash(cv.imread(pathA), cv.imread(pathB), method="phash")[0]


def blockMeanScore(pathA, pathB):
    return PerceptualHash(cv.imread(pathA), cv.imread(pathB), method="blockmean")[0]


if __name__ == "__main__":
    print("== PHash")
    main(pHashScore, "pHashScores.csv", __doc__)
    print("\n== BlockMeanHash")
    main(blockMeanScore, "blockMeanHashScores.csv", __doc__)
