"""
Finds the HistogramCompare threshold on dataset/, same split and report as tuneOrbThreshold.py.

Usage: python src/tests/tuneHistogramCompare.py [--recompute]
Rerun with --recompute after changing HistogramCompare.py.
"""
import sys

import cv2 as cv

from tuneOrbThreshold import Repo_Root, main

sys.path.insert(0, str(Repo_Root / "src" / "Tier1Caches"))
from HistogramCompare import HistogramCompare  # noqa: E402


def score(pathA, pathB):
    return HistogramCompare(cv.imread(pathA), cv.imread(pathB))[0]


if __name__ == "__main__":
    main(score, "histogramCompareScores.csv", __doc__)
