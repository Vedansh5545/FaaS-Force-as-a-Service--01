"""
Finds the OutlineComparison threshold on dataset/, same split and report as tuneOrbThreshold.py.

Usage: python src/tests/tuneOutlineComparison.py [--recompute]
Rerun with --recompute after changing OutlineComparison.py.
"""
import sys
from pathlib import Path

import cv2 as cv

Repo_Root = Path(__file__).resolve().parents[2]
if str(Repo_Root) not in sys.path:
    sys.path.insert(0, str(Repo_Root))
try:
    from src.tests.tuneOrbThreshold import Repo_Root, main
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    from tuneOrbThreshold import Repo_Root, main

sys.path.insert(0, str(Repo_Root / "src" / "Tier1Caches"))
from OutlineComparison import OutlineComparison  # noqa: E402


def score(pathA, pathB):
    return OutlineComparison(cv.imread(pathA), cv.imread(pathB))[0]


if __name__ == "__main__":
    main(score, "outlineComparisonScores.csv", __doc__)
