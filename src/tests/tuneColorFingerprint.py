"""
Finds the ColorFingerprint threshold on dataset/, same split and report as tuneOrbThreshold.py.

Usage: python src/tests/tuneColorFingerprint.py [--recompute]
Rerun with --recompute after changing ColorFingerprint.py.
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
from ColorFingerprint import ColorFingerprint  # noqa: E402


def score(pathA, pathB):
    return ColorFingerprint(cv.imread(pathA), cv.imread(pathB))[0]


if __name__ == "__main__":
    main(score, "colorFingerprintScores.csv", __doc__)
