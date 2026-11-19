"""
Finds the ChangeBucket threshold on dataset/, same split and report as tuneOrbThreshold.py.

Usage: python src/tests/tuneChangeBucket.py [--recompute]
Rerun with --recompute after changing ChangeBucket.py or OutlineComparison.py.
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
from ChangeBucket import ChangeBucket  # noqa: E402


def score(pathA, pathB):
    # threshold=0 so the bucket never stops early: the score is the full fill level, and
    # score >= t then matches ChangeBucket(..., threshold=t)[1] for every t
    # ponytail: dataset only has the two end frames, so with no `between` frames this equals
    # OutlineComparison; pass the in-between frames from the source MP4s to tune the real bucket
    return ChangeBucket(cv.imread(pathA), cv.imread(pathB), threshold=0)[0]


if __name__ == "__main__":
    main(score, "changeBucketScores.csv", __doc__)
