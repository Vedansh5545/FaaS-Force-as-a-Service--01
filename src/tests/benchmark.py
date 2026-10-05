"""
Feeds every case in src/tests/cases/ to each technique and reports accuracy + speed.

Each case is 2 frames. Score >= Similarity_Threshold = accept (similar, reuse the answer); below = reject.
False-same is the dangerous error (we'd reuse a stale answer), so it's reported separately.

Usage: python src/tests/makeCases.py && python src/tests/benchmark.py
"""
import sys
import time
from pathlib import Path

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Tier1Caches"))
from ChangeBucket import ChangeBucket  # noqa: E402
from ColorFingerprint import ColorFingerprint  # noqa: E402
from OutlineComparison import OutlineComparison  # noqa: E402

Similarity_Threshold = 0.9
Cases_Folder = Path(__file__).parent / "cases"
Image_Types = {".jpg", ".jpeg", ".png"}

# each takes (frame1, frame2) and returns (score, accept?)
Techniques_To_Test = {
    "ColorFingerprint": ColorFingerprint.similar,
    "OutlineComparison": OutlineComparison.similar,
    "ChangeBucket": ChangeBucket.similar,
}


def loadCases():
    # a case = a folder holding exactly 2 images; sorted name order gives frame1, frame2
    for d in sorted(p for p in Cases_Folder.rglob("*") if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in Image_Types)
        if len(files) == 2:
            yield d.name, d.name.startswith("same"), [cv.imread(str(f)) for f in files]


def main():
    cases = list(loadCases())
    if not cases:
        sys.exit(f"no cases in {Cases_Folder} - run src/tests/makeCases.py first")

    stats = {m: {"right": 0, "falseSame": 0, "falseDiff": 0, "secs": 0.0} for m in Techniques_To_Test}
    print(f"{'case':38} {'expect':6} " + " ".join(f"{m:>20}" for m in Techniques_To_Test))
    for name, expectSame, (frame1, frame2) in cases:
        row = []
        for m, fn in Techniques_To_Test.items():
            t = time.perf_counter()
            score, same = fn(frame1, frame2, Similarity_Threshold)
            stats[m]["secs"] += time.perf_counter() - t
            ok = same == expectSame
            stats[m]["right"] += ok
            stats[m]["falseSame"] += same and not expectSame
            stats[m]["falseDiff"] += expectSame and not same
            row.append(f"{score:6.3f} {'ok' if ok else 'MISS':>4}")
        print(f"{name:38} {'same' if expectSame else 'diff':6} " + " ".join(f"{r:>20}" for r in row))

    print(f"\nthreshold {Similarity_Threshold}, {len(cases)} cases")
    print(f"{'method':20} {'accuracy':>9} {'false-same':>11} {'false-diff':>11} {'ms/compare':>11}")
    for m, s in stats.items():
        print(
            f"{m:20} {s['right'] / len(cases):9.1%} {s['falseSame']:11} {s['falseDiff']:11}"
            f" {1000 * s['secs'] / len(cases):11.3f}"
        )


if __name__ == "__main__":
    main()
