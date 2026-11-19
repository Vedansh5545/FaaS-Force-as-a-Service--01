"""
Ranks every frame-comparison technique by how close it gets to 0 false hits AND 0 false misses.

False hit = we reuse the cached answer but the scene changed (dangerous: stale answer).
False miss = we recompute although nothing important changed (safe, just wasted work).

Per technique, on the 100 labelled pairs in dataset/:
  test FH/FM    threshold tuned on the 80 tune pairs, errors on the 20 held-out pairs (honest)
  0-FH misses   lowest threshold with 0 false hits on all 100 pairs -> false misses it costs
  0-FM hits     highest threshold with 0 false misses on all 100 pairs -> false hits it costs
  best errors   fewest FH+FM any single threshold can reach on all 100 pairs (0 = perfect)
  margin        lowest hit score - highest miss score; > 0 means one threshold gives 0/0
  AUC           how well the score ranks hits above misses, threshold-free (1 = perfect)
The last four are in-sample: they say if 0/0 is possible at all, not what a deployed threshold gets.
Then the tuned threshold is applied to the synthetic cases in src/tests/cases/ (if present),
which also gives the speed (ms per comparison, image decode included).

Ranking: perfect separation first, then fewest false misses at 0 false hits, then test errors.

Usage: python src/tests/benchmark.py [--recompute]
--recompute rescores the dataset (otherwise the score CSVs saved by the tuners are reused).
"""
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import tuneAkaze
import tuneBlurredSsim
import tuneChangeBucket
import tuneColorFingerprint
import tuneHistogramCompare
import tuneOutlineComparison
import tunePerceptualHash
from tuneOrbThreshold import Repo_Root, Test_Scenes, bestThreshold, computeScores, errorCounts, loadScores

Cases_Folder = Path(__file__).parent / "cases"
Image_Types = {".jpg", ".jpeg", ".png"}


def orbScore(pathA, pathB):
    import sys

    sys.path.insert(0, str(Repo_Root / "src" / "OrbFeatureDetection"))
    from Orb import comparison

    return comparison(pathA, pathB)


# name -> (scorer(pathA, pathB) -> 0-1 similarity, score CSV the tuner saves in artifacts/)
Techniques_To_Test = {
    "ORB": (orbScore, "orbScores.csv"),
    "AKAZE": (tuneAkaze.score, "akazeScores.csv"),
    "PHash": (tunePerceptualHash.pHashScore, "pHashScores.csv"),
    "BlockMeanHash": (tunePerceptualHash.blockMeanScore, "blockMeanHashScores.csv"),
    "BlurredSSIM": (tuneBlurredSsim.score, "blurredSsimScores.csv"),
    "HistogramCompare": (tuneHistogramCompare.score, "histogramCompareScores.csv"),
    "ColorFingerprint": (tuneColorFingerprint.score, "colorFingerprintScores.csv"),
    "OutlineComparison": (tuneOutlineComparison.score, "outlineComparisonScores.csv"),
    "ChangeBucket": (tuneChangeBucket.score, "changeBucketScores.csv"),
}


def separation(scores, matches):
    """In-sample: what the best possible threshold can do on these pairs."""
    hits, misses = scores[matches == 1], scores[matches == 0]
    # just above the top miss -> 0 false hits; every hit at or below it becomes a false miss
    zeroFhMisses = int(np.sum(hits <= misses.max()))
    # at the lowest hit -> 0 false misses; every miss at or above it becomes a false hit
    zeroFmHits = int(np.sum(misses >= hits.min()))
    # try every score as the threshold (plus "above everything")
    candidates = np.append(np.unique(scores), np.inf)
    bestErrors = min(int(np.sum(misses >= t) + np.sum(hits < t)) for t in candidates)
    return zeroFhMisses, zeroFmHits, bestErrors, float(hits.min() - misses.max())


def loadCases():
    # a case = a folder holding exactly 2 images; folder name starts with same_ or diff_
    for d in sorted(p for p in Cases_Folder.rglob("*") if p.is_dir()):
        files = sorted(str(f) for f in d.iterdir() if f.suffix.lower() in Image_Types)
        if len(files) == 2:
            yield d.name.startswith("same"), files


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="rescore the dataset")
    recompute = parser.parse_args().recompute
    # self-check: hits .9 .5, misses .6 .1 -> 0 FH costs 1 miss, 0 FM costs 1 hit, best 1 error, margin -.1
    zeroFhMisses, zeroFmHits, bestErrors, margin = separation(np.array([0.9, 0.5, 0.6, 0.1]), np.array([1, 1, 0, 0]))
    assert (zeroFhMisses, zeroFmHits, bestErrors) == (1, 1, 1) and abs(margin + 0.1) < 1e-9
    cases = list(loadCases())

    results = []
    for name, (scorePair, scoresFileName) in Techniques_To_Test.items():
        scoresFile = Repo_Root / "artifacts" / scoresFileName
        if recompute or not scoresFile.exists():
            print(f"scoring dataset with {name}...")
            computeScores(scorePair, scoresFile)
        rows = loadScores(scoresFile)
        assert len(rows) == 100, f"{name}: expected 100 scored pairs"

        scores = np.array([r["score"] for r in rows])
        matches = np.array([r["match"] for r in rows])
        isTest = np.array([r["scene_id"] in Test_Scenes for r in rows])
        threshold = bestThreshold(scores[~isTest], matches[~isTest])
        testFh, testFm = errorCounts(matches[isTest], (scores[isTest] >= threshold).astype(int))

        synthFh = synthFm = 0
        secs = 0.0
        for expectSame, (pathA, pathB) in cases:
            start = time.perf_counter()
            same = scorePair(pathA, pathB) >= threshold
            secs += time.perf_counter() - start
            synthFh += same and not expectSame
            synthFm += expectSame and not same

        results.append(
            {
                "name": name,
                "threshold": threshold,
                "testFh": testFh,
                "testFm": testFm,
                **dict(zip(("zeroFhMisses", "zeroFmHits", "bestErrors", "margin"), separation(scores, matches))),
                "auc": roc_auc_score(matches, scores),
                "synthFh": synthFh,
                "synthFm": synthFm,
                "ms": 1000 * secs / len(cases) if cases else float("nan"),
            }
        )

    results.sort(key=lambda r: (r["margin"] <= 0, r["zeroFhMisses"], r["testFh"] + r["testFm"], -r["auc"]))

    print(f"\ndataset: 100 pairs (50 hits / 50 misses), test scenes {sorted(Test_Scenes)}; synthetic cases: {len(cases)}")
    header = (
        f"{'rank':4} {'technique':18} {'thr':>5} {'testFH':>6} {'testFM':>6} {'0-FH misses':>11} "
        f"{'0-FM hits':>9} {'best errs':>9} {'margin':>7} {'AUC':>6} {'synFH':>5} {'synFM':>5} {'ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:4} {r['name']:18} {r['threshold']:5.2f} {r['testFh']:6} {r['testFm']:6} {r['zeroFhMisses']:11} "
            f"{r['zeroFmHits']:9} {r['bestErrors']:9} {r['margin']:7.3f} {r['auc']:6.3f} "
            f"{r['synthFh']:5} {r['synthFm']:5} {r['ms']:7.1f}"
        )

    perfect = [r["name"] for r in results if r["margin"] > 0]
    best = results[0]
    if perfect:
        print(f"\n0 false hits + 0 false misses reachable with: {', '.join(perfect)}")
    else:
        print(
            "\nNo technique reaches 0 false hits + 0 false misses on the dataset with one threshold."
            f"\nBest: {best['name']} - at 0 false hits it still costs {best['zeroFhMisses']}/50 false misses"
            f" (fewest possible errors {best['bestErrors']}/100, AUC {best['auc']:.3f})."
        )
    if not cases:
        print(f"(no synthetic cases in {Cases_Folder} - run src/tests/makeCases.py for the speed column)")


if __name__ == "__main__":
    main()
