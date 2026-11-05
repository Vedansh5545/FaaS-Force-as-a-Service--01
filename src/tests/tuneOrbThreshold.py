"""
Finds the ORB similarity threshold on dataset/ and reports it once on a held-out test split.

Positive class = match = CACHE HIT. labels.csv uses the opposite: label_id 0 = NOT CHANGED = CACHE HIT.
A false hit (FP) is the dangerous error, since we'd reuse a stale answer.

Usage: python src/tests/tuneOrbThreshold.py [--recompute]
Scores are saved outside dataset/ because verify_dataset.py rejects unexpected files there.
Rerun with --recompute after changing Orb.py.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

Repo_Root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Repo_Root / "src" / "OrbFeatureDetection"))
from Orb import comparison  # noqa: E402

Dataset_Folder = Repo_Root / "dataset"
Scores_File = Repo_Root / "artifacts" / "orbScores.csv"
Thresholds = np.linspace(0, 1, 101)
# dataset/README.md: keep scenes and source families together. These two share a family with no
# tuning scene, make exactly 20% of the pairs, and include hard misses and a vehicle viewpoint.
Test_Scenes = {"MOT17-11", "MOT17-13"}
Score_Columns = ["pair_id", "scene_id", "difficulty", "match", "score"]


def computeScores():
    with open(Dataset_Folder / "labels.csv", encoding="utf-8", newline="") as labelsFile:
        pairs = list(csv.DictReader(labelsFile))

    rows = []
    for pair in pairs:
        score = comparison(
            str(Dataset_Folder / pair["image_a_path"]),
            str(Dataset_Folder / pair["image_b_path"]),
        )
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "scene_id": pair["scene_id"],
                "difficulty": pair["difficulty"],
                "match": 1 - int(pair["label_id"]),
                "score": score,
            }
        )

    Scores_File.parent.mkdir(exist_ok=True)
    with open(Scores_File, "w", encoding="utf-8", newline="") as scoresFile:
        writer = csv.DictWriter(scoresFile, Score_Columns)
        writer.writeheader()
        writer.writerows(rows)


def loadScores():
    with open(Scores_File, encoding="utf-8", newline="") as scoresFile:
        return [
            {**row, "match": int(row["match"]), "score": float(row["score"])}
            for row in csv.DictReader(scoresFile)
        ]


def bestThreshold(scores, matches):
    f1s = [f1_score(matches, scores >= t, zero_division=0) for t in Thresholds]
    # argmax takes the first of a tie, so search from the top to keep the strictest threshold
    return Thresholds[len(Thresholds) - 1 - int(np.argmax(f1s[::-1]))]


def errorCounts(matches, predictions):
    # returns (false hits, false misses)
    _, falseHits, falseMisses, _ = confusion_matrix(matches, predictions, labels=[0, 1]).ravel()
    return int(falseHits), int(falseMisses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="rerun ORB instead of reusing saved scores")
    if parser.parse_args().recompute or not Scores_File.exists():
        computeScores()

    rows = loadScores()
    tune = [r for r in rows if r["scene_id"] not in Test_Scenes]
    test = [r for r in rows if r["scene_id"] in Test_Scenes]

    assert len(rows) == 100 and sum(r["match"] for r in rows) == 50, "expected 100 pairs, 50 hits"
    assert len(test) == 20, "test split must be 20% of the pairs"
    assert not {r["scene_id"] for r in tune} & {r["scene_id"] for r in test}, "scene in both splits"
    assert all(0 <= r["score"] <= 1 for r in rows), "similarity score outside 0-1"

    threshold = bestThreshold(
        np.array([r["score"] for r in tune]), np.array([r["match"] for r in tune])
    )
    testScores = np.array([r["score"] for r in test])
    testMatches = np.array([r["match"] for r in test])
    predictions = (testScores >= threshold).astype(int)
    alwaysHit = np.ones_like(testMatches)
    falseHits, falseMisses = errorCounts(testMatches, predictions)

    print(f"tune pairs {len(tune)}, test pairs {len(test)}, test scenes {sorted(Test_Scenes)}")
    print(f"threshold {threshold:.2f} (chosen on tune pairs only)")
    print(
        f"test accuracy {accuracy_score(testMatches, predictions):.3f}, "
        f"precision {precision_score(testMatches, predictions, zero_division=0):.3f}, "
        f"recall {recall_score(testMatches, predictions, zero_division=0):.3f}, "
        f"F1 {f1_score(testMatches, predictions, zero_division=0):.3f}"
    )
    print(f"test false hits {falseHits}, false misses {falseMisses}")
    print(
        f"always-hit baseline: accuracy {accuracy_score(testMatches, alwaysHit):.3f}, "
        f"F1 {f1_score(testMatches, alwaysHit):.3f}"
    )
    for difficulty in ("regular", "hard"):
        chosen = np.array([r["difficulty"] == difficulty for r in test])
        hits, misses = errorCounts(testMatches[chosen], predictions[chosen])
        print(f"  {difficulty:8} pairs {int(chosen.sum()):2}, false hits {hits}, false misses {misses}")


if __name__ == "__main__":
    main()
