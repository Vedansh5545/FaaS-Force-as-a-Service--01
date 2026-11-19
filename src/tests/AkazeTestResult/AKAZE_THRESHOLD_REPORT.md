# AKAZE Threshold Report

## Executive summary

The AKAZE image-comparison threshold was tuned on 100 labeled image pairs. The best threshold for balanced F1 score was **0.36**. On the held-out test set, it achieved **80.0% accuracy**, **80.0% recall**, and an **F1 score of 0.800**.

The threshold produced **2 false cache hits** and **2 false misses**. Because a false cache hit can reuse stale perception results, `0.36` should be treated as an experimental, F1-optimized threshold rather than a final production safety threshold.

## Objective

Determine a AKAZE similarity-score threshold (AKAZE keypoint matching with Lowe's ratio test) for deciding whether a previous perception result can be reused:

- **Cache hit:** the scene is sufficiently unchanged; reuse the result.
- **Cache miss:** the scene changed; run fresh perception.

The critical error is a false hit because it may reuse stale scene information.

## Test performed

The test was run by [`src/tests/tuneAkaze.py`](../tuneAkaze.py) using the labeled dataset (100 pairs, 50 cache hits, 50 cache misses, 10 source scenes). For every pair the script computed a AKAZE similarity score between 0.0 and 1.0, classified the pair as a cache hit when the score was at least the candidate threshold, and compared that with the label.

The computed scores are stored in [`akazeScores.csv`](akazeScores.csv).

## Fine-tuning method

The pairs were split by scene:

- **80 tuning pairs:** used to select the threshold.
- **20 held-out test pairs:** from `MOT17-11` and `MOT17-13`, reserved for final evaluation.

Every threshold from `0.00` through `1.00` in steps of `0.01` was scored by F1 on the 80 tuning pairs; the highest F1 won, ties going to the higher threshold.

## Results

Selected threshold:

```text
0.36
```

Held-out test results:

| Metric | Result |
|---|---:|
| Accuracy | 80.0% |
| Precision | 80.0% |
| Recall | 80.0% |
| F1 score | 0.800 |
| False cache hits | 2 |
| False cache misses | 2 |

The always-cache-hit baseline achieved 50.0% accuracy and an F1 score of 0.667.

## Threshold risk analysis

Held-out test set, other thresholds:

| Threshold | False hits | False misses |
|---:|---:|---:|
| 0.10 | 9 | 0 |
| 0.20 | 5 | 1 |
| 0.30 | 2 | 1 |
| 0.36 | 2 | 2 |
| 0.40 | 2 | 3 |
| 0.50 | 2 | 7 |
| 0.60 | 2 | 8 |
| 0.70 | 0 | 9 |
| 0.80 | 0 | 10 |
| 0.90 | 0 | 10 |

For the comparison against every other technique, run [`src/tests/benchmark.py`](../benchmark.py).

## Reproduction

Activate the project virtual environment and run:

```bash
source .venv/bin/activate
python src/tests/tuneAkaze.py --recompute
```

The visualization is available at [`akazeThreshold.png`](akazeThreshold.png).
