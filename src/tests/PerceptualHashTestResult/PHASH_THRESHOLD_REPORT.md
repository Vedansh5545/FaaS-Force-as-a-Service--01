# PHash Threshold Report

## Executive summary

The PHash image-comparison threshold was tuned on 100 labeled image pairs. The best threshold for balanced F1 score was **0.84**. On the held-out test set, it achieved **75.0% accuracy**, **70.0% recall**, and an **F1 score of 0.737**.

The threshold produced **2 false cache hits** and **3 false misses**. Because a false cache hit can reuse stale perception results, `0.84` should be treated as an experimental, F1-optimized threshold rather than a final production safety threshold.

## Objective

Determine a PHash similarity-score threshold (a 64-bit DCT perceptual hash (1 - normalised Hamming distance)) for deciding whether a previous perception result can be reused:

- **Cache hit:** the scene is sufficiently unchanged; reuse the result.
- **Cache miss:** the scene changed; run fresh perception.

The critical error is a false hit because it may reuse stale scene information.

## Test performed

The test was run by [`src/tests/tunePerceptualHash.py`](../tunePerceptualHash.py) using the labeled dataset (100 pairs, 50 cache hits, 50 cache misses, 10 source scenes). For every pair the script computed a PHash similarity score between 0.0 and 1.0, classified the pair as a cache hit when the score was at least the candidate threshold, and compared that with the label.

The computed scores are stored in [`pHashScores.csv`](pHashScores.csv).

## Fine-tuning method

The pairs were split by scene:

- **80 tuning pairs:** used to select the threshold.
- **20 held-out test pairs:** from `MOT17-11` and `MOT17-13`, reserved for final evaluation.

Every threshold from `0.00` through `1.00` in steps of `0.01` was scored by F1 on the 80 tuning pairs; the highest F1 won, ties going to the higher threshold.

## Results

Selected threshold:

```text
0.84
```

Held-out test results:

| Metric | Result |
|---|---:|
| Accuracy | 75.0% |
| Precision | 77.8% |
| Recall | 70.0% |
| F1 score | 0.737 |
| False cache hits | 2 |
| False cache misses | 3 |

The always-cache-hit baseline achieved 50.0% accuracy and an F1 score of 0.667.

## Threshold risk analysis

Held-out test set, other thresholds:

| Threshold | False hits | False misses |
|---:|---:|---:|
| 0.10 | 10 | 0 |
| 0.20 | 10 | 0 |
| 0.30 | 10 | 0 |
| 0.40 | 9 | 0 |
| 0.50 | 9 | 0 |
| 0.60 | 9 | 0 |
| 0.70 | 5 | 0 |
| 0.80 | 3 | 3 |
| 0.84 | 2 | 3 |
| 0.90 | 1 | 7 |

For the comparison against every other technique, run [`src/tests/benchmark.py`](../benchmark.py).

## Reproduction

Activate the project virtual environment and run:

```bash
source .venv/bin/activate
python src/tests/tunePerceptualHash.py --recompute
```

The visualization is available at [`pHashThreshold.png`](pHashThreshold.png).
