# HistogramCompare Threshold Report

## Executive summary

The HistogramCompare image-comparison threshold was tuned on 100 labeled image pairs. The best threshold for balanced F1 score was **0.95**. On the held-out test set, it achieved **75.0% accuracy**, **80.0% recall**, and an **F1 score of 0.762**.

The threshold produced **3 false cache hits** and **2 false misses**. Because a false cache hit can reuse stale perception results, `0.95` should be treated as an experimental, F1-optimized threshold rather than a final production safety threshold.

## Objective

Determine a HistogramCompare similarity-score threshold (hue x saturation histogram correlation) for deciding whether a previous perception result can be reused:

- **Cache hit:** the scene is sufficiently unchanged; reuse the result.
- **Cache miss:** the scene changed; run fresh perception.

The critical error is a false hit because it may reuse stale scene information.

## Test performed

The test was run by [`src/tests/tuneHistogramCompare.py`](../tuneHistogramCompare.py) using the labeled dataset (100 pairs, 50 cache hits, 50 cache misses, 10 source scenes). For every pair the script computed a HistogramCompare similarity score between 0.0 and 1.0, classified the pair as a cache hit when the score was at least the candidate threshold, and compared that with the label.

The computed scores are stored in [`histogramCompareScores.csv`](histogramCompareScores.csv).

## Fine-tuning method

The pairs were split by scene:

- **80 tuning pairs:** used to select the threshold.
- **20 held-out test pairs:** from `MOT17-11` and `MOT17-13`, reserved for final evaluation.

Every threshold from `0.00` through `1.00` in steps of `0.01` was scored by F1 on the 80 tuning pairs; the highest F1 won, ties going to the higher threshold.

## Results

Selected threshold:

```text
0.95
```

Held-out test results:

| Metric | Result |
|---|---:|
| Accuracy | 75.0% |
| Precision | 72.7% |
| Recall | 80.0% |
| F1 score | 0.762 |
| False cache hits | 3 |
| False cache misses | 2 |

The always-cache-hit baseline achieved 50.0% accuracy and an F1 score of 0.667.

## Threshold risk analysis

Held-out test set, other thresholds:

| Threshold | False hits | False misses |
|---:|---:|---:|
| 0.10 | 10 | 0 |
| 0.20 | 10 | 0 |
| 0.30 | 10 | 0 |
| 0.40 | 10 | 0 |
| 0.50 | 10 | 0 |
| 0.60 | 10 | 0 |
| 0.70 | 10 | 0 |
| 0.80 | 9 | 0 |
| 0.90 | 6 | 0 |
| 0.95 | 3 | 2 |

For the comparison against every other technique, run [`src/tests/benchmark.py`](../benchmark.py).

## Reproduction

Activate the project virtual environment and run:

```bash
source .venv/bin/activate
python src/tests/tuneHistogramCompare.py --recompute
```

The visualization is available at [`histogramCompareThreshold.png`](histogramCompareThreshold.png).
