# OutlineComparison Threshold Report

## Executive summary

The OutlineComparison image-comparison threshold was tuned on 100 labeled image pairs. The best threshold for balanced F1 score was **0.79**. On the held-out test set, it achieved **70.0% accuracy**, **60.0% recall**, and an **F1 score of 0.667**.

However, the threshold produced **2 false cache hits** and **4 false misses**. Because a false cache hit can reuse stale perception results, `0.79` should be treated as an experimental, F1-optimized threshold rather than a final production safety threshold.

## Objective

Determine a OutlineComparison similarity-score threshold for deciding whether a previous perception result can be reused:

- **Cache hit:** the scene is sufficiently unchanged; reuse the result.
- **Cache miss:** the scene changed; run fresh perception.

The critical error is a false hit because it may reuse stale scene information.

## Test performed

The test was run by [`src/tests/tuneOutlineComparison.py`](../tuneOutlineComparison.py) using the dataset in [`testing/dataset/`](../../../../testing/dataset/).

The dataset contains:

- 100 labeled image pairs
- 50 cache hits
- 50 cache misses
- 10 source scenes
- OutlineComparison similarity scores between 0.0 and 1.0

For every pair, the script:

1. Compared the two images using OutlineComparison.
2. Generated a similarity score.
3. Classified the pair as a cache hit when the score was at least the candidate threshold.
4. Compared that prediction with the labeled cache decision.

The computed scores are stored in [`src/tests/OutlineComparisonTestResult/outlineComparisonScores.csv`](src/tests/OutlineComparisonTestResult/outlineComparisonScores.csv).

## Test case and labels

Each test case consists of two frames from the same source video. The labels define:

- `label_id = 0`: unchanged / **CACHE HIT**
- `label_id = 1`: changed / **CACHE MISS**

The positive class used during tuning is a cache hit.

## Fine-tuning method

The pairs were split by scene:

- **80 tuning pairs:** used to select the threshold.
- **20 held-out test pairs:** from `MOT17-11` and `MOT17-13`, reserved for final evaluation.

The script evaluated every threshold from `0.00` through `1.00` in increments of `0.01`. For each candidate, it calculated the F1 score on the 80 tuning pairs. The threshold with the highest F1 score was selected. In the event of a tie, the higher threshold was preferred.

## Results

Selected threshold:

```text
0.79
```

Held-out test results:

| Metric | Result |
|---|---:|
| Accuracy | 70.0% |
| Precision | 75.0% |
| Recall | 60.0% |
| F1 score | 0.667 |
| False cache hits | 2 |
| False cache misses | 4 |

The always-cache-hit baseline achieved 50.0% accuracy and an F1 score of 0.667, so the tuned threshold improved on that baseline.

## Reproduction

Activate the project virtual environment and run:

```bash
source .venv/bin/activate
python src/tests/tuneOutlineComparison.py --recompute
```

The visualization is available at [`outlineComparisonThreshold.png`](outlineComparisonThreshold.png).
