# ORB Similarity Threshold Report

## Executive summary

The ORB image-comparison threshold was tuned on 100 labeled image pairs. The
best threshold for balanced F1 score was **0.16**. On the held-out test set,
it achieved **70.0% accuracy**, **90.0% recall**, and an **F1 score of 0.750**.

However, the threshold produced **5 false cache hits** and only **1 false
miss**. Because a false cache hit can reuse stale perception results, `0.16`
should be treated as an experimental, F1-optimized threshold rather than a
final production safety threshold.

For a more conservative deployment, **0.25** is a reasonable starting point
for additional validation. It produced fewer false hits on the held-out
examples while retaining the same number of false misses.

## Objective

Determine an ORB similarity-score threshold for deciding whether a previous
perception result can be reused:

- **Cache hit:** the scene is sufficiently unchanged; reuse the result.
- **Cache miss:** the scene changed; run fresh perception.

The critical error is a false hit because it may reuse stale scene
information.

## Test performed

The test was run by
[`src/tests/tuneOrbThreshold.py`](src/tests/tuneOrbThreshold.py) using the
dataset in [`testing/dataset/`](testing/dataset/).

The dataset contains:

- 100 labeled image pairs
- 50 cache hits
- 50 cache misses
- 10 source scenes
- ORB similarity scores between 0.0 and 1.0

For every pair, the script:

1. Compared the two images using ORB feature matching.
2. Generated a similarity score.
3. Classified the pair as a cache hit when the score was at least the
   candidate threshold.
4. Compared that prediction with the labeled cache decision.

The computed scores are stored in
[`artifacts/orbScores.csv`](artifacts/orbScores.csv).

## Test case and labels

Each test case consists of two frames from the same source video. The labels
define:

- `label_id = 0`: unchanged / **CACHE HIT**
- `label_id = 1`: changed / **CACHE MISS**

The positive class used during tuning is a cache hit.

## Fine-tuning method

The pairs were split by scene:

- **80 tuning pairs:** used to select the threshold.
- **20 held-out test pairs:** from `MOT17-11` and `MOT17-13`, reserved for final
  evaluation.

The script evaluated every threshold from `0.00` through `1.00` in increments
of `0.01`. For each candidate, it calculated the F1 score on the 80 tuning
pairs. The threshold with the highest F1 score was selected. In the event of a
tie, the higher threshold was preferred.

## Results

Selected threshold:

```text
0.16
```

Held-out test results:

| Metric | Result |
|---|---:|
| Accuracy | 70.0% |
| Precision | 64.3% |
| Recall | 90.0% |
| F1 score | 0.750 |
| False cache hits | 5 |
| False cache misses | 1 |

The always-cache-hit baseline achieved 50.0% accuracy and an F1 score of
0.667, so the tuned threshold improved on that baseline.

## Threshold risk analysis

The held-out test set showed the following trade-off:

| Threshold | False hits | False misses |
|---:|---:|---:|
| 0.16 | 5 | 1 |
| 0.20 | 3 | 1 |
| 0.25 | 3 | 1 |
| 0.30 | 2 | 3 |
| 0.40 | 2 | 6 |
| 0.50 | 0 | 8 |

The hit and miss score distributions overlap, so no tested threshold perfectly
separates the two classes. Increasing the threshold reduces stale-cache reuse
but causes more fresh perception runs.

## Recommendation

Use **0.16** when reporting the F1-optimized benchmark result. Do not yet
present it as a final production threshold.

For safety-oriented deployment, validate **0.25** or higher using a
cost-weighted objective that penalizes false cache hits more heavily than false
misses. Additional independently reviewed data should be collected before
finalizing the production value.

## Reproduction

Activate the project virtual environment and run:

```bash
source .venv/bin/activate
python src/tests/tuneOrbThreshold.py --recompute
```

The visualization is available at
[`artifacts/orbThreshold.png`](artifacts/orbThreshold.png).
