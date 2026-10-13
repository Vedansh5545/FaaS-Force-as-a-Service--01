"""
idea: Every frame, pour the small outline diff from the previous frame into a bucket. When it's
full, run the module again and empty it. Measuring many small changes beats one big comparison
across a second, where a head-mounted camera has turned too far for the frames to resemble each other.
One bucket per module.
"""

from OutlineComparison import OutlineComparison


def ChangeBucket(frame1, frame2, threshold=0.9, between=(), maxFrames=300):
    """Raw BGR frames in -> (score, similar enough?), pouring every consecutive diff into one bucket."""
    # maxFrames: force-run ceiling so an edgeless scene (diff ~0) can't sit silent forever;
    #   300 = 10 s at 30 fps, the longest cache timer
    sequence = [frame1, *between, frame2]
    level = 0.0
    frames = 1  # frame1 itself, nothing to compare it against yet

    for previous, current in zip(sequence, sequence[1:]):
        level += 1 - OutlineComparison(previous, current)[0]
        frames += 1
        if level >= 1 - threshold or frames >= maxFrames:
            break

    # score = how much room is left in the bucket
    score = max(0.0, 1 - level)
    return score, score >= threshold and frames < maxFrames


if __name__ == "__main__":
    import numpy as np

    frame = np.full((90, 160, 3), 20, np.uint8)
    frame[30:60, 40:80] = 200
    moved = np.full((90, 160, 3), 20, np.uint8)
    moved[30:60, 70:110] = 200

    assert ChangeBucket(frame, frame)[1]
    assert not ChangeBucket(frame, moved)[1]
    # many small steps fill the bucket even though no single pair looks different
    drift = [np.roll(frame, 4 * i, axis=1) for i in range(1, 10)]
    assert not ChangeBucket(frame, drift[-1], between=drift[:-1])[1]
    assert ChangeBucket(frame, frame, maxFrames=1) == (1.0, False)  # force-run ceiling
    print("ChangeBucket ok")
