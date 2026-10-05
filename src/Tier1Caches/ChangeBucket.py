"""
idea: Every frame, pour the small outline diff from the previous frame into a bucket. When it's
full, run the module again and empty it. Measuring many small changes beats one big comparison
across a second, where a head-mounted camera has turned too far for the frames to resemble each other.
One bucket per module.
"""

from OutlineComparison import OutlineComparison


class ChangeBucket:
    def __init__(self, threshold, maxFrames=300):
        # threshold: calibration knob, tune per module on real footage
        # maxFrames: force-run ceiling so an edgeless scene (diff ~0) can't sit silent forever;
        #   300 = 10 s at 30 fps, the longest cache timer
        self.threshold = threshold
        self.maxFrames = maxFrames
        self.level = 0.0
        self.frames = 0

    def add(self, diff):
        # returns True when the module should run now (bucket already emptied)
        self.level += diff
        self.frames += 1
        if self.level >= self.threshold or self.frames >= self.maxFrames:
            self.reset()
            return True
        return False

    def reset(self):
        # also call this whenever the module runs for any other reason
        self.level = 0.0
        self.frames = 0

    @staticmethod
    def similar(frame1, frame2, threshold=0.9):
        # two raw BGR frames in -> (score, similar enough?). Pours the one outline diff into the
        # bucket; score = how much room is left, accept at >= threshold.
        # ponytail: with only 2 frames this equals OutlineComparison.similar; the bucket only
        #   differs when fed every frame in between (see add())
        oc = OutlineComparison()
        oc.update(OutlineComparison.shrinkGrey(frame1))
        score = 1 - oc.update(OutlineComparison.shrinkGrey(frame2))
        return score, score >= threshold


if __name__ == "__main__":
    b = ChangeBucket(threshold=1.0, maxFrames=10)
    assert [b.add(0.3) for _ in range(4)] == [False, False, False, True]
    assert b.level == 0.0
    assert [b.add(0.0) for _ in range(10)][-1] is True  # ceiling fires on a still scene
    b.add(0.9); b.reset()
    assert b.add(0.5) is False
    print("ChangeBucket ok")
