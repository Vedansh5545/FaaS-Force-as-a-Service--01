"""
Generates labelled synthetic test cases into src/tests/cases/synthetic/.

Each case is a folder of 2 frames: 000.jpg = frame1, 001.jpg = frame2. Folder name starts with
"same_" or "diff_" = the expected answer.

Real footage goes in src/tests/cases/<anything>/ using the same layout; the benchmark picks it up.
"""
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

Frame_Width, Frame_Height = 1280, 720
Output_Folder = Path(__file__).parent / "cases" / "synthetic"
Random_Seeds = range(5)


def scene(colorSeed, layoutSeed, width=Frame_Width * 2):
    # wide canvas of random shapes so we can pan across it; colours and layout seeded separately
    colors = np.random.default_rng(colorSeed)
    layout = np.random.default_rng(layoutSeed)
    img = np.full((Frame_Height, width, 3), colors.integers(60, 200, 3), np.uint8)
    for _ in range(40):
        color = tuple(int(c) for c in colors.integers(0, 256, 3))
        x, y = int(layout.integers(0, width)), int(layout.integers(0, Frame_Height))
        if layout.random() < 0.5:
            size = layout.integers(40, 300, 2)
            cv.rectangle(img, (x, y), (x + int(size[0]), y + int(size[1])), color, -1)
        else:
            cv.circle(img, (x, y), int(layout.integers(20, 120)), color, -1)
    return img


def view(canvas, x=0):
    return canvas[:, x:x + Frame_Width].copy()


def noisy(img, seed):
    # camera sensor noise
    n = np.random.default_rng(seed).normal(0, 4, img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)


def scale(img, k):
    return np.clip(img * k, 0, 255).astype(np.uint8)


def pan(canvas, toPx):
    # head turned toPx pixels to the right
    return [view(canvas), view(canvas, round(toPx))]


def cases(s):
    canvas = scene(s, s)
    v = view(canvas)
    entered = v.copy()
    cv.rectangle(entered, (400, 150), (800, 550), (30, 30, 220), -1)
    return {
        "same_sensor_noise": [v, noisy(v, s)],
        "same_cloud_darker": [v, scale(v, 0.6)],
        "same_lights_brighter": [v, scale(v, 1.3)],
        "same_small_head_turn": pan(canvas, Frame_Width * 0.02),
        "same_slight_blur": [v, cv.GaussianBlur(v, (5, 5), 0)],
        "diff_new_room": [v, view(scene(s + 100, s + 100))],
        "diff_object_enters": [v, entered],
        "diff_big_head_turn": pan(canvas, Frame_Width * 0.30),
        "diff_same_colours_new_layout": [v, view(scene(s, s + 100))],
    }


if __name__ == "__main__":
    shutil.rmtree(Output_Folder, ignore_errors=True)
    for s in Random_Seeds:
        for name, frames in cases(s).items():
            d = Output_Folder / f"{name}_s{s}"
            d.mkdir(parents=True)
            for i, f in enumerate(frames):
                cv.imwrite(str(d / f"{i:03}.jpg"), f)
    print(f"wrote {len(Random_Seeds) * 9} cases to {Output_Folder}")
