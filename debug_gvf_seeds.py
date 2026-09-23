"""
Quick seed placement diagnostic for gvf_pipeline.py, checking directly
whether the over splitting seen in its floe count, confirmed on frame 0
where SAM 2 and YOLOE both treat a region as one floe while GVF splits
it into two or three, traces back to more than one seed landing inside
what is actually a single real floe.

Reuses generate_seed_points from sam2_pipeline.py rather than writing a
second copy of the same logic, since that function already reproduces
gvf_pipeline.py's own seeding stage exactly, tiled Otsu binarization,
raw local maxima, then per component suppression, the same seeds GVF's
own snake evolution starts from.

MIN_SEED_DISTANCE, ABS_SEP_SCALE_FACTOR, and ABS_SEP_MIN_PX in
gvf_pipeline.py were already retuned once, specifically against measured
ridge spacing and depth on elongated floes, 24 to 29 pixels apart with
depth around 14 to 21. If a cluster is still surviving suppression after
that tuning, worth checking its actual measured numbers before assuming
the same scale_factor formula fits, rather than raising it again on a
visual impression alone, since it may fit a different spacing to depth
ratio than what elongated floes produced.

Usage

    python debug_gvf_seeds.py

Set FRAME_INDEX below to inspect a different frame. Saves
debug_gvf_seeds.png with every seed point marked, then opens a box
drawing window on that same image, box the specific cluster of
interest and press Enter, Esc to skip this part. Prints each seed
inside the box, its coordinates and distance transform depth, the
pairwise distance between every pair, and the minimum separation
gvf_pipeline.py's current suppression logic would require to keep both
as separate seeds, so it is clear by how much this cluster does or
does not clear the existing threshold.
"""

import os

import cv2
import numpy as np

from sea_ice_preprocessing import initial_perspective_transform, perspective_transform, enhance_gray
from sam2_pipeline import generate_seed_points
from gvf_pipeline import ABS_SEP_SCALE_FACTOR, ABS_SEP_MIN_PX

RAW_VIDEO_PATH = "video01.mp4"
FRAME_INDEX = 280
FIRST_FRAME_TEMP_PATH = "first_frame_temp.jpg"


def main():
    cap = cv2.VideoCapture(RAW_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video at " + RAW_VIDEO_PATH)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first frame of " + RAW_VIDEO_PATH)
    cv2.imwrite(FIRST_FRAME_TEMP_PATH, first_frame)

    _, factor, x, y, width, height = initial_perspective_transform(FIRST_FRAME_TEMP_PATH)
    os.remove(FIRST_FRAME_TEMP_PATH)

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame " + str(FRAME_INDEX))

    corrected = perspective_transform(frame, factor, x, y)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_gray(gray)

    seed_points = generate_seed_points(enhanced)
    print("seeds found, " + str(len(seed_points)))

    annotated = corrected.copy()
    for sx, sy, _ in seed_points:
        cv2.circle(annotated, (int(round(sx)), int(round(sy))), 3, (0, 0, 255), -1)
    cv2.imwrite("debug_gvf_seeds.png", annotated)
    print("Saved debug_gvf_seeds.png.")

    print("Box the cluster you want numbers for, Enter to confirm, Esc to skip.")
    window_name = "Box the cluster in question, Enter to confirm"
    box = cv2.selectROI(window_name, annotated, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    bx, by, bw, bh = box
    if bw == 0 or bh == 0:
        print("No box drawn, skipping the cluster breakdown.")
        return

    in_box = [(sx, sy, d) for sx, sy, d in seed_points
              if bx <= sx <= bx + bw and by <= sy <= by + bh]

    print(str(len(in_box)) + " seed(s) inside the box.")
    for i, (sx, sy, d) in enumerate(in_box):
        print("  seed " + str(i) + ", x=" + f"{sx:.1f}" + " y=" + f"{sy:.1f}" + " depth=" + f"{d:.2f}")

    print("Pairwise distances against the current suppression threshold, "
          "scale_factor=" + str(ABS_SEP_SCALE_FACTOR) + " min_px=" + str(ABS_SEP_MIN_PX))
    for i in range(len(in_box)):
        for j in range(i + 1, len(in_box)):
            xi, yi, di = in_box[i]
            xj, yj, dj = in_box[j]
            dist = float(np.hypot(xi - xj, yi - yj))
            min_sep = max(ABS_SEP_SCALE_FACTOR * min(di, dj), ABS_SEP_MIN_PX)
            verdict = "WOULD MERGE" if dist < min_sep else "kept separate"
            print("  seed " + str(i) + " to seed " + str(j) + ", distance=" + f"{dist:.1f}"
                  + " required_min_sep=" + f"{min_sep:.1f}" + "  (" + verdict + ")")


if __name__ == "__main__":
    main()