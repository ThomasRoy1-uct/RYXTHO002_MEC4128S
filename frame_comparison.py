"""
frame_comparison.py

Pulls one chosen frame from each method's already annotated output video,
gvf_snake_output.mp4, yoloe_output.mp4, sam2_output.mp4,
out_liss_benchmark.mp4, and tiles the four into one side by side image, for
a qualitative comparison.

Set FRAME_INDICES below to whichever source frame indices you want a
comparison image for, then just run:  python frame_comparison.py
"""

import os
import cv2
import numpy as np

VIDEOS = {
    "GVF": "gvf_snake_output.mp4",
    "YOLOE": "yoloe_output.mp4",
    "SAM2": "sam2_output.mp4",
    "Liss": "out_liss_benchmark.mp4",
}

# the frame(s) to compare
# [280, 265] for several at once
FRAME_INDICES = [1,280]

FRAME_STRIDE = 1   # set to whatever stride the compared runs used
OUT_DIR = "figures"
HEADING_HEIGHT = 44
GAP = 18
BG_COLOR = (255, 255, 255)   
TEXT_COLOR = (0, 0, 0)


def grab_frame(path, position_in_video):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'")
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, position_in_video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(
            f"Could not read frame at position {position_in_video} from {path}, "
            f"it may be past the end of that video."
        )
    return frame


def with_heading(frame, text):
    h, w = frame.shape[:2]
    heading = np.full((HEADING_HEIGHT, w, 3), BG_COLOR, dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = max((w - text_w) // 2, 0)
    ty = (HEADING_HEIGHT + text_h) // 2
    cv2.putText(heading, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)
    return np.vstack([heading, frame])


def hstack_with_gap(images, gap):
    h = images[0].shape[0]
    strip = np.full((h, gap, 3), BG_COLOR, dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.hstack([out, strip, im])
    return out


def vstack_with_gap(images, gap):
    w = images[0].shape[1]
    strip = np.full((gap, w, 3), BG_COLOR, dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.vstack([out, strip, im])
    return out


def build_comparison(source_frame_index, out_path):
    position_in_video = source_frame_index // FRAME_STRIDE
    raw_frames = [(m, grab_frame(path, position_in_video)) for m, path in VIDEOS.items()]

    h, w = raw_frames[0][1].shape[:2]
    cells = [with_heading(cv2.resize(f, (w, h)), m) for m, f in raw_frames]

    top = hstack_with_gap(cells[0:2], GAP)
    bottom = hstack_with_gap(cells[2:4], GAP)
    grid = vstack_with_gap([top, bottom], GAP)

    cv2.imwrite(out_path, grid)
    print("Saved", out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for frame_index in FRAME_INDICES:
        out_path = os.path.join(OUT_DIR, f"fig_frame_comparison_{frame_index}.png")
        build_comparison(frame_index, out_path)


if __name__ == "__main__":
    main()