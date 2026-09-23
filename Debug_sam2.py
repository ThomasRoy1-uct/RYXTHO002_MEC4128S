"""
Quick single frame diagnostic for sam2_pipeline.py, built the same way
as compare_binary_frame.py was for the YOLOE pipeline, to check
directly whether SAM 2's masks on a given frame are including pixels
that look like water to a human eye, rather than inferring this from
contour outlines alone, which only show a mask's boundary, not how far
a neighbouring mask might extend underneath it.

Runs interactive calibration once, same as the full pipeline, then
processes a single frame, so this is fast to rerun on different frame
indices without waiting through the full video.

Usage

    python debug_sam2_mask_overlay.py

Set FRAME_INDEX below to inspect a different frame. Saves two images,
one with contour outlines only, matching what sam2_output.mp4 shows,
and one with the full union ice mask filled in as a translucent red
overlay, so any real water pixels wrongly included become directly
visible rather than hidden inside a thin contour line.
"""

import os

import cv2
import numpy as np
import torch

from sea_ice_preprocessing import initial_perspective_transform, perspective_transform, enhance_gray
from sam2_pipeline import (
    generate_seed_points,
    masks_from_seeds,
    build_ice_mask,
    valid_pixel_mask,
    select_device,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CONFIG,
    APPLY_GRAY_ENHANCEMENT,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

RAW_VIDEO_PATH = "video01.mp4"
FRAME_INDEX = 305
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

    model_input = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if APPLY_GRAY_ENHANCEMENT else corrected
    model_input_rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)

    print("Loading SAM 2.")
    device = select_device()
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    with torch.inference_mode():
        predictor.set_image(model_input_rgb)
        instance_masks = masks_from_seeds(predictor, seed_points, gray.shape)

    print("floes kept after filtering, " + str(len(instance_masks)))

    frame_area = gray.shape[0] * gray.shape[1]
    areas_percent = sorted(
        (100.0 * int(mask.sum()) / frame_area for mask in instance_masks),
        reverse=True,
    )
    print("largest five mask areas, as percent of frame, "
          + ", ".join(f"{a:.1f}" for a in areas_percent[:5]))

    if instance_masks:
        largest_mask = max(instance_masks, key=lambda m: int(m.sum()))
        largest_only = corrected.copy()
        blue_layer = np.zeros_like(largest_only)
        blue_layer[:, :, 0] = 255
        mask_3ch_largest = np.stack([largest_mask] * 3, axis=-1).astype(bool)
        blended_largest = cv2.addWeighted(largest_only, 0.5, blue_layer, 0.5, 0)
        largest_only[mask_3ch_largest] = blended_largest[mask_3ch_largest]
        cv2.imwrite("debug_sam2_largest_mask.png", largest_only)
        print("Saved debug_sam2_largest_mask.png, the single largest kept mask, filled in blue.")

    ice_mask = build_ice_mask(instance_masks, gray.shape)
    valid_mask = valid_pixel_mask(gray)
    sic = 100.0 * int((ice_mask & valid_mask).sum()) / max(int(valid_mask.sum()), 1)
    print("SIC, " + f"{sic:.2f}" + " percent")

    contours_only = corrected.copy()
    for mask in instance_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(contours_only, contours, -1, (0, 140, 255), 1)
    cv2.imwrite("debug_sam2_contours.png", contours_only)

    overlay = corrected.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 2] = 255
    mask_3ch = np.stack([ice_mask] * 3, axis=-1).astype(bool)
    blended = cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0)
    overlay[mask_3ch] = blended[mask_3ch]
    for sx, sy, _ in seed_points:
        cv2.circle(overlay, (int(round(sx)), int(round(sy))), 2, (0, 255, 255), -1)
    cv2.imwrite("debug_sam2_filled_overlay.png", overlay)

    print("Saved debug_sam2_contours.png and debug_sam2_filled_overlay.png.")


if __name__ == "__main__":
    main()