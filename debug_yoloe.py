"""
Quick single frame diagnostic for yoloe_pipeline.py, built the same way
as debug_sam2_mask_overlay.py was for the SAM 2 pipeline, to check raw
detection counts and mask coverage on one specific frame without
waiting through the full video, useful for the handful of frames where
yoloe_sic_log.csv came out far lower than GVF and SAM 2 agreed on,
frame 280 being the clearest example so far, where raw detections came
out at 7 against 69 floes GVF resolved on the identical scene.

Runs interactive calibration once, then the same one time visual
prompt box drawing step the full pipeline uses, since PROMPT_MODE is
assumed to be "visual" here, matching the current default in
yoloe_pipeline.py. Respects TILED_INFERENCE from yoloe_pipeline.py, so
TILE_SIZE_PX, TILE_OVERLAP_PX, and MERGE_IOU_THRESHOLD can be tuned
directly there and tested here on a single frame before committing to
a full run.

Usage

    python debug_yoloe_frame.py

Set FRAME_INDEX below to inspect a different frame, and
REFERENCE_FRAME_INDEX to test drawing the visual prompt on a different
frame than the one being evaluated. Prints the raw detection count
after tiling and merging, then the kept floe count and SIC, and saves
a contour outline image and a filled overlay image, the same pair
debug_sam2_mask_overlay.py produces, so the two can be compared side
by side on the same frame.
"""

import os

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from sea_ice_preprocessing import initial_perspective_transform
from yoloe_pipeline import (
    preprocess_frame,
    select_visual_prompt_boxes,
    build_ice_mask,
    valid_pixel_mask,
    run_tiled_detection,
    detections_from_result,
    generate_tiles,
    MODEL_WEIGHTS,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    TILED_INFERENCE,
)

RAW_VIDEO_PATH = "video01.mp4"
FRAME_INDEX = 0
REFERENCE_FRAME_INDEX = 0   # which frame the visual prompt boxes get drawn on, independent of FRAME_INDEX
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

    if REFERENCE_FRAME_INDEX == 0:
        reference_source_frame = first_frame
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, REFERENCE_FRAME_INDEX)
        ok, reference_source_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read frame " + str(REFERENCE_FRAME_INDEX)
                                + " for the visual prompt reference.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame " + str(FRAME_INDEX))

    print("Loading YOLOE.")
    model = YOLOE(MODEL_WEIGHTS)

    _, _, reference_image = preprocess_frame(reference_source_frame, factor, x, y)
    print("Box a handful of floes on frame " + str(REFERENCE_FRAME_INDEX) + ", covering a "
          "range of sizes and shapes, Enter after each one, Esc when done.")
    bboxes = select_visual_prompt_boxes(reference_image)
    visual_prompts = {"bboxes": bboxes, "cls": np.zeros(len(bboxes), dtype=int)}
    print(str(len(bboxes)) + " example box(es) captured.")

    corrected, gray, model_input = preprocess_frame(frame, factor, x, y)
    print("frame size, width=" + str(gray.shape[1]) + " height=" + str(gray.shape[0]))

    if TILED_INFERENCE:
        tile_count = len(generate_tiles(gray.shape[1], gray.shape[0]))
        print("TILED_INFERENCE is on, " + str(tile_count) + " tiles for this frame.")
        detections = run_tiled_detection(model, model_input, reference_image, visual_prompts, gray.shape)
    else:
        results = model.predict(model_input, refer_image=reference_image, visual_prompts=visual_prompts,
                                 predictor=YOLOEVPSegPredictor, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                 verbose=False)
        detections = [(mask, cls_id) for mask, cls_id, _ in detections_from_result(results[0], gray.shape)]

    print("detections after tiling and merging, " + str(len(detections)))

    ice_mask, instance_masks = build_ice_mask(detections, gray.shape, ice_class_index=0)
    valid_mask = valid_pixel_mask(gray)
    sic = 100.0 * int((ice_mask & valid_mask).sum()) / max(int(valid_mask.sum()), 1)
    print("floes kept, " + str(len(instance_masks)))
    print("SIC, " + f"{sic:.2f}" + " percent")

    contours_only = corrected.copy()
    for mask in instance_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(contours_only, contours, -1, (0, 140, 255), 1)
    cv2.imwrite("debug_yoloe_contours.png", contours_only)

    overlay = corrected.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 2] = 255
    mask_3ch = np.stack([ice_mask] * 3, axis=-1).astype(bool)
    blended = cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0)
    overlay[mask_3ch] = blended[mask_3ch]
    cv2.imwrite("debug_yoloe_filled_overlay.png", overlay)

    print("Saved debug_yoloe_contours.png and debug_yoloe_filled_overlay.png.")


if __name__ == "__main__":
    main()