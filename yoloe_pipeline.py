"""
YOLOE11 video pipeline

Preprocessing (perspective correction, and optionally gray level
enhancement) is handled by sea_ice_preprocessing.py.

Two prompting strategies are supported, switched with PROMPT_MODE
below.

"text" uses Ultralytics' open vocabulary text prompt embeddings,
set_classes with a list of plain language class names, "pancake ice"
and "black water" by default. 

"visual" instead uses one or more example bounding boxes drawn by hand
around real floes on a chosen reference frame, and asks the model to
find more objects that look like those examples rather than matching a
text description. This mirrors the prompt based mode Passerotti et al.
(2026) found worked far better than an unguided approach for the
Segment Anything Model on this same kind of imagery.

A note on which frame the visual prompt comes from. REFERENCE_FRAME_INDEX
below defaults to 0, drawing the example boxes on the very first frame
of the video.

A note on tiled inference, TILED_INFERENCE below. 
Rather than handing the model the whole frame at once, it
gets split into overlapping tiles, each one run through detectionseparately.
This multiplies runtime by the number of tiles per frame, expect a full video run to take considerably longer
than the untiled version.

APPLY_GRAY_ENHANCEMENT below defaults to False for this
reason. 

Usage

Reads RAW_VIDEO_PATH below, runs interactive perspective calibration
once against its first frame. If PROMPT_MODE is "visual", a second
window then opens on that same first frame, drag a box tightly around
one clear, well separated floe, and press Enter or Space to confirm.
"""

import csv
import os
import time

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from sea_ice_preprocessing import initial_perspective_transform, perspective_transform, enhance_gray



# Configuration

RAW_VIDEO_PATH = "video01.mp4"                    
OUTPUT_VIDEO_PATH = "yoloe_output.mp4"
OUTPUT_CSV_PATH = "yoloe_sic_log.csv"
FIRST_FRAME_TEMP_PATH = "first_frame_temp.jpg"    

FRAME_STRIDE = 1   

REFERENCE_FRAME_INDEX = 0   # which raw frame the visual prompt boxes get drawn on

MODEL_WEIGHTS = "yoloe-11l-seg.pt"

PROMPT_MODE = "visual"   # "text" for the open vocabulary class list below, "visual" for a hand drawn example box

CLASSES = ["pancake ice", "black water"]   # only used when PROMPT_MODE is "text"
ICE_CLASS_NAME = "pancake ice"             # only used when PROMPT_MODE is "text"

CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.3

APPLY_GRAY_ENHANCEMENT = False   
TILED_INFERENCE = False   
TILE_SIZE_PX = 200          # pixel size of each square tile. 
TILE_OVERLAP_PX = 100       # overlap between neighbouring tiles, needs to be at least as large as the biggest floe
MERGE_IOU_THRESHOLD = 0.5   
VALID_PIXEL_MIN_BRIGHTNESS = 3   # near black pixels are the warp border

DEBUG_VERBOSE = True   



# Preprocessing wrapper, shared between the one time prompt setup and the main loop
def preprocess_frame(frame, factor, x, y):
    """Runs perspective correction and, if enabled, gray level
    enhancement, on one raw frame."""
    corrected = perspective_transform(frame, factor, x, y)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    if APPLY_GRAY_ENHANCEMENT:
        enhanced = enhance_gray(gray)
        model_input = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        model_input = corrected

    return corrected, gray, model_input



# Visual prompt setup

def select_visual_prompt_boxes(image):
    """Opens an interactive window on the given frame so several example
    floes can be boxed by hand.
    """
    window_name = "Box a few different floes, Enter after each, Esc when done"
    boxes = cv2.selectROIs(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    boxes = np.array(boxes).reshape(-1, 4)
    boxes = boxes[(boxes[:, 2] > 0) & (boxes[:, 3] > 0)]   

    if len(boxes) == 0:
        raise RuntimeError("No visual prompt boxes were drawn. Rerun the script and box "
                            "at least one floe, pressing Enter after each, before pressing Esc.")

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


# Tiled inference

def generate_tiles(width, height, tile_size=TILE_SIZE_PX, overlap=TILE_OVERLAP_PX):
    """Returns a list of (x0, y0, x1, y1) tile boundaries covering the
    whole frame, overlapping by `overlap` pixels between neighbours so a
    floe sitting across a seam still appears whole in at least one tile.
    """
    stride = max(tile_size - overlap, 1)
    tiles = []
    y0 = 0
    while True:
        y1 = min(y0 + tile_size, height)
        x0 = 0
        while True:
            x1 = min(x0 + tile_size, width)
            tiles.append((x0, y0, x1, y1))
            if x1 >= width:
                break
            x0 += stride
        if y1 >= height:
            break
        y0 += stride
    return tiles


def detections_from_result(result, tile_shape):
    if result.masks is None:
        return []

    h, w = tile_shape[:2]
    mask_data = result.masks.data.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    detections = []
    for mask, cls_id, conf in zip(mask_data, class_ids, confidences):
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_uint8 = (mask > 0.5).astype(np.uint8)
        detections.append((mask_uint8, int(cls_id), float(conf)))
    return detections


def merge_tile_detections(detections, iou_threshold=MERGE_IOU_THRESHOLD):
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    for mask, cls_id, conf in detections:
        is_duplicate = False
        for kept_mask, kept_cls_id, _ in kept:
            if cls_id != kept_cls_id:
                continue
            intersection = int(np.logical_and(mask, kept_mask).sum())
            if intersection == 0:
                continue
            union = int(np.logical_or(mask, kept_mask).sum())
            iou = intersection / union if union > 0 else 0.0
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((mask, cls_id, conf))
    return [(mask, cls_id) for mask, cls_id, _ in kept]


def run_tiled_detection(model, model_input, reference_image, visual_prompts, frame_shape):
    """Runs YOLOE separately on each overlapping tile of model_input,
    translates every detection's mask back into full frame coordinates,
    then merges detections that appear in more than one tile via
    merge_tile_detections.
    """
    h, w = frame_shape[:2]
    tiles = generate_tiles(w, h)

    raw_detections = []
    for (x0, y0, x1, y1) in tiles:
        tile_img = model_input[y0:y1, x0:x1]

        if PROMPT_MODE == "visual":
            results = model.predict(tile_img, refer_image=reference_image, visual_prompts=visual_prompts,
                                     predictor=YOLOEVPSegPredictor, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                     verbose=False)
        else:
            results = model.predict(tile_img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        tile_detections = detections_from_result(results[0], (y1 - y0, x1 - x0))
        for mask, cls_id, conf in tile_detections:
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y0:y1, x0:x1] = mask
            raw_detections.append((full_mask, cls_id, conf))

    return merge_tile_detections(raw_detections)



# Mask handling
def build_ice_mask(detections, frame_shape, ice_class_index):
    """Builds one binary mask covering every pixel touched by at least
    one ice instance detection, plus a list of the individual instance
    masks for drawing. 
    The union mask is what the SIC calculation uses, so overlapping or
    duplicate detections do not get double counted in the area. """
    h, w = frame_shape[:2]
    union_mask = np.zeros((h, w), dtype=np.uint8)
    instance_masks = []

    for mask, cls_id in detections:
        if cls_id != ice_class_index:
            continue
        instance_masks.append(mask)
        union_mask = np.maximum(union_mask, mask)

    return union_mask, instance_masks


def valid_pixel_mask(gray, min_brightness=VALID_PIXEL_MIN_BRIGHTNESS):
    
    return (gray >= min_brightness).astype(np.uint8)



# Main video loop, calibration once, then preprocessing and YOLOE together
def main():
    if PROMPT_MODE not in ("text", "visual"):
        raise ValueError("PROMPT_MODE must be 'text' or 'visual', got " + repr(PROMPT_MODE))

    cap = cv2.VideoCapture(RAW_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video at " + RAW_VIDEO_PATH)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first frame of " + RAW_VIDEO_PATH)
    cv2.imwrite(FIRST_FRAME_TEMP_PATH, first_frame)

    # Interactive perspective calibration, adjust the trackbars, press Esc to lock in
    _, factor, x, y, width, height = initial_perspective_transform(FIRST_FRAME_TEMP_PATH)
    os.remove(FIRST_FRAME_TEMP_PATH)

    print("Loading YOLOE, this can take a moment on first run.")
    model = YOLOE(MODEL_WEIGHTS)

    visual_prompts = None
    reference_image = None

    if PROMPT_MODE == "text":
        model.set_classes(CLASSES, model.get_text_pe(CLASSES))
        ice_class_index = CLASSES.index(ICE_CLASS_NAME)
        class_names_for_debug = CLASSES
    else:
        
        if REFERENCE_FRAME_INDEX == 0:
            reference_source_frame = first_frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, REFERENCE_FRAME_INDEX)
            ok, reference_source_frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read frame " + str(REFERENCE_FRAME_INDEX)
                                    + " for the visual prompt reference.")

        # Build the same corrected, enhanced frame the loop itself will
        # use, so the box drawn here lines up with what the model
        # actually receives every frame
        _, _, reference_image = preprocess_frame(reference_source_frame, factor, x, y)
        print("A window will open on frame " + str(REFERENCE_FRAME_INDEX) + ". Box a handful "
              "of floes that between them cover the range of sizes and shapes in the scene, "
              "small, large, round, angular, tightly packed with faint boundaries too if this "
              "frame shows any, pressing Enter after each one, then Esc when done.")
        bboxes = select_visual_prompt_boxes(reference_image)
        visual_prompts = {"bboxes": bboxes, "cls": np.zeros(len(bboxes), dtype=int)}   # same class for every example
        ice_class_index = 0
        class_names_for_debug = ["floe examples, visual prompt"]
        print(str(len(bboxes)) + " example box(es) captured for the visual prompt.")

    # Rewind to the start, the frame used for calibration still needs to
    # go through the full pipeline like every other frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_fps = max(fps / FRAME_STRIDE, 1.0)
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
                              out_fps, (width, height))

    csv_file = open(OUTPUT_CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["source_frame_index", "num_floes", "sic_percent", "exec_time_s"])

    if TILED_INFERENCE:
        tile_count = len(generate_tiles(width, height))
        print("TILED_INFERENCE is on, frame size " + str(width) + " x " + str(height)
              + ", " + str(tile_count) + " tiles per frame, expect roughly that "
              "many times the per frame runtime of a single untiled pass.")
        if tile_count <= 1:
            print("  WARNING, only 1 tile, TILE_SIZE_PX is larger than the frame itself, "
                  "tiling is having no effect at all, lower TILE_SIZE_PX before trusting this run.")

    frame_index = 0
    processed_count = 0
    start_time = time.time()

    while True:
        if frame_index % FRAME_STRIDE != 0:
            ok = cap.grab()
            if not ok:
                break
            frame_index += 1
            continue

        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()

        corrected, gray, model_input = preprocess_frame(frame, factor, x, y)

        if TILED_INFERENCE:
            detections = run_tiled_detection(model, model_input, reference_image, visual_prompts, gray.shape)
        else:
            if PROMPT_MODE == "visual":
                results = model.predict(model_input, refer_image=reference_image, visual_prompts=visual_prompts,
                                         predictor=YOLOEVPSegPredictor, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                         verbose=False)
            else:
                results = model.predict(model_input, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            detections = [(mask, cls_id) for mask, cls_id, _ in detections_from_result(results[0], gray.shape)]

        if DEBUG_VERBOSE:
            total_raw = len(detections)
            counts_by_class = {}
            for _, cls_id in detections:
                name = class_names_for_debug[cls_id] if cls_id < len(class_names_for_debug) else "class_" + str(cls_id)
                counts_by_class[name] = counts_by_class.get(name, 0) + 1
            print("  raw detections total=" + str(total_raw) + ", by class=" + str(counts_by_class))

        ice_mask, instance_masks = build_ice_mask(detections, gray.shape, ice_class_index)
        valid_mask = valid_pixel_mask(gray)
        valid_pixels = max(int(valid_mask.sum()), 1)
        sic = 100.0 * int((ice_mask & valid_mask).sum()) / valid_pixels
        floe_count = len(instance_masks)

        exec_time = time.time() - t0
        csv_writer.writerow([frame_index, floe_count, round(sic, 3), round(exec_time, 4)])

        annotated = corrected.copy()
        for mask in instance_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(annotated, contours, -1, (0, 140, 255), 1)
        cv2.putText(annotated, "SIC: " + f"{sic:.1f}" + "%  Floes: " + str(floe_count),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(annotated)

        processed_count += 1
        elapsed = time.time() - start_time
        print("Frame " + str(frame_index) + " of " + str(total_frames)
              + ", SIC=" + f"{sic:.2f}" + "%  floes=" + str(floe_count)
              + "  frame_time=" + f"{exec_time:.2f}" + "s  elapsed=" + f"{elapsed:.1f}" + "s")

        frame_index += 1

    cap.release()
    writer.release()
    csv_file.close()
    print("Done. " + str(processed_count) + " of " + str(total_frames) + " frames processed, stride "
          + str(FRAME_STRIDE) + ".")
    print("Output video, " + OUTPUT_VIDEO_PATH)
    print("Per frame log, " + OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()