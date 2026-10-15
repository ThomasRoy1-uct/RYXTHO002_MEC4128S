"""
SAM 2 video pipeline.


CIRCULARITY_MIN and ECCENTRICITY_MAX below started as placeholder
values, not sourced from the paper, with a note to check them against
whatever thresholds gvf_pipeline.py itself uses for its own
hallucination filtering, so all three pipelines are held to the same
standard of what counts as a valid floe shape. 

Preprocessing, perspective correction and gray level enhancement, is
handled by sea_ice_preprocessing.py.Gray level enhancement is always applied before
seed generation here. SAM 2 is a pretrained model and collapsing its input
to grayscale throws away information its pretrained weights likely
rely on, so this defaults to False.


Usage


Reads RAW_VIDEO_PATH below, runs interactive perspective calibration
once against its first frame, then processes the rest of the video.
Writes an annotated video and a per frame CSV log of floe count and
sea ice concentration.

Checkpoint and config need to be downloaded separately per Meta's SAM 2
setup instructions, and both need to point at files that actually exist
on disk, these names match the SAM 2.1 large release, swap for whichever
checkpoint size is actually installed
"""

import csv
import os
import time

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sea_ice_preprocessing import initial_perspective_transform, perspective_transform, enhance_gray
from gvf_pipeline import tiled_otsu_binarize, raw_local_maxima, suppress_seeds_in_component



# Configuration

RAW_VIDEO_PATH = "video01.mp4"                    
OUTPUT_VIDEO_PATH = "sam2_output.mp4"
OUTPUT_CSV_PATH = "sam2_sic_log.csv"
FIRST_FRAME_TEMP_PATH = "first_frame_temp.jpg"    

FRAME_STRIDE = 1  


SAM2_CHECKPOINT = "/Users/thomasroy/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

APPLY_GRAY_ENHANCEMENT = False   

SEED_MIN_COMPONENT_AREA = 15   # mirrors gvf_pipeline.py's own component area floor

# Shape filters applied to each mask after SAM 2 returns it, matching
# Passerotti et al. 
CONVEXITY_MIN = 0.8       # Passerotti et al.  masks at or below this are excluded
CIRCULARITY_MIN = 0.5     
ECCENTRICITY_MAX = 0.97    
MIN_MASK_AREA_PX = 20     # drops near empty masks that indicate a failed prompt rather than a real floe
MAX_MASK_AREA_FRACTION = 0.05   # drops masks covering more than 5 percent of the frame

VALID_PIXEL_MIN_BRIGHTNESS = 3   # near black pixels are the warp border

DEBUG_VERBOSE = True   



# Seed generation, reusing gvf_pipeline.py's own seeding stage directly
def generate_seed_points(enhanced_gray):
    """Reproduces gvf_pipeline.py's own seeding stage, tiled Otsu
    binarization, raw local maxima on the distance transform, then per
    component seed suppression."""
    binary = tiled_otsu_binarize(enhanced_gray)
    raw_seeds = raw_local_maxima(binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    seed_points = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < SEED_MIN_COMPONENT_AREA:
            continue
        comp_mask = (labels == i).astype(np.uint8)
        seeds_in_comp_raw = [
            s for s in raw_seeds
            if comp_mask[int(round(s[1])), int(round(s[0]))] == 1
        ]
        seed_points.extend(suppress_seeds_in_component(seeds_in_comp_raw))

    return seed_points



# Shape filters, matching Passerotti et al. (2026) approach 

def mask_convexity(contour):
    """Ratio of contour area to its own convex hull area, close to 1 for
    the rounded, mostly convex shape of a real floe, lower for irregular
    open water regions"""
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0:
        return 0.0
    return area / hull_area


def mask_circularity(contour):
    """Standard isoperimetric circularity, 1.0 for a perfect circle,
    lower for elongated or irregular shapes."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return 4.0 * np.pi * area / (perimeter ** 2)


def mask_eccentricity(contour):
    """Eccentricity of the contour's best fit ellipse, 0 for a circle,
    approaching 1 for an increasingly elongated shape.
    """
    if len(contour) < 5:
        return 1.0
    (_, _), (minor_axis, major_axis), _ = cv2.fitEllipse(contour)
    major_axis, minor_axis = max(major_axis, minor_axis), min(major_axis, minor_axis)
    if major_axis <= 0:
        return 1.0
    return float(np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2)))


def is_valid_floe_shape(contour):
    return (
        mask_convexity(contour) > CONVEXITY_MIN
        and mask_circularity(contour) >= CIRCULARITY_MIN
        and mask_eccentricity(contour) <= ECCENTRICITY_MAX
    )



# SAM 2 mask generation, one point prompt per seed
def masks_from_seeds(predictor, seed_points, frame_shape):
    """Runs SAM 2 once per seed with a single positive point prompt"""
    h, w = frame_shape[:2]
    instance_masks = []
    discard_reasons = {
        "empty_mask": 0,
        "too_small": 0,
        "too_large": 0,
        "failed_shape": 0,
        "failed_convexity": 0,
        "failed_circularity": 0,
        "failed_eccentricity": 0,
    }

    for x, y, _ in seed_points:
        point_coords = np.array([[float(x), float(y)]])
        point_labels = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best_mask = (masks[int(np.argmax(scores))] > 0).astype(np.uint8)

        if best_mask.shape != (h, w):
            best_mask = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            discard_reasons["empty_mask"] += 1
            continue
        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < MIN_MASK_AREA_PX:
            discard_reasons["too_small"] += 1
            continue
        if cv2.contourArea(contour) > MAX_MASK_AREA_FRACTION * h * w:
            discard_reasons["too_large"] += 1
            continue

        # Checked individually rather than through is_valid_floe_shape
        
        convexity = mask_convexity(contour)
        circularity = mask_circularity(contour)
        eccentricity = mask_eccentricity(contour)
        shape_ok = True
        if convexity <= CONVEXITY_MIN:
            discard_reasons["failed_convexity"] += 1
            shape_ok = False
        if circularity < CIRCULARITY_MIN:
            discard_reasons["failed_circularity"] += 1
            shape_ok = False
        if eccentricity > ECCENTRICITY_MAX:
            discard_reasons["failed_eccentricity"] += 1
            shape_ok = False
        if not shape_ok:
            discard_reasons["failed_shape"] += 1
            continue

        clean_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean_mask, [contour], -1, 1, thickness=cv2.FILLED)
        instance_masks.append(clean_mask)

    if DEBUG_VERBOSE:
        total_discarded = (
            discard_reasons["empty_mask"] + discard_reasons["too_small"]
            + discard_reasons["too_large"] + discard_reasons["failed_shape"]
        )
        print("  seeds=" + str(len(seed_points)) + " kept=" + str(len(instance_masks))
              + " discarded=" + str(total_discarded)
              + " (empty_mask=" + str(discard_reasons["empty_mask"])
              + " too_small=" + str(discard_reasons["too_small"])
              + " too_large=" + str(discard_reasons["too_large"])
              + " failed_shape=" + str(discard_reasons["failed_shape"]) + ")")
        print("  of the failed_shape cases, which check each one tripped, convexity="
              + str(discard_reasons["failed_convexity"])
              + " circularity=" + str(discard_reasons["failed_circularity"])
              + " eccentricity=" + str(discard_reasons["failed_eccentricity"])
              + " (a seed can fail more than one check at once, so these can add up to more than failed_shape)")

    return instance_masks


def build_ice_mask(instance_masks, frame_shape):
    """Unions every surviving instance mask into one binary mask for the
    SIC calculation"""
    h, w = frame_shape[:2]
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in instance_masks:
        union_mask = np.maximum(union_mask, mask)
    return union_mask


def valid_pixel_mask(gray, min_brightness=VALID_PIXEL_MIN_BRIGHTNESS):
    return (gray >= min_brightness).astype(np.uint8)


# Device selection
# Selects CPU or local GPU

def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on Apple Silicon MPS. Meta's own SAM 2 notes this backend can give slightly different numerical results than CUDA, worth keeping in mind if a comparison run ever happens on different hardware.")
    else:
        device = torch.device("cpu")
        print("Running on CPU. SAM 2 is considerably slower here than the other two pipelines, expect a much higher exec_time_s per frame in the log, Passerotti et al. report roughly 0.3 seconds per floe for SAM-P on their own hardware, a useful reference point once the log has real numbers to compare against.")
    return device



# Main video loop, calibration once, then preprocessing and SAM 2 together
def main():
    cap = cv2.VideoCapture(RAW_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video at " + RAW_VIDEO_PATH)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Grab the first frame once, save it to disk, since calibration reads
    # from a file path rather than a frame already in memory
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first frame of " + RAW_VIDEO_PATH)
    cv2.imwrite(FIRST_FRAME_TEMP_PATH, first_frame)

    # Interactive calibration, adjust the trackbars, press Esc to lock in
    _, factor, x, y, width, height = initial_perspective_transform(FIRST_FRAME_TEMP_PATH)
    os.remove(FIRST_FRAME_TEMP_PATH)

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_fps = max(fps / FRAME_STRIDE, 1.0)
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
                              out_fps, (width, height))

    csv_file = open(OUTPUT_CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["source_frame_index", "num_floes", "sic_percent", "exec_time_s"])

    print("Loading SAM 2, this can take a moment on first run.")
    device = select_device()
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

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

        # Preprocessing, using the calibration 
        corrected = perspective_transform(frame, factor, x, y)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        enhanced = enhance_gray(gray)   # always used for seeding

        seed_points = generate_seed_points(enhanced)

        if APPLY_GRAY_ENHANCEMENT:
            model_input = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            model_input = corrected

        # SAM 2 expects RGB channel order, OpenCV loads and holds frames in BGR
        model_input_rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            predictor.set_image(model_input_rgb)
            instance_masks = masks_from_seeds(predictor, seed_points, gray.shape)

        ice_mask = build_ice_mask(instance_masks, gray.shape)
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
        for sx, sy, _ in seed_points:
            cv2.circle(annotated, (int(round(sx)), int(round(sy))), 2, (0, 0, 255), -1)
        cv2.putText(annotated, "SIC: " + f"{sic:.1f}" + "%  Floes: " + str(floe_count),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(annotated)

        processed_count += 1
        elapsed = time.time() - start_time
        print("Frame " + str(frame_index) + " of " + str(total_frames)
              + ", SIC=" + f"{sic:.2f}" + "%  floes=" + str(floe_count)
              + "  seeds=" + str(len(seed_points))
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
