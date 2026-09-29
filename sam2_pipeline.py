"""
SAM 2 video pipeline, benchmarked against the GVF snake and YOLOE11
pipelines on the same source footage and the same two comparison
metrics, sea ice concentration (SIC) percentage and floe count.

Passerotti et al. (2026) tested SAM in two modes, a fully automatic
grid prompted mode (SAM-A) and a seed point prompted mode (SAM-P), and
found SAM-P performed clearly better overall, missing far fewer floes
than SAM-A across their benchmark set. The seeds used for SAM-P in the
paper come directly from the same distance transform stage already
used to initialize their GVF snake, one seed per distinct floe or
merged cluster, rather than from a fixed grid across the whole frame.
This script follows that same design. It does not run SAM 2 in fully
automatic mode at all, instead it imports the seed generation functions
straight from gvf_pipeline.py (tiled_otsu_binarize, raw_local_maxima,
suppress_seeds_in_component) and reruns that exact seeding stage on
each frame, then hands every resulting seed to SAM 2 as a single point
prompt, matching the paper's description of SAM-P reusing the GVF
seeding stage's own output rather than generating its own.

Passerotti et al. also tested SAM 2 directly against the original SAM
and found it detected on average 40 percent fewer floes in automatic
mode and 13 percent fewer in prompt based mode on this kind of imagery,
attributing the gap to SAM 2 handling low contrast scenes worse, and
chose to continue with the original SAM for the rest of their paper for
that reason. This project benchmarks SAM 2 specifically rather than the
original SAM, so going in expecting SAM 2 to underperform its own
published reputation here is a documented finding for this exact
problem domain, not a setup mistake.

After each seed produces a mask, the paper applies shape based filters,
convexity, circularity, and eccentricity, to remove masks that swallow
open water rather than isolating a single floe, since SAM cannot tell
ice and water apart on its own, only that something in the frame forms
a distinct region around the prompted point. This script applies the
same three filters. The convexity cutoff of 0.8 is taken directly from
Section 4.2.1 of the paper, masks at or below that value are excluded,
and is left alone for that reason, changing a paper sourced constant to
move this benchmark's numbers closer to another method is the same
tuning for results the project already ruled out elsewhere (GVF's
watershed option, the Liss benchmark's own parameters).

CIRCULARITY_MIN and ECCENTRICITY_MAX below started as placeholder
values, not sourced from the paper, with a note to check them against
whatever thresholds gvf_pipeline.py itself uses for its own
hallucination filtering, so all three pipelines are held to the same
standard of what counts as a valid floe shape. That check has now been
done for one of the two. ECCENTRICITY_MAX is set to 0.9, matching
gvf_pipeline.py's own ECCENTRICITY_HALLUCINATION_MIN exactly, since
both are screening for the same thing, an overly elongated, line like
shape, so borrowing GVF's already measured cutoff is a direct fix
rather than a guess. CIRCULARITY_MIN has no equivalent to borrow, GVF's
own circularity check flags shapes above 1.0 as hallucinated near
perfect circles, a discretisation artefact check on the upper end,
while this filter is screening the lower end, rejecting irregular open
water blobs that never resembled a floe to begin with. The two checks
are not measuring the same failure mode, so there is nothing to align
CIRCULARITY_MIN against yet, it remains a placeholder and should be
validated the way ABS_SEP_SCALE_FACTOR was in gvf_pipeline.py, measured
directly against confirmed real floe masks and confirmed water blob
masks on this footage, not adjusted by eye.

Preprocessing, perspective correction and gray level enhancement, is
handled by sea_ice_preprocessing.py, the same module the other two
pipelines import from. Gray level enhancement is always applied before
seed generation here, since that is what gvf_pipeline.py itself does
before its own seeding stage runs, and matching that input is what
makes these genuinely the same seeds the paper describes reusing,
rather than a separately tuned approximation of them. Whether the
enhanced or plain color frame is handed to SAM 2 itself for mask
prediction is a separate question, controlled by APPLY_GRAY_ENHANCEMENT
below, and follows the same reasoning already used in
yoloe_pipeline.py, SAM 2 is a pretrained model and collapsing its input
to grayscale throws away information its pretrained weights likely
rely on, so this defaults to False.

A judgment call worth flagging directly. gvf_pipeline.py's own seeding
stage runs tiled_otsu_binarize and raw_local_maxima, then loops over
connected components to scope suppress_seeds_in_component to each one
in turn, skipping components below a small area floor. The three
functions themselves are imported here, so any tuning already done
inside them carries over automatically, but the connected component
loop that ties them together is reconstructed locally below rather than
imported, since it currently lives inline inside gvf_pipeline.py's
segment_frame rather than as its own function. If segment_frame's loop
has since changed, a different area floor, an added step on the binary
mask, this function needs the matching change too. Worth a direct side
by side check of seed counts between the two logs on the same frames
before relying on floe counts lining up across pipelines.

Usage

    python sam2_pipeline.py

Reads RAW_VIDEO_PATH below, runs interactive perspective calibration
once against its first frame, then processes the rest of the video.
Writes an annotated video and a per frame CSV log of floe count and
sea ice concentration.
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


# =============================================================================
# Configuration
# =============================================================================
RAW_VIDEO_PATH = "video01.mp4"                    # same raw footage as gvf_pipeline.py and yoloe_pipeline.py
OUTPUT_VIDEO_PATH = "sam2_output.mp4"
OUTPUT_CSV_PATH = "sam2_sic_log.csv"
FIRST_FRAME_TEMP_PATH = "first_frame_temp.jpg"    # written once, deleted after calibration

FRAME_STRIDE = 1  

# Checkpoint and config need to be downloaded separately per Meta's SAM 2
# setup instructions, and both need to point at files that actually exist
# on disk, these names match the SAM 2.1 large release, swap for whichever
# checkpoint size is actually installed
SAM2_CHECKPOINT = "/Users/thomasroy/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

APPLY_GRAY_ENHANCEMENT = False   # applies to the frame SAM 2 sees for mask prediction, not to seed generation, see the note near the top of this file

SEED_MIN_COMPONENT_AREA = 15   # mirrors gvf_pipeline.py's own component area floor, check this stays in sync

# Shape filters applied to each mask after SAM 2 returns it, matching
# Passerotti et al. Section 4.2.1's approach of filtering out masks that
# capture open water rather than a single floe
CONVEXITY_MIN = 0.8       # Passerotti et al. Section 4.2.1, masks at or below this are excluded, paper sourced, leave as is
CIRCULARITY_MIN = 0.5     # still a placeholder, no GVF equivalent to borrow, see the note near the top of this file, needs direct measurement against confirmed floe versus water blob masks before trusting this value
ECCENTRICITY_MAX = 0.97    # matched to gvf_pipeline.py's own ECCENTRICITY_HALLUCINATION_MIN, was 0.95, both filters screen for the same overly elongated, line like failure mode
MIN_MASK_AREA_PX = 20     # drops near empty masks that indicate a failed prompt rather than a real floe
MAX_MASK_AREA_FRACTION = 0.05   # drops masks covering more than 5 percent of the frame, added after debug_sam2_mask_overlay.py showed a single point prompt returning a mask spanning nearly the whole packed ice field, which still passed all three shape filters since the outer boundary of a dense cluster is itself fairly round and convex, with 87 to 150 seeds typically landing on one frame here, no single floe should plausibly occupy anywhere near this much of it, not yet re-tested against how many masks it actually rejects per frame

VALID_PIXEL_MIN_BRIGHTNESS = 3   # near black pixels are the warp border, not water, matches the other two pipelines

DEBUG_VERBOSE = True   # prints raw seed count against surviving floe count per frame, to separate seed generation failures from SAM 2 or shape filter failures


# =============================================================================
# Seed generation, reusing gvf_pipeline.py's own seeding stage directly
# =============================================================================
def generate_seed_points(enhanced_gray):
    """Reproduces gvf_pipeline.py's own seeding stage, tiled Otsu
    binarization, raw local maxima on the distance transform, then per
    component seed suppression, since Passerotti et al. Section 4.2.2
    describe SAM-P's seeds as coming directly from the distance
    transform matrix introduced in their Section 4.1, the same seeding
    stage already used to initialize the GVF snake. Returns a flat list
    of (x, y, d) seed tuples across the whole frame, one seed per
    distinct floe or merged cluster still needing separation."""
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


# =============================================================================
# Shape filters, matching Passerotti et al. Section 4.2.1's approach to
# removing masks that capture open water rather than a single floe
# =============================================================================
def mask_convexity(contour):
    """Ratio of contour area to its own convex hull area, close to 1 for
    the rounded, mostly convex shape of a real floe, lower for irregular
    open water regions. Matches the convexity coefficient used in
    Passerotti et al. Section 4.2.1."""
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
    approaching 1 for an increasingly elongated shape. Needs at least 5
    points to fit an ellipse at all, contours with fewer are already too
    small or degenerate to be a real floe, treated as maximally
    eccentric so they fail the filter below."""
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


# =============================================================================
# SAM 2 mask generation, one point prompt per seed
# =============================================================================
def masks_from_seeds(predictor, seed_points, frame_shape):
    """Runs SAM 2 once per seed with a single positive point prompt,
    keeps the highest scoring of the three candidate masks SAM 2 returns
    for an ambiguous single point, then applies the shape filters above.
    Seeds that fail shape filtering are discarded rather than corrected,
    consistent with the rest of this project's benchmarking philosophy,
    a model's own failure to isolate a clean floe from a given prompt is
    itself something this benchmark is measuring, not something to
    quietly patch around."""
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

        # Checked individually rather than through is_valid_floe_shape here,
        # so a seed that fails more than one criterion gets counted against
        # each one, since the point of this breakdown is finding out which
        # threshold is actually doing the rejecting, not just that one of
        # them did
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
    SIC calculation, matching the union operation Passerotti et al.
    describe in Section 4.2.1 for consolidating SAM's output into a
    single unified mask."""
    h, w = frame_shape[:2]
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in instance_masks:
        union_mask = np.maximum(union_mask, mask)
    return union_mask


def valid_pixel_mask(gray, min_brightness=VALID_PIXEL_MIN_BRIGHTNESS):
    """Near black pixels are treated as the blank border warpPerspective
    leaves outside the mapped source region, not open water, and are
    excluded from the sea ice concentration denominator. Matches the
    function of the same name in gvf_pipeline.py and yoloe_pipeline.py."""
    return (gray >= min_brightness).astype(np.uint8)


# =============================================================================
# Device selection
# =============================================================================
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


# =============================================================================
# Main video loop, calibration once, then preprocessing and SAM 2 together
# =============================================================================
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

    # Rewind to the start, the frame used for calibration still needs to
    # go through the full pipeline like every other frame
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

        # Preprocessing, using the calibration locked in above
        corrected = perspective_transform(frame, factor, x, y)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        enhanced = enhance_gray(gray)   # always used for seeding, see the note near the top of this file

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
