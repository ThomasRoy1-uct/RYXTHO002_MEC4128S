"""
gvf_stage_figure.py

Generates the individual pipeline stage panels for ONE frame of your own
footage, run through this project's actual GVF snake pipeline
(gvf_pipeline.py), for use as a worked-example figure in thesis Section
3.4.1. Panel lettering follows the convention of Passerotti et al.
(2026) Figure 4, so caption it as your own reimplementation shown in
their style, citing them, not as a reproduction of their figure.

This version also tallies WHY each snake-evolved contour was rejected,
edge touching, circularity, or eccentricity, counted separately for the
ellipse-initialised and circle-initialised branches, so you can see
where floes are actually being lost rather than just how many survive.
This is a diagnostic addition to the figure script only, it does not
change gvf_pipeline.py or the benchmark's actual segmentation.

EXCLUDE_EDGE_TOUCHING_FLOES below controls whether an edge-touching
contour is actually dropped from the panels, or only counted and kept.
It is still tallied in the rejection breakdown either way, this toggle
only changes whether it removes the contour from the figure.

Usage
-----
    python gvf_stage_figure.py

Reads RAW_VIDEO_PATH below, runs the interactive perspective calibration
once against the first frame, exactly as gvf_pipeline.py does, then
seeks to FRAME_INDEX and runs the full pipeline on that single frame,
saving each stage as its own PNG plus one assembled panel figure to
OUTPUT_DIR, and printing the rejection breakdown.
"""

import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sea_ice_preprocessing import initial_perspective_transform, perspective_transform, enhance_gray
from gvf_pipeline import (
    tiled_otsu_binarize, raw_local_maxima, suppress_seeds_in_component,
    separate_merged_object, make_circular_contour, make_elliptical_contour,
    compute_edge_map, compute_gvf, evolve_snake, touches_edge,
    circularity, eccentricity,
    CIRCULARITY_HALLUCINATION_MAX, ECCENTRICITY_HALLUCINATION_MIN,
    MIN_COMPONENT_AREA, ENHANCE_GRAY_ENABLED, APPLY_BILATERAL, APPLY_DIFFUSION,
    APPLY_CLAHE, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RAW_VIDEO_PATH = "video01.mp4"     # same footage as gvf_pipeline.py
FRAME_INDEX = 0                   # which frame to build the figure from
OUTPUT_DIR = "gvf_stage_figure"
FIRST_FRAME_TEMP_PATH = "first_frame_temp.jpg"

EXCLUDE_EDGE_TOUCHING_FLOES = False   # False: edge-touching contours are still tallied below, but kept in the figure

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Re-run the per component loop from gvf_pipeline.segment_frame(), but keep
# every intermediate result, and tally rejection reasons per branch
# ----------------------------------------------------------------------------
def classify_and_filter(c1, w, h, counts, branch):
    """Mirrors gvf_pipeline.is_hallucination(), but tallies which check
    actually failed instead of collapsing straight to a bool. A contour
    can trip more than one check; each is counted independently, so the
    three counts can sum to more than the number of rejected contours.

    Edge touching is always counted, but only actually rejects the
    contour when EXCLUDE_EDGE_TOUCHING_FLOES is True."""
    rejected = False
    if touches_edge(c1, w, h):
        counts[branch]["edge"] += 1
        if EXCLUDE_EDGE_TOUCHING_FLOES:
            rejected = True
    circ = circularity(c1)
    if circ > CIRCULARITY_HALLUCINATION_MAX:
        counts[branch]["circularity"] += 1
        rejected = True
    ecc = eccentricity(c1)
    if ecc >= ECCENTRICITY_HALLUCINATION_MIN:
        counts[branch]["eccentricity"] += 1
        rejected = True
    return not rejected


def segment_frame_with_stages(enhanced):
    h, w = enhanced.shape
    binary = tiled_otsu_binarize(enhanced)
    raw_seeds = raw_local_maxima(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    fx, fy = compute_edge_map(enhanced)
    gvf_u, gvf_v = compute_gvf(fx, fy)

    kept_seeds = []
    direct_contours = []
    ellipse_initial_contours = []
    circle_initial_contours = []
    ellipse_snake_contours = []
    circle_snake_contours = []

    counts = {
        "ellipse": {"edge": 0, "circularity": 0, "eccentricity": 0, "attempted": 0, "kept": 0},
        "circle": {"edge": 0, "circularity": 0, "eccentricity": 0, "attempted": 0, "kept": 0},
    }

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue

        comp_mask = (labels == i).astype(np.uint8)
        seeds_in_comp_raw = [s for s in raw_seeds if comp_mask[int(round(s[1])), int(round(s[0]))] == 1]
        seeds_in_comp = suppress_seeds_in_component(seeds_in_comp_raw)
        kept_seeds.extend(seeds_in_comp)

        if len(seeds_in_comp) <= 1:
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = contours[0].reshape(-1, 2).astype(np.float64)
                if not EXCLUDE_EDGE_TOUCHING_FLOES or not touches_edge(c, w, h):
                    direct_contours.append(c)
            continue

        ellipses = separate_merged_object(comp_mask, len(seeds_in_comp))
        if ellipses:
            for e in ellipses:
                c0 = make_elliptical_contour(*e)
                ellipse_initial_contours.append(c0)
                c1 = evolve_snake(c0, gvf_u, gvf_v)
                counts["ellipse"]["attempted"] += 1
                if classify_and_filter(c1, w, h, counts, "ellipse"):
                    ellipse_snake_contours.append(c1)
                    counts["ellipse"]["kept"] += 1
        else:
            for (sx, sy, d) in seeds_in_comp:
                c0 = make_circular_contour(sx, sy, max(d, 3.0))
                circle_initial_contours.append(c0)
                c1 = evolve_snake(c0, gvf_u, gvf_v)
                counts["circle"]["attempted"] += 1
                if classify_and_filter(c1, w, h, counts, "circle"):
                    circle_snake_contours.append(c1)
                    counts["circle"]["kept"] += 1

    return {
        "binary": binary,
        "seeds": kept_seeds,
        "direct_contours": direct_contours,
        "ellipse_initial_contours": ellipse_initial_contours,
        "circle_initial_contours": circle_initial_contours,
        "ellipse_snake_contours": ellipse_snake_contours,
        "circle_snake_contours": circle_snake_contours,
        "counts": counts,
    }


# ----------------------------------------------------------------------------
# Panel rendering helpers
# ----------------------------------------------------------------------------
def draw_contours(base_gray, contours, color=(0, 0, 255)):
    canvas = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    for c in contours:
        pts = c.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1)
    return canvas


def draw_seeds(base_gray, seeds, color=(0, 0, 255), radius=2):
    canvas = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    for (x, y, d) in seeds:
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, -1)
    return canvas


def filled_mask_image(base_gray, contours):
    mask = np.zeros_like(base_gray, dtype=np.uint8)
    for c in contours:
        pts = c.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def save(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    return path


def build_panel_figure(panel_paths, out_path):
    rows = [
        [panel_paths["a"]],
        [panel_paths["b"]],
        [panel_paths["c"], panel_paths["d"], panel_paths["e"]],
        [panel_paths["f"], panel_paths["g"], panel_paths["h"]],
        [panel_paths["i"]],
    ]
    n_rows = len(rows)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 14))
    for r, row_paths in enumerate(rows):
        n_in_row = len(row_paths)
        offset = (n_cols - n_in_row) // 2
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            idx = c - offset
            if 0 <= idx < n_in_row:
                img = cv2.cvtColor(cv2.imread(row_paths[idx]), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                label = "abcdefghi"[sum(len(x) for x in rows[:r]) + idx]
                ax.text(0.03, 0.06, f"({label})", transform=ax.transAxes,
                        fontsize=11, color="white", va="bottom", ha="left",
                        bbox=dict(facecolor="0.35", edgecolor="none", pad=2))
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
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
        raise RuntimeError("Could not read frame " + str(FRAME_INDEX) + " of " + RAW_VIDEO_PATH)

    corrected = perspective_transform(frame, factor, x, y)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    if ENHANCE_GRAY_ENABLED:
        enhanced = enhance_gray(gray, apply_bilateral=APPLY_BILATERAL, apply_diffusion=APPLY_DIFFUSION,
                                 apply_clahe=APPLY_CLAHE, clahe_clip_limit=CLAHE_CLIP_LIMIT,
                                 clahe_tile_size=CLAHE_TILE_SIZE)
    else:
        enhanced = gray

    stages = segment_frame_with_stages(enhanced)

    panel_paths = {}
    panel_paths["a"] = save("a_enhanced.png", enhanced)

    binary_vis = (stages["binary"] * 255).astype(np.uint8)
    panel_paths["b"] = save("b_seeds_on_binary.png", draw_seeds(binary_vis, stages["seeds"]))

    panel_paths["c"] = save("c_direct_perimeters.png",
                             draw_contours(enhanced, stages["direct_contours"]))
    panel_paths["d"] = save("d_elliptical_initial.png",
                             draw_contours(enhanced, stages["ellipse_initial_contours"]))
    panel_paths["e"] = save("e_circular_initial.png",
                             draw_contours(enhanced, stages["circle_initial_contours"]))

    panel_paths["f"] = save("f_direct_floes.png",
                             filled_mask_image(enhanced, stages["direct_contours"]))
    panel_paths["g"] = save("g_gvf_from_ellipse.png",
                             filled_mask_image(enhanced, stages["ellipse_snake_contours"]))
    panel_paths["h"] = save("h_gvf_from_circle.png",
                             filled_mask_image(enhanced, stages["circle_snake_contours"]))

    all_contours = stages["direct_contours"] + stages["ellipse_snake_contours"] + stages["circle_snake_contours"]
    panel_paths["i"] = save("i_final_segmentation.png",
                             filled_mask_image(enhanced, all_contours))

    build_panel_figure(panel_paths, os.path.join(OUTPUT_DIR, "assembled_panel_figure.png"))

    print("Panels written to", OUTPUT_DIR)
    print("Seeds kept:", len(stages["seeds"]))
    print("Direct (isolated) floes:", len(stages["direct_contours"]))
    print()
    for branch in ("ellipse", "circle"):
        c = stages["counts"][branch]
        print(f"{branch.capitalize()} branch: {c['attempted']} contours attempted, {c['kept']} kept")
        print(f"  rejected for touching the frame edge: {c['edge']}")
        print(f"  rejected for circularity > {CIRCULARITY_HALLUCINATION_MAX}: {c['circularity']}")
        print(f"  rejected for eccentricity >= {ECCENTRICITY_HALLUCINATION_MIN}: {c['eccentricity']}")
        print("  (a contour can trip more than one check, so these can sum to more than the number rejected)")
        print()


if __name__ == "__main__":
    main()