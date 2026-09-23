"""
Binary mask and final contour comparison for gvf_pipeline.py, checking
directly which stage is losing ice area. GVF's SIC has sat well below
both SAM 2 and YOLOE all session, even on frames where GVF's own floe
count is higher than YOLOE's, meaning area, not floe count, is the
actual gap. Three candidate explanations have built up without being
told apart yet, Passerotti's own documented GVF weakness, the ellipse
fit used for merged regions since the ABS_SEP_SCALE_FACTOR change
potentially fitting tighter than the true combined footprint, and
APPLY_BINARY_OPENING eating thin or small ice at the very first stage
before anything else runs.

Produces one image with four panels, the plain corrected frame for
reference, the binary mask straight out of tiled_otsu_binarize before
any opening, the same binary mask after APPLY_BINARY_OPENING runs, and
the final floe mask actually used for the SIC calculation, built from
every direct and snake contour together. Comparing these directly shows
where area gets lost. Area already missing before opening runs points
at thresholding itself, tile size or the enhancement feeding it. A big
drop specifically after opening points at that one step. Only the
final panel losing area relative to the post opening binary mask points
at ellipse fitting and snake evolution instead.

Usage

    python debug_gvf_binary_stages.py

Set FRAME_INDEX below to inspect a different frame. Prints pixel counts
at each stage alongside the saved image, so the visual comparison and
the actual numbers sit together.
"""

import os

import cv2
import numpy as np

from sea_ice_preprocessing import initial_perspective_transform, perspective_transform, enhance_gray
from gvf_pipeline import (
    tiled_otsu_binarize, segment_frame, contours_to_mask, APPLY_BINARY_OPENING,
    raw_local_maxima, suppress_seeds_in_component, compute_edge_map, compute_gvf,
    separate_merged_object, make_circular_contour, make_elliptical_contour,
    evolve_snake, touches_edge, is_hallucination, MIN_COMPONENT_AREA,
)

RAW_VIDEO_PATH = "video01.mp4"
FRAME_INDEX = 0
FIRST_FRAME_TEMP_PATH = "first_frame_temp.jpg"


def to_overlay(mask, base):
    overlay = base.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 2] = 255
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
    blended = cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0)
    overlay[mask_3ch] = blended[mask_3ch]
    return overlay


def per_component_breakdown(enhanced):
    """Mirrors gvf_pipeline.py's segment_frame loop exactly, function by
    function, but also captures per component binary area, seed count,
    and final contour area along the way, none of which segment_frame
    itself exposes, since it only returns the final flat contour lists.
    Isolated floes, one seed, go through cv2.findContours directly on
    the binary blob, no ellipse or snake involved, so their shrinkage
    should sit close to zero if the theory that only merged, multi seed
    regions are losing area holds. Printed sorted by shrinkage, binary
    area minus final area, largest first, so the worst affected floes
    show up at the top rather than needing to be found by eye."""
    h, w = enhanced.shape
    binary = tiled_otsu_binarize(enhanced)
    raw_seeds = raw_local_maxima(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    fx, fy = compute_edge_map(enhanced)
    gvf_u, gvf_v = compute_gvf(fx, fy)

    rows = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue

        comp_mask = (labels == i).astype(np.uint8)
        binary_area = int(stats[i, cv2.CC_STAT_AREA])
        seeds_in_comp_raw = [s for s in raw_seeds if comp_mask[int(round(s[1])), int(round(s[0]))] == 1]
        seeds_in_comp = suppress_seeds_in_component(seeds_in_comp_raw)
        num_seeds = len(seeds_in_comp)

        if num_seeds <= 1:
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            c = contours[0].reshape(-1, 2).astype(np.float64)
            if touches_edge(c, w, h):
                continue
            final_area = int(cv2.contourArea(c.astype(np.float32)))
            rows.append((binary_area, final_area, num_seeds, "direct"))
            continue

        ellipses = separate_merged_object(comp_mask, num_seeds)
        if ellipses:
            initial_contours = [make_elliptical_contour(*e) for e in ellipses]
        else:
            initial_contours = [make_circular_contour(sx, sy, max(d, 3.0)) for (sx, sy, d) in seeds_in_comp]

        component_final_area = 0
        kept_any = False
        for c0 in initial_contours:
            c1 = evolve_snake(c0, gvf_u, gvf_v)
            if not touches_edge(c1, w, h) and not is_hallucination(c1):
                component_final_area += int(cv2.contourArea(c1.astype(np.float32)))
                kept_any = True
        if kept_any:
            rows.append((binary_area, component_final_area, num_seeds, "snake"))

    rows.sort(key=lambda r: r[0] - r[1], reverse=True)
    print("Per component breakdown, sorted by shrinkage, binary area minus final area, largest first")
    for binary_area, final_area, num_seeds, kind in rows[:15]:
        shrink = binary_area - final_area
        pct = 100.0 * shrink / binary_area if binary_area else 0.0
        print("  binary_area=" + str(binary_area) + " final_area=" + str(final_area)
              + " seeds=" + str(num_seeds) + " path=" + kind
              + " shrink=" + str(shrink) + " (" + f"{pct:.1f}" + "%)")

    direct_rows = [r for r in rows if r[3] == "direct"]
    if direct_rows:
        avg_direct_shrink_pct = sum(100.0 * (b - f) / b for b, f, _, _ in direct_rows if b) / len(direct_rows)
        print("average shrink for direct, single seed floes, " + f"{avg_direct_shrink_pct:.1f}"
              + "%, should sit close to zero if isolated floes are unaffected")


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

    binary_before_opening = tiled_otsu_binarize(enhanced, apply_opening=False)
    binary_after_opening = tiled_otsu_binarize(enhanced, apply_opening=True)

    direct_contours, snake_contours = segment_frame(enhanced)
    final_mask = contours_to_mask(direct_contours + snake_contours, gray.shape)

    print("APPLY_BINARY_OPENING is currently " + str(APPLY_BINARY_OPENING) + " in gvf_pipeline.py, "
          "this comparison builds both versions directly regardless, to isolate that step's own effect")
    print("ice pixels before opening, " + str(int(binary_before_opening.sum())))
    print("ice pixels after opening, " + str(int(binary_after_opening.sum()))
          + ", change=" + str(int(binary_after_opening.sum()) - int(binary_before_opening.sum())))
    print("ice pixels in final contour mask, " + str(int(final_mask.sum()))
          + ", change from post opening binary, "
          + str(int(final_mask.sum()) - int(binary_after_opening.sum())))

    per_component_breakdown(enhanced)

    panels = [
        (corrected.copy(), "plain frame"),
        (to_overlay(binary_before_opening, corrected), "binary, before opening"),
        (to_overlay(binary_after_opening, corrected), "binary, after opening"),
        (to_overlay(final_mask, corrected), "final contour mask"),
    ]
    labelled = []
    for panel, label in panels:
        panel = panel.copy()
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        labelled.append(panel)

    top = np.hstack([labelled[0], labelled[1]])
    bottom = np.hstack([labelled[2], labelled[3]])
    combined = np.vstack([top, bottom])
    cv2.imwrite("debug_gvf_binary_stages.png", combined)
    print("Saved debug_gvf_binary_stages.png.")


if __name__ == "__main__":
    main()