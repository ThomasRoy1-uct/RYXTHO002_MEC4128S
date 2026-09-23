"""
GVF snake video pipeline, matched to Passerotti et al. (2026, Earth and
Space Science, "Sea Ice Floe Segmentation in Close Range Optical Imagery
Using Active Contour and Foundation Models"), Section 4.1.

Preprocessing (perspective correction and gray level enhancement) is
handled by sea_ice_preprocessing.py, so this script runs calibration once
against the first frame of the raw video, then every subsequent frame is
corrected, enhanced, and segmented in the same loop.

ENHANCE_GRAY_ENABLED and the three APPLY_* flags below control gray
level enhancement independently, so each step's contribution can be
tested in isolation on the same footage. Setting ENHANCE_GRAY_ENABLED to
False falls back to a plain grayscale conversion, matching the pipeline's
behaviour before this feature was added.

This replaces an earlier seeding method (k means clustering on HSV Value
and Saturation, then a flat distance transform cutoff for seeds) with the
paper's actual pipeline.

  1. Binarize via Otsu thresholding, run per tile rather than globally,
     on the (optionally enhanced) grayscale frame. The paper does this
     over 12 x 12 m clusters of about 20 floes each, since a small
     region's gray scale histogram has clearer modes than one histogram
     over a whole image of mixed ice concentration. No metre per pixel
     calibration exists for this camera yet, so tiles are defined in
     pixels instead, see TILE_SIZE_PX below. A light morphological
     opening is then applied to the combined binary mask, matching a
     post threshold cleanup step in Liss (2023), see
     APPLY_BINARY_OPENING below, this is not part of the paper's method
     itself, it is a noise reduction step borrowed from the predecessor
     project.

  2. Distance transform the binary mask, then find every true local
     maximum of that distance map, a pixel counts only if its value is
     greater than or equal to every neighbour's in a small window. Raw
     local maxima are found across the whole frame first, then
     deduplicated separately within each connected component, since what
     counts as too close together to be a separate object depends on how
     large that particular object is, not on a single value shared
     across the whole frame.

     Within one connected component, each pair of seeds is compared
     against a separation floor based on their own shared depth, the
     smaller of the two, not a single deepest value taken from elsewhere
     in the same component. An earlier version of this function used one
     deepest value per component, which worked when a component
     contained one floe, but broke down for a component containing
     several floes of very different sizes chained together, since a
     large floe's own depth elsewhere in the blob set a threshold far too
     loose to catch two shallow, closely spaced seeds sitting on a
     smaller, unrelated floe in the same blob, see
     suppress_seeds_in_component below.

  3. For each connected foreground blob, if it contains exactly one seed
     after suppression, it is an isolated floe, segmented directly from
     the binarization, no snake needed. If it contains more than one
     seed, it is a merged region, and morphological erosion is attempted
     to split it into separate pieces. Where erosion succeeds, only the
     largest num_seeds fragments at that point are kept, see the note in
     separate_merged_object below on why that cap was added. Each kept
     piece gets an initial contour as an ellipse fitted to match its
     second moment (covariance matrix), not a boundary fit. Where erosion
     cannot cleanly separate the pieces, each seed instead gets a
     circular initial contour, sized from its own distance transform
     value.

  4. The GVF snake then expands every non isolated initial contour on the
     same (optionally enhanced) frame, using the Xu and Prince (1998)
     diffusion equation, with a numerically stable time step bounded by
     both the diffusion term and the reaction term.

  5. Converged contours are filtered for hallucinations (near perfect
     circles, flagged by circularity over 1.0, or line like slivers,
     flagged by eccentricity of 0.9 or higher) and for floes with an open
     boundary at the image edge, both are excluded from the final count
     and from the sea ice concentration calculation.

A note on SIC, since this went back and forth once already. A stage by
stage check with debug_gvf_binary_stages.py found that merged, multi
seed regions lose real area during erosion based separation and ellipse
fitting, up to 82 percent on the worst measured case, while the binary
mask itself tracks visible floe boundaries closely at every stage. That
led to briefly measuring SIC straight from the binary mask instead of
the final contour mask, which fixed the measured shrinkage but caused
GVF's SIC to overshoot past both SAM 2 and YOLOE on most frames instead
of landing near them, and meant GVF's SIC no longer measured the same
thing Passerotti et al. report for GVF, whose own concentration figure
comes from the same floe level contours as their floe count, carrying
the same missing and misinterpreted floe errors baked in. Raised with
the project supervisor, the decision was to keep SIC contour based, as
below, matching the paper's own method, and to note the binary mask
shrinkage and the option of measuring SIC from it directly in the
thesis recommendations rather than building it into the comparison
itself.

A related option, replacing erosion based separation with watershed
reconstruction, letting eroded seed pieces flood fill back out to their
true, un-eroded boundary rather than fitting an ellipse to the shrunk
piece, was also raised and set aside for the same reason, the brief was
to keep GVF implemented the same way Passerotti et al. describe it,
not a different, independently improved method.

A further option, feeding the ML pipelines the binary mask directly
rather than the corrected colour frame, since it separates ice from
water cleanly, was also discussed and declined for the comparison
itself, the point of this benchmark is what each model can do with a
normal frame, not what they could do with GVF's own preprocessing
handed to them, also worth a mention in recommendations rather than
part of the result.

What this script intentionally does NOT try to match, the paper reports
this per floe process running some tens of seconds per floe (Section
4.1), evaluated on a set of still images, not a continuous video. Applied
to every frame of a full video at that rate would be impractical. This
implementation is substantially faster in testing (roughly 2 seconds per
500 x 500 frame with several dozen floes, on CPU), but a full video can
still be slow. FRAME_STRIDE below controls how many frames are actually
processed, skipping frames is a video specific compromise the paper's own
still image method has no need for, everything within a processed frame
follows the paper's method as described above.

Usage
-----
    python gvf_pipeline.py

Reads RAW_VIDEO_PATH below, runs interactive perspective calibration once
against its first frame, then processes the rest of the video. Writes an
annotated video and a per frame CSV log of floe count and sea ice
concentration.
"""

import csv
import os
import time

import cv2
import numpy as np

from sea_ice_preprocessing import initial_perspective_transform, perspective_transform, enhance_gray


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RAW_VIDEO_PATH = "video01.mp4"                   # your original, unprocessed footage
OUTPUT_VIDEO_PATH = "gvf_snake_output.mp4"
OUTPUT_CSV_PATH = "gvf_snake_sic_log.csv"
FIRST_FRAME_TEMP_PATH = "first_frame_temp.jpg"   # written once, deleted after calibration

FRAME_STRIDE = 5          # process every Nth frame, see the note above on runtime
TILE_SIZE_PX = 150        # local Otsu tile size, paper uses 12x12 m tiles of about 20 floes
APPLY_BINARY_OPENING = True   # toggle independently to test its effect in isolation from other changes
OPENING_KERNEL_SIZE = 5       # matches Liss (2023), kernel size five for erosion and dilation noise cleanup

# Gray level enhancement toggles, see enhance_gray in sea_ice_preprocessing.py
ENHANCE_GRAY_ENABLED = True   # master switch, False falls back to plain grayscale
APPLY_BILATERAL = True
APPLY_DIFFUSION = True
APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0        # lower this first if CLAHE is sharpening false internal boundaries
CLAHE_TILE_SIZE = 8

MIN_SEED_DISTANCE = 4.0   # raised from 1.0, suppresses secondary peaks along a single floe's medial axis
ABS_SEP_SCALE_FACTOR = 3.6   # raised from 2.0, measured directly via debug_gvf_seeds.py against six confirmed over split floes, ratios 2.20 to 3.41, and two confirmed genuinely separate pairs, ratios 4.28 and 5.72, 3.5 sits inside that gap and was settled on by checking the resulting output visually
ABS_SEP_MIN_PX = 8           # lower bound, guards very shallow seeds sitting just above MIN_SEED_DISTANCE
MIN_COMPONENT_AREA = 15   # connected components smaller than this are treated as noise, not floes

N_POINTS = 40              # points per closed contour
MU = 0.15                  # GVF regularisation weight
GVF_ITER = 120              # GVF diffusion iterations per frame
ALPHA, BETA, GAMMA = 0.02, 0.02, 1.0   # snake tension, rigidity, step size weights
SNAKE_ITER = 150            # snake evolution iterations per initial contour

EROSION_KERNEL_SIZE = 3     # structuring element size for the merged object separation attempt
MAX_EROSION_ITER = 15       # erosion attempts before falling back to circles

CIRCULARITY_HALLUCINATION_MAX = 1.0    # contours more circular than a true circle, discretisation artefact
ECCENTRICITY_HALLUCINATION_MIN = 0.9   # contours this elongated are treated as line like noise
EDGE_MARGIN_PX = 2                     # contours within this many pixels of the frame edge are excluded

VALID_PIXEL_MIN_BRIGHTNESS = 3   # near black pixels are treated as masked out (warp border), not water, checked on the plain grayscale frame, not the enhanced one


# ----------------------------------------------------------------------------
# Stage 1, tiled Otsu binarization
# ----------------------------------------------------------------------------
def tiled_otsu_binarize(gray, tile_size=TILE_SIZE_PX, apply_opening=APPLY_BINARY_OPENING,
                         opening_kernel_size=OPENING_KERNEL_SIZE):
    """Foreground and background split via Otsu, run per tile. Matches
    Section 4.1, run over small clusters so the local histogram has
    clearer modes than a single global histogram over a full frame of
    mixed ice concentration would.

    apply_opening runs a light morphological opening (erosion followed by
    dilation) on the combined binary mask afterwards, matching Liss
    (2023)'s post threshold cleanup step, kernel size five there also.
    This is not part of Passerotti et al. (2026) Section 4.1 itself, it
    is a noise reduction step borrowed from the predecessor project.

    Confirmed directly via debug_gvf_binary_stages.py to track visible
    floe boundaries closely both before and after opening runs. This
    stage measures ice extent well, the SIC accuracy question this
    project ran into lived downstream, in separation and fitting, not
    here, see the note near the top of this file for how that was
    resolved."""
    h, w = gray.shape
    binary = np.zeros_like(gray, dtype=np.uint8)
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
            tile = gray[y0:y1, x0:x1]
            if tile.size == 0 or tile.max() == tile.min():
                continue
            _, tile_bin = cv2.threshold(tile, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary[y0:y1, x0:x1] = tile_bin

    if apply_opening:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


# ----------------------------------------------------------------------------
# Stage 2, distance transform and per component local maxima seeding
# ----------------------------------------------------------------------------
def raw_local_maxima(binary, neighborhood=5, min_dist_value=MIN_SEED_DISTANCE):
    """Finds every pixel that is a true local maximum of the distance
    transform, matching the paper's local maxima in this distance map
    serve as seeds. No suppression is applied here, that happens
    separately per connected component in suppress_seeds_in_component,
    since a single global separation distance cannot fit floes of very
    different sizes appearing within the same frame."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    local_max = cv2.dilate(dist, np.ones((neighborhood, neighborhood), np.uint8))
    maxima_mask = ((dist == local_max) & (dist > min_dist_value)).astype(np.uint8)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(maxima_mask)

    seeds = []
    for i in range(1, num_labels):
        x, y = centroids[i]
        d = float(dist[int(round(y)), int(round(x))])
        seeds.append((x, y, d))
    return seeds


def suppress_seeds_in_component(seeds_in_comp, scale_factor=ABS_SEP_SCALE_FACTOR, min_sep_px=ABS_SEP_MIN_PX):
    """Deduplicates seeds belonging to one connected component, one floe
    or one merged cluster, keeping the deepest seed in any cluster of
    seeds too close together to be genuinely separate objects.

    Each pair of seeds is compared against a separation floor based on
    their own shared depth, the smaller of the two, not a single value
    borrowed from elsewhere in the same component. An earlier version of
    this function used one deepest value for the whole component, which
    worked for a component containing one floe, but broke down for a
    component containing several floes of very different sizes chained
    together, since a large floe's own depth elsewhere in the blob set a
    threshold far too loose to catch two shallow, closely spaced seeds on
    a smaller, unrelated floe in the same blob.

    scale_factor was retuned directly against measured spacing and depth
    on real footage rather than guessed, twice now. The first pass, on
    elongated floes, adjacent medial axis peaks were found roughly 24 to
    29 pixels apart while carrying depth values around 14 to 21, a ratio
    of roughly 1.4 to 2.0, landing on 2.0. A second pass measured against
    a different, more irregular floe shape found ratios of 2.20 to 3.41
    across six further confirmed over split cases, alongside two
    confirmed genuinely separate pairs at 4.28 and 5.72, landing on 3.5
    as a value inside that gap, checked visually against the resulting
    output. This remains a measurement based value, not a derived
    constant, and the gap between the two groups is not fully mapped,
    since the should be one side extended further every time more
    examples were checked, worth treating as provisional rather than
    settled if further examples turn up beyond this range."""
    if not seeds_in_comp:
        return []
    seeds_sorted = sorted(seeds_in_comp, key=lambda s: -s[2])
    kept = []
    for x, y, d in seeds_sorted:
        is_duplicate = False
        for kx, ky, kd in kept:
            min_sep = max(scale_factor * min(d, kd), min_sep_px)
            if np.hypot(x - kx, y - ky) < min_sep:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((x, y, d))
    return kept


# ----------------------------------------------------------------------------
# Stage 3, classify isolated vs merged objects, separate merged ones
# ----------------------------------------------------------------------------
def ellipse_from_moments(mask):
    """Fits an ellipse by matching the second moment of the object's
    covariance matrix, the phrase Section 4.1 uses, rather than fitting to
    boundary points. Returns (cx, cy, semi_major, semi_minor, angle_rad),
    or None if the mask has too few pixels to define a covariance."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return None
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    semi_major = 2.0 * np.sqrt(max(eigvals[0], 1.0))
    semi_minor = 2.0 * np.sqrt(max(eigvals[1], 1.0))
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    return cx, cy, semi_major, semi_minor, angle


def separate_merged_object(component_mask, num_seeds, max_erosion_iter=MAX_EROSION_ITER,
                            kernel_size=EROSION_KERNEL_SIZE, min_area=MIN_COMPONENT_AREA):
    """Attempts to split a merged, multi seed blob via morphological
    erosion, matching Section 4.1's separating adjacent objects by
    smoothing their edges through a morphological erosion process.
    Returns a list of ellipse parameters, one per separated piece, or an
    empty list if erosion never reaches num_seeds separate pieces before
    the object erodes away, in which case the caller falls back to
    circles, matching the paper's description for floes that share
    significant portions of their edges and cannot be separated
    successfully using morphological erosion.

    Only the num_seeds largest qualifying fragments are kept once the
    threshold is reached. The stopping check is len(pieces) >= num_seeds,
    not equality, and the exact iteration that first satisfies it can
    include small noise fragments alongside the real pieces, especially
    when num_seeds is low and the loop stops after only a little erosion.
    Keeping only the largest num_seeds fragments discards that noise
    rather than returning it as extra floes.

    Each ellipse is fitted to a piece after erosion has already shrunk
    it, not against the original, pre erosion boundary, so it does come
    out smaller than the true floe, more so the more passes a heavily
    merged region needed, measured directly via
    debug_gvf_binary_stages.py. A margin compensation added to
    ellipse_from_moments to correct for this was tried and made no
    measurable difference to that shrinkage, the snake's own tension and
    rigidity convergence appears to be the actual bottleneck rather than
    the initial ellipse size, so the compensation was removed again
    rather than left in as dead weight. SIC is measured from this final
    contour mask, so this shrinkage does affect the concentration
    figure, not just the shape of the drawn contours, see the note near
    the top of this file for why that trade off was accepted."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = component_mask.copy()

    for _ in range(max_erosion_iter):
        eroded = cv2.erode(eroded, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
        pieces = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]

        if len(pieces) >= num_seeds:
            pieces_by_area = sorted(pieces, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            pieces_by_area = pieces_by_area[:num_seeds]
            ellipses = []
            for i in pieces_by_area:
                piece_mask = (labels == i).astype(np.uint8)
                ellipse = ellipse_from_moments(piece_mask)
                if ellipse is not None:
                    ellipses.append(ellipse)
            return ellipses

        if eroded.sum() == 0:
            break

    return []


# ----------------------------------------------------------------------------
# Stage 4, GVF diffusion (numerically stable) and snake evolution
# ----------------------------------------------------------------------------
def make_circular_contour(cx, cy, radius, n_points=N_POINTS):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)


def make_elliptical_contour(cx, cy, a, b, theta, n_points=N_POINTS):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x_local = a * np.cos(angles)
    y_local = b * np.sin(angles)
    x = cx + x_local * np.cos(theta) - y_local * np.sin(theta)
    y = cy + x_local * np.sin(theta) + y_local * np.cos(theta)
    return np.stack([x, y], axis=1)


def compute_edge_map(intensity):
    smoothed = cv2.GaussianBlur(intensity.astype(np.float64), (5, 5), 1.0)
    Ix = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    f = np.sqrt(Ix ** 2 + Iy ** 2)
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    fx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    fy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    return fx, fy


def laplacian(field):
    """5 point discrete Laplacian via np.roll. This wraps at the image
    edges rather than using a zero flux boundary, a minor artefact right
    at the frame border, not usually visible mid frame."""
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4 * field
    )


def compute_gvf(fx, fy, mu=MU, n_iter=GVF_ITER):
    """Xu and Prince (1998) GVF diffusion, with a numerically stable time
    step. The step is bounded by both the diffusion term (mu) and the
    reaction term (b = fx squared plus fy squared), the latter was
    previously left unbounded, which let the field blow up past 1e100 in
    testing over 120 iterations. This version keeps the field's magnitude
    bounded through all iterations."""
    u, v = fx.copy(), fy.copy()
    b = fx ** 2 + fy ** 2
    c1, c2 = b * fx, b * fy
    dt = min(0.25 / mu, 1.0 / (b.max() + 1e-8), 1.0)
    for _ in range(n_iter):
        u = u + dt * (mu * laplacian(u) - b * u + c1)
        v = v + dt * (mu * laplacian(v) - b * v + c2)
    mag = np.sqrt(u ** 2 + v ** 2) + 1e-8
    return u / mag, v / mag


def build_internal_matrix(n_points, alpha, beta, gamma, dt=1.0):
    """Circulant tension and rigidity internal energy matrix, inverted
    once since it depends only on fixed constants, reused for every
    contour, every frame."""
    A2 = -2 * np.eye(n_points) + np.eye(n_points, k=1) + np.eye(n_points, k=-1)
    A2 += np.eye(n_points, k=n_points - 1) + np.eye(n_points, k=-(n_points - 1))
    A4 = 6 * np.eye(n_points) - 4 * (np.eye(n_points, k=1) + np.eye(n_points, k=-1))
    A4 += (np.eye(n_points, k=2) + np.eye(n_points, k=-2))
    A4 += -4 * (np.eye(n_points, k=n_points - 1) + np.eye(n_points, k=-(n_points - 1)))
    A4 += (np.eye(n_points, k=n_points - 2) + np.eye(n_points, k=-(n_points - 2)))
    A = alpha * A2 - beta * A4
    I = np.eye(n_points)
    return np.linalg.inv(I - (dt / gamma) * A)


INTERNAL_INV = build_internal_matrix(N_POINTS, ALPHA, BETA, GAMMA)  # module level constant, computed once


def sample_bilinear(field, xs, ys):
    h, w = field.shape
    xs_c = np.clip(xs, 0, w - 1)
    ys_c = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs_c).astype(int)
    y0 = np.floor(ys_c).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs_c - x0
    wy = ys_c - y0
    top = field[y0, x0] * (1 - wx) + field[y0, x1] * wx
    bottom = field[y1, x0] * (1 - wx) + field[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def evolve_snake(contour, gvf_u, gvf_v, internal_inv=INTERNAL_INV, n_iterations=SNAKE_ITER,
                  dt=1.0, gamma=GAMMA):
    C = contour.copy()
    for _ in range(n_iterations):
        Fx = sample_bilinear(gvf_u, C[:, 0], C[:, 1])
        Fy = sample_bilinear(gvf_v, C[:, 0], C[:, 1])
        C[:, 0] = internal_inv @ (C[:, 0] + (dt / gamma) * Fx)
        C[:, 1] = internal_inv @ (C[:, 1] + (dt / gamma) * Fy)
    return C


# ----------------------------------------------------------------------------
# Stage 5, hallucination filtering and edge exclusion
# ----------------------------------------------------------------------------
def circularity(contour):
    """4 pi Area over Perimeter squared. A true circle is 1.0,
    discretisation noise on tiny or near circular contours can push this
    slightly over 1, the signature Section 4.1 uses to flag hallucinated
    near perfect circles."""
    pts = contour.astype(np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(pts)
    perimeter = cv2.arcLength(pts, closed=True)
    if perimeter <= 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def eccentricity(contour):
    """0 for a circle, approaching 1 for a flattened, line like sliver."""
    pts = contour.astype(np.float32).reshape(-1, 1, 2)
    if len(pts) < 5:
        return 0.0
    (_, _), (major, minor), _ = cv2.fitEllipse(pts)
    major, minor = max(major, minor), min(major, minor)
    if major <= 0:
        return 0.0
    ratio = minor / major
    return float(np.sqrt(max(1 - ratio ** 2, 0.0)))


def touches_edge(contour, width, height, margin=EDGE_MARGIN_PX):
    return (
        np.any(contour[:, 0] <= margin) or np.any(contour[:, 0] >= width - 1 - margin) or
        np.any(contour[:, 1] <= margin) or np.any(contour[:, 1] >= height - 1 - margin)
    )


def is_hallucination(contour, circularity_max=CIRCULARITY_HALLUCINATION_MAX,
                      eccentricity_min=ECCENTRICITY_HALLUCINATION_MIN):
    return circularity(contour) > circularity_max or eccentricity(contour) >= eccentricity_min


# ----------------------------------------------------------------------------
# Full per frame pipeline
# ----------------------------------------------------------------------------
def segment_frame(enhanced):
    """Runs the full Section 4.1 pipeline on one frame, returns
    (direct_contours, snake_contours), both lists of (n_points, 2)
    arrays, already filtered for hallucinations and edge touching floes.
    Both the binarization and the GVF force field are computed from
    whatever frame is passed in, plain or enhanced, matching the paper's
    use of one common input for both stages."""
    h, w = enhanced.shape
    binary = tiled_otsu_binarize(enhanced)
    raw_seeds = raw_local_maxima(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    fx, fy = compute_edge_map(enhanced)
    gvf_u, gvf_v = compute_gvf(fx, fy)

    direct_contours = []
    snake_contours = []

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue

        comp_mask = (labels == i).astype(np.uint8)
        seeds_in_comp_raw = [s for s in raw_seeds if comp_mask[int(round(s[1])), int(round(s[0]))] == 1]
        seeds_in_comp = suppress_seeds_in_component(seeds_in_comp_raw)

        if len(seeds_in_comp) <= 1:
            # isolated floe, straight from the binarization, no snake needed
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = contours[0].reshape(-1, 2).astype(np.float64)
                if not touches_edge(c, w, h):
                    direct_contours.append(c)
            continue

        ellipses = separate_merged_object(comp_mask, len(seeds_in_comp))
        if ellipses:
            initial_contours = [make_elliptical_contour(*e) for e in ellipses]
        else:
            initial_contours = [make_circular_contour(sx, sy, max(d, 3.0)) for (sx, sy, d) in seeds_in_comp]

        for c0 in initial_contours:
            c1 = evolve_snake(c0, gvf_u, gvf_v)
            if not touches_edge(c1, w, h) and not is_hallucination(c1):
                snake_contours.append(c1)

    return direct_contours, snake_contours


def contours_to_mask(contours, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for c in contours:
        pts = c.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def valid_pixel_mask(gray, min_brightness=VALID_PIXEL_MIN_BRIGHTNESS):
    """Near black pixels are treated as the blank border warpPerspective
    can leave outside the mapped source region, not open water, and are
    excluded from the sea ice concentration denominator, matching the
    paper excluding masked out regions introduced by orthorectification.
    Deliberately checked against the plain grayscale frame, not the
    enhanced one, since CLAHE and diffusion can shift what counts as near
    black right at the warp border."""
    return (gray >= min_brightness).astype(np.uint8)


# ----------------------------------------------------------------------------
# Main video loop, calibration once, then preprocessing and GVF together
# ----------------------------------------------------------------------------
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
    csv_writer.writerow(["source_frame_index", "num_direct_floes", "num_snake_floes",
                          "total_floes", "sic_percent", "exec_time_s"])

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

        # Preprocessing, using the calibration locked in above, then
        # optional gray level enhancement matching Passerotti et al.
        # (2026), controlled by the toggles at the top of this file
        corrected = perspective_transform(frame, factor, x, y)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

        if ENHANCE_GRAY_ENABLED:
            enhanced = enhance_gray(gray, apply_bilateral=APPLY_BILATERAL, apply_diffusion=APPLY_DIFFUSION,
                                     apply_clahe=APPLY_CLAHE, clahe_clip_limit=CLAHE_CLIP_LIMIT,
                                     clahe_tile_size=CLAHE_TILE_SIZE)
        else:
            enhanced = gray

        # GVF snake segmentation, run on whichever frame was produced above
        direct_contours, snake_contours = segment_frame(enhanced)
        all_contours = direct_contours + snake_contours

        # SIC measured from the final contour mask, matching Passerotti
        # et al. Section 4.1, confirmed with the project supervisor over
        # the binary mask alternative tried briefly, see the note near
        # the top of this file for why
        floe_mask = contours_to_mask(all_contours, gray.shape)
        valid_mask = valid_pixel_mask(gray)
        valid_pixels = max(int(valid_mask.sum()), 1)
        sic = 100.0 * int((floe_mask & valid_mask).sum()) / valid_pixels

        exec_time = time.time() - t0
        csv_writer.writerow([frame_index, len(direct_contours), len(snake_contours),
                              len(all_contours), round(sic, 3), round(exec_time, 4)])

        annotated = corrected.copy()
        for c in direct_contours:
            pts = c.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        for c in snake_contours:
            pts = c.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 0, 255), thickness=1)
        cv2.putText(annotated, "SIC: " + f"{sic:.1f}" + "%  Floes: " + str(len(all_contours)),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(annotated)

        processed_count += 1
        elapsed = time.time() - start_time
        print("Frame " + str(frame_index) + " of " + str(total_frames)
              + ", SIC=" + f"{sic:.2f}" + "%  floes=" + str(len(all_contours))
              + "  frame_time=" + f"{exec_time:.2f}" + "s  elapsed=" + f"{elapsed:.1f}" + "s")

        frame_index += 1

    cap.release()
    writer.release()
    csv_file.close()
    print("Done. " + str(processed_count) + " of " + str(total_frames) + " frames processed (stride "
          + str(FRAME_STRIDE) + ").")
    print("Output video: " + OUTPUT_VIDEO_PATH)
    print("Per frame log: " + OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()