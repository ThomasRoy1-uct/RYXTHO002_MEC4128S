#!/usr/bin/env python3
"""
Sea ice video preprocessing pipeline.

Implements the preprocessing steps described in Passerotti et al. (2026, Earth
and Space Science, "Sea Ice Floe Segmentation in Close Range Optical Imagery
Using Active Contour"), adapted from single still images to a video stream.

Steps applied to every frame, in order:

  1. Perspective correction, taking the oblique shipborne view to a top down
     view. The paper achieves this through full orthorectification, using a
     camera projection matrix built from intrinsic parameters (focal length,
     lens distortion, from a chessboard calibration) and extrinsic parameters
     (sensor height, tilt, roll, from the ship mounting geometry). That
     calibration data is not yet available for this camera, so this script
     uses a four point homography as a practical proxy, mapping a manually
     selected trapezoid on the sea surface to a rectangle. Once real
     calibration parameters exist for the Agulhas II camera, the homography
     step can be swapped for a proper projection matrix without touching the
     rest of the pipeline.

  2. Region of interest crop, keeping only the near field portion of the
     rectified frame. The paper notes that floe edges lose sharpness with
     distance from the sensor after rectification, and restricts analysis to
     the near field for this reason. The kept fraction is configurable.

  3. Image enhancement, combining an edge preserving bilateral filter, an
     anisotropic (Perona and Malik, 1990) diffusion filter, and contrast
     limited adaptive histogram equalisation (CLAHE). The paper describes
     this step explicitly as gray level homogenization, run on a single
     channel grayscale image, not per colour channel. This script follows
     that directly, converting to grayscale before the three enhancement
     steps. The output frame is still written as a three channel image
     (the grayscale result duplicated across B, G, and R), so the video
     file stays a normal playable video and stays shape compatible with
     RGB shaped model input, but there is no colour information left in
     it, matching the paper's single enhanced grayscale image that all of
     GVF, SAM, and SAM 2 are run on downstream.

Night frame handling (dark channel prior low light recovery) has been
removed, the footage this runs on doesn't include night imagery.

Model inference, GVF snake, YOLOE, and SAM 2, is intentionally out of scope
here. This script only produces a preprocessed video, ready to be fed into
whichever segmentation model is being benchmarked next.

Usage
-----
    python sea_ice_video_preprocessing.py

By default this looks for a file called video01.mp4 in the same folder as
the script. To use a different file or path instead, pass it explicitly

    python sea_ice_video_preprocessing.py --input some_other_clip.mp4

The first time a given input video is processed, an interactive window opens
on the first frame asking for four clicks, marking the corners of the sea
surface region to rectify, in order top left, top right, bottom right,
bottom left. Those points are cached in a json file next to the video, so
later runs on the same camera position are fully automatic and need no
further input. Pass --calibrate to redo this step, for example after the
camera has been moved.

A quick way to check the calibration and enhancement settings before
committing to processing a whole clip is:

    python sea_ice_video_preprocessing.py --input clip.mp4 --preview-only

which writes a single processed frame as a png next to the output path.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Configuration defaults, all overridable from the command line
# ----------------------------------------------------------------------------

VIDEO_PATH = "video01.mp4"       # <-- change this to point at a different video, no need to touch anything else

DEFAULT_DST_WIDTH = 800          # width in pixels of the top down output canvas
DEFAULT_DST_HEIGHT = 1200        # fallback only, used if the aspect ratio estimate is degenerate, see estimate_aspect_ratio
DEFAULT_ROI_KEEP_RATIO = 0.65    # fraction of the rectified frame height kept, nearest to the camera
MIN_OUTPUT_SHORT_SIDE_PX = 600   # the shorter of the two final output dimensions is scaled up to at least this
MAX_OUTPUT_LONG_SIDE_PX = 2500   # safety ceiling, stops an extreme aspect ratio demanding a huge canvas
DEFAULT_CLAHE_CLIP_LIMIT = 2.5
DEFAULT_CLAHE_TILE_GRID = (8, 8)
DEFAULT_DIFFUSION_ITER = 8
DEFAULT_DIFFUSION_KAPPA = 25.0
DEFAULT_DIFFUSION_GAMMA = 0.15
DEFAULT_BILATERAL_D = 9
DEFAULT_BILATERAL_SIGMA_COLOR = 60
DEFAULT_BILATERAL_SIGMA_SPACE = 60


# ----------------------------------------------------------------------------
# Perspective calibration, cached per video since the camera mount is fixed
# ----------------------------------------------------------------------------

def calibration_path_for(video_path):
    root, _ = os.path.splitext(video_path)
    return root + "_perspective_calibration.json"


def load_calibration(video_path):
    path = calibration_path_for(video_path)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return np.array(data["src_points"], dtype=np.float32)
    return None


def save_calibration(video_path, src_points):
    path = calibration_path_for(video_path)
    with open(path, "w") as f:
        json.dump({"src_points": src_points.tolist()}, f, indent=2)
    print("Saved perspective calibration to " + path)


def interactive_calibration(frame):
    """
    Opens a window on the given frame and lets the user click four points,
    in order top left, top right, bottom right, bottom left, marking the
    sea surface region that should be mapped to a top down rectangle.

    The shape isn't constrained to any particular form, a true rectangle
    of sea surface only projects as a symmetric trapezoid if the camera
    has zero roll, and there's no IMU data here to confirm that, so
    forcing that shape could make a real calibration worse rather than
    better. Instead, a line is drawn connecting each point to the last as
    they're placed, closing back to the first point on the fourth click,
    so the shape is visible while clicking, a badly placed point is
    obvious immediately rather than only after the fact.
    """
    points = []
    display = frame.copy()
    window_name = "Click 4 points, top left, top right, bottom right, bottom left, then press any key"

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], (0, 255, 255), 2)
            if len(points) == 4:
                cv2.line(display, points[-1], points[0], (0, 255, 255), 2)
            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)

    print("Click the four corners of the region to rectify, in order, top left, "
          "top right, bottom right, bottom left. Press any key once all four are placed.")

    while len(points) < 4:
        cv2.waitKey(50)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    if len(points) != 4:
        raise RuntimeError("Calibration needs exactly four points, got " + str(len(points)))

    return np.array(points, dtype=np.float32)


def save_calibration_check_image(video_path, frame, src_points):
    """
    Draws the four calibration points and the quadrilateral connecting them
    onto the raw frame, and saves it next to the video. This is the fastest
    way to check whether the selected region actually covers a sensible
    area of sea ice, rather than a small or misplaced patch, before running
    the perspective transform on a whole clip.
    """
    labelled = frame.copy()
    labels = ["TL", "TR", "BR", "BL"]
    pts = src_points.astype(int)

    cv2.polylines(labelled, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    for (x, y), label in zip(pts, labels):
        cv2.circle(labelled, (x, y), 10, (0, 255, 0), -1)
        cv2.putText(labelled, label, (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                     1.2, (0, 255, 0), 3)

    root, _ = os.path.splitext(video_path)
    check_path = root + "_calibration_check.png"
    cv2.imwrite(check_path, labelled)
    print("Saved calibration check image to " + check_path)
    print("Open it and confirm the red box covers the sea ice area you want rectified, "
          "not a small patch and not the ship structure, before processing the full video.")


def ensure_minimum_output_size(dst_width, dst_height, roi_keep_ratio,
                                min_short_side=MIN_OUTPUT_SHORT_SIDE_PX,
                                max_long_side=MAX_OUTPUT_LONG_SIDE_PX):
    """
    Scales the whole rectified canvas up, keeping its proportions exactly
    as they are, so that the shorter of the two FINAL output dimensions
    (width, and height after the near field crop) meets min_short_side.

    This matters because a wide, shallow calibration box, typical for a
    ship mounted camera, otherwise produces a short output height once
    the crop is applied on top of an already short warped canvas, small
    floes can end up just a handful of pixels across, too coarse for the
    circularity and eccentricity checks the GVF script's hallucination
    filter relies on. Scaling the canvas up rather than changing its
    shape keeps floes correctly proportioned, it just gives them more
    pixels to be measured with.

    max_long_side is a safety ceiling, an extremely shallow calibration
    box could otherwise demand a very wide canvas to hit the same short
    side floor, this stops that from silently costing far more processing
    time than intended.
    """
    output_width = dst_width
    output_height = dst_height * roi_keep_ratio
    short_side = min(output_width, output_height)

    if short_side >= min_short_side:
        return dst_width, dst_height

    scale = min_short_side / short_side
    long_side = max(output_width, output_height)
    if long_side * scale > max_long_side:
        scale = max_long_side / long_side

    return int(round(dst_width * scale)), int(round(dst_height * scale))


def estimate_aspect_ratio(src_points):
    """
    Estimates how tall the rectified output should be relative to its
    width, from the actual shape of the four clicked points, rather than
    always forcing a fixed rectangle regardless of what was selected.

    Without true camera calibration there is still no way to know the
    real-world metres each pixel represents, so this cannot make the
    output metrically accurate. What it does fix is shape distortion,
    previously every calibration, wide trapezoid or narrow one, got
    warped into the same fixed width by height rectangle, which stretches
    or squashes floes unevenly depending on how far that fixed shape was
    from the true proportions of the clicked area. Matching the output
    aspect ratio to the clicked quadrilateral's own proportions avoids
    that.
    """
    tl, tr, br, bl = src_points
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)

    avg_width = (top_width + bottom_width) / 2.0
    avg_height = (left_height + right_height) / 2.0

    if avg_width <= 0:
        return DEFAULT_DST_HEIGHT / DEFAULT_DST_WIDTH
    return avg_height / avg_width


def get_perspective_matrix(src_points, dst_width, dst_height):
    dst_points = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_points, dst_points)


# ----------------------------------------------------------------------------
# Anisotropic diffusion, Perona and Malik (1990)
# ----------------------------------------------------------------------------

def anisotropic_diffusion(gray, num_iter=DEFAULT_DIFFUSION_ITER,
                           kappa=DEFAULT_DIFFUSION_KAPPA, gamma=DEFAULT_DIFFUSION_GAMMA):
    """
    Edge preserving anisotropic diffusion applied to a single channel image.
    Smooths regions with low gradients while preserving strong edges such as
    floe boundaries, matching the description in Section 3.3 of the paper.
    """
    img = gray.astype(np.float32)
    for _ in range(num_iter):
        delta_n = np.roll(img, -1, axis=0) - img
        delta_s = np.roll(img, 1, axis=0) - img
        delta_e = np.roll(img, -1, axis=1) - img
        delta_w = np.roll(img, 1, axis=1) - img

        c_n = np.exp(-(delta_n / kappa) ** 2)
        c_s = np.exp(-(delta_s / kappa) ** 2)
        c_e = np.exp(-(delta_e / kappa) ** 2)
        c_w = np.exp(-(delta_w / kappa) ** 2)

        img += gamma * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)

    return np.clip(img, 0, 255).astype(np.uint8)


def enhance_frame(bgr_frame, clahe):
    """
    Gray level homogenization, exactly the phrase the paper uses for this
    step, combining an edge preserving bilateral filter, anisotropic
    diffusion, and CLAHE, run on a single channel grayscale version of the
    frame rather than on a colour channel. This deliberately discards hue
    and saturation, following the paper's method rather than trying to
    preserve colour through it. The grayscale result is duplicated across
    all three channels before returning, purely so the output stays a
    normal three channel video frame, the actual content has no colour
    information in it.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(
        gray, DEFAULT_BILATERAL_D, DEFAULT_BILATERAL_SIGMA_COLOR, DEFAULT_BILATERAL_SIGMA_SPACE
    )
    diffused = anisotropic_diffusion(filtered)
    enhanced = clahe.apply(diffused)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------------
# Full per frame pipeline
# ----------------------------------------------------------------------------

def process_frame(frame, perspective_matrix, dst_size, roi_keep_ratio, clahe):
    warped = cv2.warpPerspective(frame, perspective_matrix, dst_size)

    dst_width, dst_height = dst_size
    keep_height = int(dst_height * roi_keep_ratio)
    roi = warped[dst_height - keep_height:dst_height, :]

    return enhance_frame(roi, clahe)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sea ice video preprocessing pipeline")
    parser.add_argument("--input", default=VIDEO_PATH, help="path to the input video, defaults to VIDEO_PATH above")
    parser.add_argument("--output", default=None,
                         help="path to the output video, defaults to the input name with a _preprocessed suffix")
    parser.add_argument("--calibrate", action="store_true",
                         help="force a new interactive perspective calibration, even if a cached one exists")
    parser.add_argument("--dst-width", type=int, default=DEFAULT_DST_WIDTH)
    parser.add_argument("--dst-height", type=int, default=None,
                         help="height of the rectified canvas, defaults to matching the aspect ratio "
                              "of the four clicked calibration points, pass this to override that")
    parser.add_argument("--roi-keep-ratio", type=float, default=DEFAULT_ROI_KEEP_RATIO)
    parser.add_argument("--preview-only", action="store_true",
                         help="process and save only the first frame, useful for checking calibration quickly")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit("Input video not found, " + args.input)

    output_path = args.output
    if output_path is None:
        root, _ = os.path.splitext(args.input)
        output_path = root + "_preprocessed.mp4"

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit("Could not open video, " + args.input)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok:
        sys.exit("Could not read the first frame of the video")

    src_points = None if args.calibrate else load_calibration(args.input)
    if src_points is None:
        src_points = interactive_calibration(first_frame)
        save_calibration(args.input, src_points)

    save_calibration_check_image(args.input, first_frame, src_points)

    if args.dst_height is None:
        aspect_ratio = estimate_aspect_ratio(src_points)
        dst_width, dst_height = args.dst_width, int(round(args.dst_width * aspect_ratio))
        dst_width, dst_height = ensure_minimum_output_size(dst_width, dst_height, args.roi_keep_ratio)
        print("No --dst-height given, matching the clicked region's own proportions, "
              "aspect ratio " + f"{aspect_ratio:.2f}" + ", using dst width " + str(dst_width)
              + " and dst height " + str(dst_height) + " so the shorter output side reaches at least "
              + str(MIN_OUTPUT_SHORT_SIDE_PX) + "px.")
    else:
        dst_width, dst_height = args.dst_width, args.dst_height

    dst_size = (dst_width, dst_height)
    perspective_matrix = get_perspective_matrix(src_points, dst_width, dst_height)

    clahe = cv2.createCLAHE(clipLimit=DEFAULT_CLAHE_CLIP_LIMIT, tileGridSize=DEFAULT_CLAHE_TILE_GRID)

    keep_height = int(dst_height * args.roi_keep_ratio)
    out_size = (dst_width, keep_height)

    if args.preview_only:
        processed = process_frame(first_frame, perspective_matrix, dst_size, args.roi_keep_ratio, clahe)
        preview_path = os.path.splitext(output_path)[0] + "_preview.png"
        cv2.imwrite(preview_path, processed)
        print("Saved preview frame to " + preview_path)
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, out_size)
    if not writer.isOpened():
        cap.release()
        sys.exit("Could not open the output video for writing, " + output_path)

    # rewind so the first frame, already read above for calibration, is not skipped
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_index = 0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        processed = process_frame(frame, perspective_matrix, dst_size, args.roi_keep_ratio, clahe)
        writer.write(processed)

        frame_index += 1
        if frame_index % 25 == 0 or frame_index == total_frames:
            elapsed = time.time() - start_time
            print("Processed frame " + str(frame_index) + " of " + str(total_frames)
                  + ", elapsed " + f"{elapsed:.1f}" + " s")

    cap.release()
    writer.release()
    print("Done. Preprocessed video written to " + output_path)


if __name__ == "__main__":
    main()