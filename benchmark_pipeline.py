"""
Liss (2023) contouring method benchmark, run against video01.mp4 using
the same shared preprocessing (initial_perspective_transform,
perspective_transform, enhance_gray) used by the GVF, YOLOE, and SAM 2
pipelines, for a fair comparison.

Contouring logic, binary thresholding, morphological opening, external
contour detection, and outlier rejection, is taken from
SEA_ICE_Functions.py in the LSSOSC001_MEC4128S repository. Sea ice
concentration is computed the same way Liss's analysis() function did,
as a pixel area ratio, so no separate real world calibration step is
needed. Ice type categorization, pancake, brash, frazil, is deliberately
left out.

Running this script produces an annotated output video,
out_liss_benchmark.mp4, showing the preprocessed frame with inlier
contours in green, outlier contours in red, and floe count and sea ice
concentration overlaid as text on each frame, the same information
Liss's own create_video() reported, plus a per frame CSV log,
out_liss_benchmark.csv, matching the GVF, YOLOE, and SAM 2 pipelines.
"""

import csv
import cv2
import numpy as np
import statistics
import time
from sea_ice_preprocessing import initial_perspective_transform, preprocess_frame


def nothing(x):
    """Required no-op callback for OpenCV trackbars."""
    pass


def set_threshold_interactive(frame_gray):
    """
    Liss's interactive threshold tuning, adapted to run on an already
    preprocessed grayscale frame from the shared pipeline rather than
    his hardcoded image_trans.jpg. Drag the trackbar until the contour
    overlay looks right, then press Esc to lock in the value.

    :param frame_gray: a single preprocessed grayscale frame, as a numpy array
    :return: the chosen threshold value
    """
    cv2.namedWindow('image')
    cv2.createTrackbar("threshold", "image", 127, 255, nothing)

    kernal = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    display_base = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    threshold = 127

    while True:
        threshold = cv2.getTrackbarPos("threshold", "image")

        ret, thresh = cv2.threshold(frame_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        img = display_base.copy()
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        cv2.putText(img, f"Number of contours = {len(contours)}", (50, 100), font, 0.8, (0, 0, 255))

        cv2.imshow('image', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return threshold


def liss_threshold_contour(imgray, threshold, kernel_size=5):
    """
    Liss's adopted contouring method, binary thresholding, morphological
    opening, external contour detection. Takes an already preprocessed
    grayscale frame from the shared pipeline and a threshold value.
    """
    ret, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY)

    kernal = np.ones((kernel_size, kernel_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours, thresh


def find_outliers(contours, data_area, sensitivity=3):
    """
    Liss's outlier rejection, unchanged in logic. Removes contours whose
    area falls outside mean plus or minus sensitivity standard deviations.
    """
    m = np.mean(data_area)
    st_dev = statistics.stdev(data_area)
    u_bound = m + sensitivity * st_dev
    l_bound = m - sensitivity * st_dev

    inliers_contours, outliers_contours = [], []
    inliers_areas, outliers_areas = [], []

    for i in range(len(data_area)):
        if (data_area[i] > u_bound) or (data_area[i] < l_bound):
            outliers_contours.append(contours[i])
            outliers_areas.append(data_area[i])
        else:
            inliers_contours.append(contours[i])
            inliers_areas.append(data_area[i])

    return tuple(inliers_contours), tuple(outliers_contours), inliers_areas, outliers_areas, len(inliers_contours), len(outliers_contours)


def liss_benchmark(frame_gray, threshold):
    """
    Full Liss contouring benchmark step for a single frame. Returns
    inlier and outlier contours, floe count, and sea ice concentration,
    computed the same way Liss's analysis() function did, as a pixel
    area ratio, so no real world calibration is required. Ice type
    categorization is left out.
    """
    contours, thresh_mask = liss_threshold_contour(frame_gray, threshold)
    raw_areas = [cv2.contourArea(c) for c in contours]
    num_contours = len(contours)

    if num_contours > 1:
        inliers, outliers, inlier_areas, outlier_areas, num_in, num_out = find_outliers(list(contours), raw_areas)
    else:
        inliers, outliers = contours, []
        inlier_areas, outlier_areas = raw_areas, []
        num_in, num_out = num_contours, 0

    frame_area = frame_gray.shape[0] * frame_gray.shape[1]
    denom = frame_area - sum(outlier_areas)
    SIC = round(100 * (sum(raw_areas) - sum(outlier_areas)) / denom, 3) if denom > 0 else 0.0

    return {
        "contours": inliers,
        "outlier_contours": outliers,
        "areas_px": inlier_areas,
        "num_inliers": num_in,
        "num_outliers": num_out,
        "SIC": SIC,
        "thresh_mask": thresh_mask,
    }


def annotate_frame(frame_gray, result):
    """
    Draws inlier contours in green, outlier contours in red, and floe
    count and sea ice concentration as text, onto a color copy of the
    preprocessed frame, for writing to the output video.
    """
    img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.drawContours(img, result["contours"], -1, (0, 255, 0), 1)
    cv2.drawContours(img, result["outlier_contours"], -1, (0, 0, 255), 1)

    cv2.putText(img, f"Floe count = {result['num_inliers']}", (10, 30), font, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"SIC = {result['SIC']}%", (10, 60), font, 0.7, (0, 0, 255), 2)

    return img


def write_results_csv(frame_results, output_csv):
    """
    Writes the per frame benchmark log to CSV, one row per frame, using
    the same column names as the SAM 2 pipeline's log,
    source_frame_index, num_floes, sic_percent, exec_time_s, so all
    four benchmark logs line up for the comparison stage.
    """
    fieldnames = ["source_frame_index", "num_floes", "sic_percent", "exec_time_s"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_results:
            writer.writerow(row)

    print(f"Saved {len(frame_results)} rows to {output_csv}")


def main():
    videofile = "video01.mp4"
    output_video = "out_liss_benchmark.mp4"
    output_csv = "out_liss_benchmark.csv"

    cap = cv2.VideoCapture(videofile)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    # initial_perspective_transform needs a file path, not a frame array,
    # so save the first frame out before calling it. Same calibration
    # step used by the other three pipelines.
    cv2.imwrite("first_frame.jpg", first_frame)
    _, factor, x, y, _, _ = initial_perspective_transform("first_frame.jpg")

    # Preprocess the first frame with the locked in calibration, then
    # tune the threshold against it
    first_gray = preprocess_frame(first_frame, factor, x, y)
    threshold = set_threshold_interactive(first_gray)

    out_height, out_width = first_gray.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = 5  # lower this to slow playback down further, raise it to speed up
    writer = cv2.VideoWriter(output_video, fourcc, output_fps, (out_width, out_height), True)

    frame_results = []
    frame_no = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        frame_gray = preprocess_frame(frame, factor, x, y)
        result = liss_benchmark(frame_gray, threshold)
        exec_time_s = round(time.perf_counter() - start_time, 4)

        annotated = annotate_frame(frame_gray, result)
        writer.write(annotated)

        frame_results.append({
            "source_frame_index": frame_no,
            "num_floes": result["num_inliers"],
            "sic_percent": result["SIC"],
            "exec_time_s": exec_time_s,
        })

        frame_no += 1

    cap.release()
    writer.release()

    print(f"Saved {frame_no} frames to {output_video}")

    write_results_csv(frame_results, output_csv)

    return frame_results


if __name__ == "__main__":
    results = main()