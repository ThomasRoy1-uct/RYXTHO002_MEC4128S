"""
Threshold sweep for Liss's contouring benchmark, run against video01.mp4
using the same shared preprocessing as benchmark_pipeline.py. Tests a
range of threshold values non-interactively, one full pass over the
video per threshold, and reports mean floe count, mean SIC, and the
correlation between floe count and SIC for each. Used to find where
the outlier driven inflation, seen earlier as a strong negative
correlation between floe count and SIC, disappears, which is treated
as the point where segmentation is properly resolving floes rather
than merging them into outlier blobs.



Reads video01.mp4 from the current folder. Reuses liss_benchmark from
benchmark_pipeline.py, so that file must be in the same folder.
"""

import cv2
import pandas as pd

from benchmark_pipeline import liss_benchmark
from sea_ice_preprocessing import initial_perspective_transform, preprocess_frame

VIDEO_FILE = "video01.mp4"
SUMMARY_CSV = "threshold_sweep_summary.csv"
PER_FRAME_CSV = "threshold_sweep_per_frame.csv"

# Narrowed sweep
THRESHOLDS_TO_TEST = list(range(140, 176, 3))


def run_single_threshold(cap, factor, x, y, threshold):
    """Runs the full video once at a single threshold, returns per frame floe count and SIC lists."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    floes, sics = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = preprocess_frame(frame, factor, x, y)
        result = liss_benchmark(frame_gray, threshold)
        floes.append(result["num_inliers"])
        sics.append(result["SIC"])

    return floes, sics


def main():
    cap = cv2.VideoCapture(VIDEO_FILE)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    # Same calibration step used by benchmark_pipeline.py.
    cv2.imwrite("first_frame.jpg", first_frame)
    _, factor, x, y, _, _ = initial_perspective_transform("first_frame.jpg")

    summary_rows = []
    per_frame_rows = []

    for threshold in THRESHOLDS_TO_TEST:
        floes, sics = run_single_threshold(cap, factor, x, y, threshold)
        df = pd.DataFrame({"num_floes": floes, "sic_percent": sics})
        corr = df["num_floes"].corr(df["sic_percent"])

        row = {
            "threshold": threshold,
            "mean_floes": round(df["num_floes"].mean(), 2),
            "mean_sic": round(df["sic_percent"].mean(), 2),
            "floes_sic_corr": round(corr, 3),
        }
        summary_rows.append(row)

        print(f"threshold={row['threshold']}  mean_floes={row['mean_floes']}  "
              f"mean_sic={row['mean_sic']}  corr={row['floes_sic_corr']}")

        for source_frame_index, (num_floes, sic_percent) in enumerate(zip(floes, sics)):
            per_frame_rows.append({
                "threshold": threshold,
                "source_frame_index": source_frame_index,
                "num_floes": num_floes,
                "sic_percent": sic_percent,
            })

    cap.release()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {SUMMARY_CSV}")

    per_frame_df = pd.DataFrame(per_frame_rows)
    per_frame_df.to_csv(PER_FRAME_CSV, index=False)
    print(f"Saved {PER_FRAME_CSV}")


if __name__ == "__main__":
    main()