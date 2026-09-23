"""
Comparison plots across all four pipeline logs, gvf_snake_sic_log.csv,
sam2_sic_log.csv, yoloe_sic_log.csv, and out_liss_benchmark.csv, matched
by source_frame_index. Built to sit alongside the other scripts in this
project rather than as a one off notebook, so it can be rerun any time a
fresh set of logs comes out of a pipeline run.

Needs pandas and matplotlib, neither used by the pipeline scripts
themselves, install first if not already present.

    pip install pandas matplotlib

Produces one PNG with sixteen panels. Top row, SIC over the video for
all four methods together, floe count over the video the same way, then
a bar chart of each method's average floe count for a quick summary.
Next row, a bar chart of each method's average SIC, the other two slots
in that row left empty. Next two rows, pairwise scatter plots of SIC,
one panel per pair of methods, six pairs across four methods. Final two
rows, the same six pairs for floe count. Every scatter panel includes a
dashed diagonal line marking where two methods would agree exactly, and
the correlation coefficient printed in the corner, so agreement can be
read directly rather than only inferred from how tightly the points
cluster.

Note, the Liss benchmark currently logs every frame rather than the
step of five the other three pipelines use. The merge below is an inner
join on source_frame_index, so it already narrows everything down to
the frame indices the other three share, and Liss's extra rows are
dropped automatically, no separate handling needed. If Liss's script is
later changed to log on the same step of five, nothing here needs to
change.

Usage

    python compare_pipelines.py

Reads the four CSV filenames below from the current folder, saves
comparison_plots.png.
"""

import itertools
import pandas as pd
import matplotlib.pyplot as plt

GVF_CSV = "gvf_snake_sic_log.csv"
SAM2_CSV = "sam2_sic_log.csv"
YOLOE_CSV = "yoloe_sic_log.csv"
LISS_CSV = "out_liss_benchmark.csv"
OUTPUT_PATH = "comparison_plots.png"

COLORS = {"GVF": "#4C72B0", "SAM 2": "#C44E52", "YOLOE": "#90EE90", "Liss": "#DD8452"}

# Each method's floe count and SIC column names, once loaded into the merged dataframe
METHOD_COLUMNS = {
    "GVF": ("gvf_floes", "gvf_sic"),
    "SAM 2": ("sam2_floes", "sam2_sic"),
    "YOLOE": ("yoloe_floes", "yoloe_sic"),
    "Liss": ("liss_floes", "liss_sic"),
}


def load_and_merge():
    gvf = pd.read_csv(GVF_CSV).rename(columns={"total_floes": "gvf_floes", "sic_percent": "gvf_sic"})
    sam2 = pd.read_csv(SAM2_CSV).rename(columns={"num_floes": "sam2_floes", "sic_percent": "sam2_sic"})
    yoloe = pd.read_csv(YOLOE_CSV).rename(columns={"num_floes": "yoloe_floes", "sic_percent": "yoloe_sic"})
    liss = pd.read_csv(LISS_CSV).rename(columns={"num_floes": "liss_floes", "sic_percent": "liss_sic"})

    df = gvf[["source_frame_index", "gvf_floes", "gvf_sic"]]
    df = df.merge(sam2[["source_frame_index", "sam2_floes", "sam2_sic"]], on="source_frame_index")
    df = df.merge(yoloe[["source_frame_index", "yoloe_floes", "yoloe_sic"]], on="source_frame_index")
    df = df.merge(liss[["source_frame_index", "liss_floes", "liss_sic"]], on="source_frame_index")
    return df


def plot_time_series(ax, df, metric, title, ylabel):
    """metric is either "sic" or "floes", selects which column of each method to plot."""
    for method, (floes_col, sic_col) in METHOD_COLUMNS.items():
        col = sic_col if metric == "sic" else floes_col
        ax.plot(df["source_frame_index"], df[col], label=method, color=COLORS[method], linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("source_frame_index")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_pair_scatter(ax, df, col_a, col_b, label_a, label_b, title):
    ax.scatter(df[col_a], df[col_b], s=10, alpha=0.6, color="#55A868")
    lo = min(df[col_a].min(), df[col_b].min())
    hi = max(df[col_a].max(), df[col_b].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    corr = df[col_a].corr(df[col_b])
    ax.text(0.05, 0.92, "r = " + f"{corr:.2f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title(title)


def plot_average_floes_bar(ax, df):
    methods = list(METHOD_COLUMNS.keys())
    means = [df[METHOD_COLUMNS[m][0]].mean() for m in methods]
    ax.bar(methods, means, color=[COLORS[m] for m in methods])
    ax.set_title("Average floe count per frame")
    ax.set_ylabel("Floe count")


def plot_average_sic_bar(ax, df):
    methods = list(METHOD_COLUMNS.keys())
    means = [df[METHOD_COLUMNS[m][1]].mean() for m in methods]
    ax.bar(methods, means, color=[COLORS[m] for m in methods])
    ax.set_title("Average SIC per frame")
    ax.set_ylabel("SIC percent")


def main():
    df = load_and_merge()

    methods = list(METHOD_COLUMNS.keys())
    pairs = list(itertools.combinations(methods, 2))  # six pairs across four methods

    fig, axes = plt.subplots(6, 3, figsize=(16, 26))

    plot_time_series(axes[0, 0], df, "sic", "SIC percent across the video", "SIC percent")
    plot_time_series(axes[0, 1], df, "floes", "Floe count across the video", "Floe count")
    plot_average_floes_bar(axes[0, 2], df)

    plot_average_sic_bar(axes[1, 0], df)
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    sic_axes = axes[2:4].flatten()
    for ax, (method_a, method_b) in zip(sic_axes, pairs):
        col_a, col_b = METHOD_COLUMNS[method_a][1], METHOD_COLUMNS[method_b][1]
        plot_pair_scatter(ax, df, col_a, col_b, f"{method_a} SIC", f"{method_b} SIC",
                           f"SIC, {method_a} vs {method_b}")

    floe_axes = axes[4:6].flatten()
    for ax, (method_a, method_b) in zip(floe_axes, pairs):
        col_a, col_b = METHOD_COLUMNS[method_a][0], METHOD_COLUMNS[method_b][0]
        plot_pair_scatter(ax, df, col_a, col_b, f"{method_a} floes", f"{method_b} floes",
                           f"Floe count, {method_a} vs {method_b}")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print("Saved " + OUTPUT_PATH)


if __name__ == "__main__":
    main()