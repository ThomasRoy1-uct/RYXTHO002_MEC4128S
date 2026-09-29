"""
compare_methods.py

Reads the per frame CSV logs produced by gvf_pipeline.py, yoloe_pipeline.py,
sam2_pipeline.py and benchmark_pipeline.py, aligns them by source frame
index with the Liss (2023) benchmark as the reference, and produces the
comparison plots for the output metrics and analysis framework section and
the Results chapter.

Styled after Passerotti et al. (2026) Figure 11, relative error and
distribution style comparison against a reference, substituting the Liss
baseline for their manual benchmark, since no manually segmented ground
truth exists for this footage. Floe size is not included, since it would
need a pixel to area calibration that is not available for this project.

Expects the four log CSVs in the same folder as this script, using the
filenames each pipeline already writes.

Produces, in a figures subfolder:
  fig_sic_per_frame.png              SIC vs frame, all four methods overlaid
  fig_floe_count_per_frame.png       floe count vs frame, all four methods overlaid
  fig_sic_<method>_vs_liss.png       SIC vs frame, one method against Liss on its own, one file per method
  fig_floe_count_<method>_vs_liss.png  floe count vs frame, one method against Liss on its own, one file per method
  fig_relative_error_count.png       mean relative error in floe count vs Liss, per method
  fig_relative_error_sic.png         mean relative error in SIC vs Liss, per method
  fig_time_per_floe.png              computation time per floe, per method

Run with:  python compare_methods.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- configuration -------------------------------------------------------

LOG_PATHS = {
    "GVF": "gvf_snake_sic_log.csv",
    "YOLOE": "yoloe_sic_log.csv",
    "SAM2": "sam2_sic_log.csv",
    "Liss": "out_liss_benchmark.csv",
}

BASELINE = "Liss"   # the reference every relative error is measured against

# one colour per method, kept identical across every plot in this script.
# Okabe and Ito's colourblind safe palette, chosen for maximum separation
# between all four lines when they overlap heavily on the same axes
COLORS = {
    "GVF": "#0072B2",     # blue
    "YOLOE": "#D55E00",   # vermillion
    "SAM2": "#009E73",    # bluish green
    "Liss": "#000000",    # black, the baseline
}

OUT_DIR = "figures"


# ---- loading ---------------------------------------------------------------

def load_log(method, path):
    """
    Loads one pipeline's per frame CSV log and harmonises it to a common
    set of columns, source_frame_index, floe_count, sic_percent, exec_time_s.

    The GVF log is the only one that splits floe count into
    num_direct_floes and num_snake_floes, with total_floes as their sum, so
    that is what gets used here. The other three logs already write a
    single num_floes column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find the {method} log at '{path}'. Run that "
            f"pipeline first, or check LOG_PATHS at the top of this script."
        )
    df = pd.read_csv(path)

    if "total_floes" in df.columns:
        df = df.rename(columns={"total_floes": "floe_count"})
    elif "num_floes" in df.columns:
        df = df.rename(columns={"num_floes": "floe_count"})
    else:
        raise ValueError(f"{path} has neither total_floes nor num_floes, columns found: {list(df.columns)}")

    missing = {"source_frame_index", "floe_count", "sic_percent", "exec_time_s"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns {missing}")

    df = df[["source_frame_index", "floe_count", "sic_percent", "exec_time_s"]].copy()
    df["method"] = method
    return df.sort_values("source_frame_index").reset_index(drop=True)


def load_all():
    frames = {m: load_log(m, p) for m, p in LOG_PATHS.items()}

    # a quick, honest check that the four runs are actually comparable,
    # since a relative error only means something if every method was run
    # at the same frame stride on the same video
    strides = {m: df["source_frame_index"].diff().dropna().mode().iat[0] for m, df in frames.items()}
    if len(set(strides.values())) > 1:
        print("Warning, the four logs do not share the same frame stride, "
              f"{strides}. Relative error and the per frame plots will only "
              "compare the frame indices the runs actually have in common.")
    return frames


# ---- plot 1 and 2, raw series per frame, all methods together --------------

def plot_series_per_frame(frames, value_col, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for m, df in frames.items():
        ax.plot(df["source_frame_index"], df[value_col], label=m,
                 color=COLORS[m], linewidth=1.0)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # legend sits below the axes, out of the way of the lines themselves,
    # rather than floating inside the plot where it can sit on top of data
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# ---- new, each method against Liss on its own -------------------------------

def plot_method_vs_baseline(frames, method, value_col, ylabel, title, out_path, baseline=BASELINE):
    """The same per frame series as plot_series_per_frame, but just one
    method against the baseline, so the two lines are easy to follow
    without the other methods crossing through them."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name in (baseline, method):
        df = frames[name]
        ax.plot(df["source_frame_index"], df[value_col], label=name,
                 color=COLORS[name], linewidth=1.2)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# ---- plots 3 and 4, relative error against the Liss baseline ---------------

def merged_wide(frames, value_col):
    """One row per frame index that every method actually shares, one
    column per method, an inner join so a relative error is never computed
    against a frame some method skipped."""
    wide = None
    for m, df in frames.items():
        s = df.set_index("source_frame_index")[value_col].rename(m)
        wide = s.to_frame() if wide is None else wide.join(s, how="inner")
    return wide.sort_index()


def relative_error(wide, baseline=BASELINE):
    """Per frame percent relative error of each method against the
    baseline column. Undefined, left as NaN, on a frame where the baseline
    itself is zero, since a percent error against zero has no meaning."""
    base = wide[baseline]
    out = pd.DataFrame(index=wide.index)
    for col in wide.columns:
        if col == baseline:
            continue
        out[col] = np.where(base != 0, 100.0 * (wide[col] - base) / base, np.nan)
    return out


def plot_relative_error_bar(re_df, ylabel, title, out_path):
    """Mirrors Passerotti et al. (2026) Figure 11a and 11g, a relative
    error per method against a reference line at zero, only their
    reference is the manual benchmark and this one is the Liss baseline."""
    methods = list(re_df.columns)
    means = [re_df[m].mean(skipna=True) for m in methods]
    stds = [re_df[m].std(skipna=True) for m in methods]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[m] for m in methods])
    ax.axhline(0, color=COLORS[BASELINE], linewidth=1.0, linestyle="--",
                label=f"{BASELINE} baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# ---- plot 5, computation time per floe --------------------------------------

def plot_time_per_floe(frames, out_path):
    """Time per floe rather than time per frame, following Passerotti et
    al. (2026), so a densely packed frame is not read as a slower method
    just because it had more floes to contour. Computed as total execution
    time over total floe count across the whole run, rather than averaging
    a per frame ratio, so frames with zero floes do not need to be
    dropped or cause a division by zero."""
    methods, values_ms = [], []
    for m, df in frames.items():
        total_time = df["exec_time_s"].sum()
        total_floes = df["floe_count"].sum()
        per_floe = (total_time / total_floes) if total_floes > 0 else np.nan
        methods.append(m)
        values_ms.append(per_floe * 1000.0)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(methods))
    ax.bar(x, values_ms, color=[COLORS[m] for m in methods])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Computation time per floe [ms]")
    ax.set_title("Computational cost per floe, by method")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# ---- entry point -------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = load_all()

    plot_series_per_frame(
        frames, "sic_percent", "SIC [%]",
        "Sea ice concentration per frame, all methods",
        os.path.join(OUT_DIR, "fig_sic_per_frame.png"),
    )
    plot_series_per_frame(
        frames, "floe_count", "Floe count",
        "Floe count per frame, all methods",
        os.path.join(OUT_DIR, "fig_floe_count_per_frame.png"),
    )

    # each method against Liss on its own, easier to read than four lines
    # crossing each other, one pair of figures per method
    for m in frames:
        if m == BASELINE:
            continue
        tag = m.lower()
        plot_method_vs_baseline(
            frames, m, "sic_percent", "SIC [%]",
            f"SIC per frame, {m} against {BASELINE}",
            os.path.join(OUT_DIR, f"fig_sic_{tag}_vs_liss.png"),
        )
        plot_method_vs_baseline(
            frames, m, "floe_count", "Floe count",
            f"Floe count per frame, {m} against {BASELINE}",
            os.path.join(OUT_DIR, f"fig_floe_count_{tag}_vs_liss.png"),
        )

    wide_count = merged_wide(frames, "floe_count")
    wide_sic = merged_wide(frames, "sic_percent")

    re_count = relative_error(wide_count)
    re_sic = relative_error(wide_sic)

    plot_relative_error_bar(
        re_count, "Relative error in floe count [%]",
        f"Floe count relative to the {BASELINE} baseline",
        os.path.join(OUT_DIR, "fig_relative_error_count.png"),
    )
    plot_relative_error_bar(
        re_sic, "Relative error in SIC [%]",
        f"SIC relative to the {BASELINE} baseline",
        os.path.join(OUT_DIR, "fig_relative_error_sic.png"),
    )

    plot_time_per_floe(frames, os.path.join(OUT_DIR, "fig_time_per_floe.png"))

    print(f"\nDone. Eleven figures saved to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()