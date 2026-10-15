"""
compare_methods.py

Reads the per frame CSV logs produced by gvf_pipeline.py, yoloe_pipeline.py,
sam2_pipeline.py and benchmark_pipeline.py, aligns them by source frame
index with the Liss (2023) benchmark as the reference, and produces the
comparison plots for the output metrics and analysis framework section and
the Results chapter.

Expects the four log CSVs in the same folder as this script, using the
filenames each pipeline already writes.

Produces, in a figures subfolder:
  fig_sic_per_frame.png              SIC vs frame, all four methods overlaid
  fig_floe_count_per_frame.png       floe count vs frame, all four methods overlaid
  fig_sic_<method>_vs_liss.png       SIC vs frame, one method against Liss on its own, one file per method
  fig_floe_count_<method>_vs_liss.png  floe count vs frame, one method against Liss on its own, one file per method
  fig_own_<method>.png               SIC and floe count vs frame, one method on its own, with its mean marked, one file per method
  fig_relative_error_count.png       mean relative error in floe count vs Liss, per method
  fig_relative_error_sic.png         mean relative error in SIC vs Liss, per method
  fig_box_sic.png                    box and whisker summary of SIC, all four methods
  fig_box_floe_count.png             box and whisker summary of floe count, all four methods
  fig_time_per_floe.png              computation time per floe, per method
  fig_time_per_frame_and_floe.png    computation time per frame and per floe, side by side, per method

Run with python compare_methods.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# configuration 

LOG_PATHS = {
    "GVF": "gvf_snake_sic_log.csv",
    "YOLOE": "yoloe_sic_log.csv",
    "SAM2": "sam2_sic_log.csv",
    "Liss": "out_liss_benchmark.csv",
}

BASELINE = "Liss"   # the reference every relative error is measured against

COLORS = {
    "GVF": "#0072B2",     # blue
    "YOLOE": "#D55E00",   # vermillion
    "SAM2": "#009E73",    # bluish green
    "Liss": "#000000",    # black, the baseline
}

OUT_DIR = "figures"


# loading 

def load_log(method, path):
    """
    Loads one pipeline's per frame CSV log
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

    strides = {m: df["source_frame_index"].diff().dropna().mode().iat[0] for m, df in frames.items()}
    if len(set(strides.values())) > 1:
        print("Warning, the four logs do not share the same frame stride, "
              f"{strides}. Relative error and the per frame plots will only "
              "compare the frame indices the runs actually have in common.")
    return frames


#   raw series per frame, all methods together 

def plot_series_per_frame(frames, value_col, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for m, df in frames.items():
        ax.plot(df["source_frame_index"], df[value_col], label=m,
                 color=COLORS[m], linewidth=1.0)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# each method against Liss on its own

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


# each method on its own, with its mean marked 

def plot_own_output(frames, method, out_path):
    
    df = frames[method]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    sic_mean = df["sic_percent"].mean()
    ax1.plot(df["source_frame_index"], df["sic_percent"],
              color=COLORS[method], linewidth=1.0)
    ax1.axhline(sic_mean, color=COLORS[method], linewidth=1.2, linestyle="--",
                 label=f"Mean = {sic_mean:.1f}%")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("SIC [%]")
    ax1.set_title("Sea ice concentration per frame")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    count_mean = df["floe_count"].mean()
    ax2.plot(df["source_frame_index"], df["floe_count"],
              color=COLORS[method], linewidth=1.0)
    ax2.axhline(count_mean, color=COLORS[method], linewidth=1.2, linestyle="--",
                 label=f"Mean = {count_mean:.1f}")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Floe count")
    ax2.set_title("Floe count per frame")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(f"{method}, own output over the sequence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# relative error against the Liss baseline 

def merged_wide(frames, value_col):
    wide = None
    for m, df in frames.items():
        s = df.set_index("source_frame_index")[value_col].rename(m)
        wide = s.to_frame() if wide is None else wide.join(s, how="inner")
    return wide.sort_index()


def relative_error(wide, baseline=BASELINE):
    """Per frame percent relative error of each method against the
    baseline column"""
    base = wide[baseline]
    out = pd.DataFrame(index=wide.index)
    for col in wide.columns:
        if col == baseline:
            continue
        out[col] = np.where(base != 0, 100.0 * (wide[col] - base) / base, np.nan)
    return out


def plot_relative_error_bar(re_df, ylabel, title, out_path):
    """Relative error per method against a reference line at zero"""
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


# box and whisker summary across methods

def plot_box_whisker(frames, value_col, ylabel, title, out_path, baseline=BASELINE):
    """
    Box and whisker summary of one metric across all four methods, median,
    interquartile range, and whiskers, with the Liss baseline drawn
    alongside the other three.
    """
    methods = list(frames.keys())
    methods = [m for m in methods if m != baseline] + [baseline]
    data = [frames[m][value_col].dropna().values for m in methods]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white",
                                "markeredgecolor": "black", "markersize": 5},
                     medianprops={"color": "black"})
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.6)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# computation time per floe

def plot_time_per_floe(frames, out_path):
    """Time per floe"""
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


# computation time per frame and per floe, side by side

def plot_time_per_frame_and_floe(frames, out_path):
    """Time per frame"""
    methods = list(frames.keys())
    per_frame_ms, per_floe_ms = [], []
    for m in methods:
        df = frames[m]
        total_time = df["exec_time_s"].sum()
        total_floes = df["floe_count"].sum()
        per_frame_ms.append(df["exec_time_s"].mean() * 1000.0)
        per_floe_ms.append((total_time / total_floes) * 1000.0 if total_floes > 0 else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.arange(len(methods))

    ax1.bar(x, per_frame_ms, color=[COLORS[m] for m in methods])
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Mean time per frame [ms]")
    ax1.set_title("Per frame")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.bar(x, per_floe_ms, color=[COLORS[m] for m in methods])
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("Time per floe [ms]")
    ax2.set_title("Per floe")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Computational cost, per frame versus per floe, by method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# entry point

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

    # each method against Liss on its own
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

    # each method entirely on its own, with its mean marked
    for m in frames:
        tag = m.lower()
        plot_own_output(frames, m, os.path.join(OUT_DIR, f"fig_own_{tag}.png"))

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

    # box and whisker summaries
    plot_box_whisker(
        frames, "sic_percent", "SIC [%]",
        "Distribution of sea ice concentration, by method",
        os.path.join(OUT_DIR, "fig_box_sic.png"),
    )
    plot_box_whisker(
        frames, "floe_count", "Floe count",
        "Distribution of floe count, by method",
        os.path.join(OUT_DIR, "fig_box_floe_count.png"),
    )

    plot_time_per_floe(frames, os.path.join(OUT_DIR, "fig_time_per_floe.png"))
    plot_time_per_frame_and_floe(frames, os.path.join(OUT_DIR, "fig_time_per_frame_and_floe.png"))

    print(f"\nDone. Figures saved to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()