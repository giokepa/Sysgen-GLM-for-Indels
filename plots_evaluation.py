import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = (
    "/Users/amelielaura/Documents/"
    "eval_motif_only__PAIRED__ALT_DUPBOTH__A65_B47_bothFlanks100_bothBetween20_cleaned.csv"
)

OUTPUT_DIRECTORY = (
    "/Users/amelielaura/Documents/Project6/outputs/"
    "eval_only_motif_based/eval_plots"
)

PANEL_COLORS: Dict[str, str] = {
    "A_only": "#56B4E9",
    "B_only": "#009E73",
    "both_flanks": "#D55E00",
    "both_between": "#CC79A7",
}

REF_COLOR = "#0072B2"
EDGE_COLOR = "#222222"
MEDIAN_COLOR = "#000000"
BG_BOX_TEXT = "#FFFFFF"

ALT_ALPHA = 0.90
REF_ALPHA = 0.90

def ensure_directory_exists(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)

def convert_series_to_numeric_safe(series: pd.Series) -> pd.Series:
    clean_series = series.replace(["NA", ""], np.nan)
    numeric_series = pd.to_numeric(clean_series, errors="coerce")
    return numeric_series


def paired_clean(reference: pd.Series, alternative: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    reference_array = reference.to_numpy(dtype=float)
    alternative_array = alternative.to_numpy(dtype=float)

    finite_mask = np.isfinite(reference_array) & np.isfinite(alternative_array)
    reference_clean = reference_array[finite_mask]
    alternative_clean = alternative_array[finite_mask]

    return reference_clean, alternative_clean


def paired_stats(reference_values: np.ndarray, alternative_values: np.ndarray) -> Dict[str, Any]:
    if reference_values.size < 2:
        return {
            "n_pairs": int(reference_values.size),
            "mean_reference": np.nan,
            "mean_alternative": np.nan,
            "median_reference": np.nan,
            "median_alternative": np.nan,
            "mean_difference_alternative_minus_reference": np.nan,
            "median_difference_alternative_minus_reference": np.nan,
            "paired_t_pvalue": np.nan,
            "wilcoxon_pvalue": np.nan,
        }

    diffs = alternative_values - reference_values

    t_result = stats.ttest_rel(
        alternative_values,
        reference_values,
        nan_policy="omit",
    )

    try:
        wilcoxon_result = stats.wilcoxon(
            diffs,
            zero_method="wilcox",
            alternative="two-sided",
        )
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)
    except Exception:
        wilcoxon_pvalue = np.nan

    stats_dict: Dict[str, Any] = {
        "n_pairs": int(reference_values.size),
        "mean_reference": float(np.mean(reference_values)),
        "mean_alternative": float(np.mean(alternative_values)),
        "median_reference": float(np.median(reference_values)),
        "median_alternative": float(np.median(alternative_values)),
        "mean_difference_alternative_minus_reference": float(np.mean(diffs)),
        "median_difference_alternative_minus_reference": float(np.median(diffs)),
        "paired_t_pvalue": float(t_result.pvalue) if np.isfinite(t_result.pvalue) else np.nan,
        "wilcoxon_pvalue": wilcoxon_pvalue,
    }
    return stats_dict


def add_boxplot_subplot(
    ax: plt.Axes,
    reference_values: np.ndarray,
    alternative_values: np.ndarray,
    title: str,
    stats_dict: Dict[str, Any],
    alt_color: str,
) -> None:
    boxplot_result = ax.boxplot(
        [reference_values, alternative_values],
        labels=["REFERENCE", "ALTERNATIVE"],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": MEDIAN_COLOR, "linewidth": 2.4},
        whiskerprops={"color": EDGE_COLOR, "linewidth": 1.2},
        capprops={"color": EDGE_COLOR, "linewidth": 1.2},
        boxprops={"edgecolor": EDGE_COLOR, "linewidth": 1.2},
    )

    reference_box = boxplot_result["boxes"][0]
    alternative_box = boxplot_result["boxes"][1]

    reference_box.set_facecolor(REF_COLOR)
    reference_box.set_alpha(REF_ALPHA)

    alternative_box.set_facecolor(alt_color)
    alternative_box.set_alpha(ALT_ALPHA)

    random_generator = np.random.default_rng(0)
    value_lists = [reference_values, alternative_values]

    for group_index, values in enumerate(value_lists, start=1):
        if values.size == 0:
            continue

        x_positions = random_generator.normal(loc=group_index, scale=0.045, size=values.size)
        ax.scatter(
            x_positions,
            values,
            s=10,
            alpha=0.25,
            color=EDGE_COLOR,
            linewidths=0,
            zorder=1,
        )

    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    text_lines = (
        f"n = {stats_dict['n_pairs']}\n"
        f"Δmean (ALT-REF) = {stats_dict['mean_difference_alternative_minus_reference']:.4g}\n"
        f"Δmedian (ALT-REF) = {stats_dict['median_difference_alternative_minus_reference']:.4g}\n"
        f"t-test p = {stats_dict['paired_t_pvalue']:.3g}\n"
        f"Wilcoxon p = {stats_dict['wilcoxon_pvalue']:.3g}"
    )

    ax.text(
        0.03,
        0.97,
        text_lines,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": BG_BOX_TEXT,
            "alpha": 0.9,
            "edgecolor": "none",
        },
    )


def compute_global_ylim(arrays: List[np.ndarray], pad_frac: float = 0.10) -> Tuple[float, float]:
    if len(arrays) == 0:
        return None, None

    concatenated = np.concatenate(arrays)
    finite_vals = concatenated[np.isfinite(concatenated)]
    if finite_vals.size == 0:
        return None, None

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))

    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    padding = (vmax - vmin) * pad_frac
    return vmin - padding, vmax + padding

def main() -> None:

    ensure_directory_exists(OUTPUT_DIRECTORY)
    data_frame = pd.read_csv(CSV_PATH, dtype=str).replace("NA", np.nan)

    # Decide which column defines the grouping into panels.
    if "bucket" in data_frame.columns:
        group_column = "bucket"
    elif "label" in data_frame.columns:
        group_column = "label"
    else:
        raise ValueError("CSV must contain either 'bucket' or 'label' column.")

    required_columns = ["ce_ref_motif", "ce_alt_motif"]
    for column_name in required_columns:
        if column_name not in data_frame.columns:
            raise ValueError(f"Missing required column '{column_name}' in CSV.")
        data_frame[column_name] = convert_series_to_numeric_safe(data_frame[column_name])
    panel_keys = ["A_only", "B_only", "both_flanks", "both_between"]

    stats_records: List[Dict[str, Any]] = []
    all_ce_arrays: List[np.ndarray] = []

    figure, axes_array = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    flat_axes: List[plt.Axes] = list(axes_array.flatten())

    for axis, key in zip(flat_axes, panel_keys):
        subset = data_frame[data_frame[group_column] == key]
        if subset.empty:
            axis.set_title(f"{key}: (no data)", fontsize=11)
            axis.axis("off")
            continue

        ref_ce, alt_ce = paired_clean(subset["ce_ref_motif"], subset["ce_alt_motif"])
        ce_stats = paired_stats(ref_ce, alt_ce)

        all_ce_arrays.append(ref_ce)
        all_ce_arrays.append(alt_ce)

        stats_record: Dict[str, Any] = {"group": key, "metric": "motif_only_cross_entropy"}
        stats_record.update(ce_stats)
        stats_records.append(stats_record)

        panel_title = f"{key}: REFERENCE vs ALTERNATIVE motif-only cross-entropy"
        alt_color = PANEL_COLORS.get(key, "#999999")

        add_boxplot_subplot(
            ax=axis,
            reference_values=ref_ce,
            alternative_values=alt_ce,
            title=panel_title,
            stats_dict=ce_stats,
            alt_color=alt_color,
        )
    global_ymin, global_ymax = compute_global_ylim(all_ce_arrays, pad_frac=0.10)
    if global_ymin is not None:
        for axis in flat_axes:
            if axis.has_data():
                axis.set_ylim(global_ymin, global_ymax)
    flat_axes[0].set_ylabel("cross-entropy (motif-only)")
    flat_axes[2].set_ylabel("cross-entropy (motif-only)")

    figure.suptitle(
        "Motif-only cross-entropy: REFERENCE vs ALTERNATIVE\n"
        "Paired tests: paired t-test and Wilcoxon signed-rank",
        fontsize=14,
    )
    figure.tight_layout(rect=[0, 0.03, 1, 0.93])

    plot_path = os.path.join(
        OUTPUT_DIRECTORY,
        "boxplots_REF_vs_ALT__motif_only_CE__4panels.png",
    )
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    stats_data_frame = pd.DataFrame(stats_records)
    stats_path = os.path.join(
        OUTPUT_DIRECTORY,
        "paired_stats_REF_vs_ALT__motif_only_CE__4panels.csv",
    )
    stats_data_frame.to_csv(stats_path, index=False)

    print(f"Saved combined plot: {plot_path}")
    print(f"Saved stats table: {stats_path}")
    print(f"Grouping column used: {group_column}")


if __name__ == "__main__":
    main()
