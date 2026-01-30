#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare reference and alternative motif-only models for three motif classes (motif A, motif B, both)
with statistical annotations.


For each motif class and each metric (log-likelihood sum, cross-entropy):
  - pairs reference/alternative values
  - computes paired statistics (means, medians, paired t-test, Wilcoxon test)
  - generates boxplots (reference vs alternative):
      row 1: log-likelihood sum of motif-only model
      row 2: cross-entropy of motif-only model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
CSV_PATH = (
    "/Users/amelielaura/Documents/Project6/outputs/"
    "eval_only_motif_based/evaluation/"
    "eval_motif_only__ALTmustHaveDeletions__A100_B100_both100_cleaned.csv"
)
OUTPUT_DIRECTORY = (
    "/Users/amelielaura/Documents/Project6/outputs/"
    "eval_only_motif_based/eval_plots"
)

# --------------------------------------------------------------------
# Colorblind-friendly colors
# --------------------------------------------------------------------
CLASS_COLORS = {
    "motif A": "#56B4E9",      # sky blue
    "motif B": "#009E73",      # bluish green
    "both motifs": "#D55E00",  # vermilion
}

REF_COLOR_LL = "#0072B2"   # dark blue for reference in log-likelihood row
EDGE_COLOR = "#222222"     # dark gray for outlines and jitter points
BG_BOX_TEXT = "#FFFFFF"    # white background for annotation box
MEDIAN_COLOR = "#000000"   # strong black for median line

ALT_ALPHA = 0.9
REF_ALPHA = 0.9


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def ensure_directory_exists(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def convert_series_to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(["NA", ""], np.nan), errors="coerce")


def paired_clean(reference: pd.Series, alternative: pd.Series):
    reference_values = reference.to_numpy(dtype=float)
    alternative_values = alternative.to_numpy(dtype=float)
    mask = np.isfinite(reference_values) & np.isfinite(alternative_values)
    return reference_values[mask], alternative_values[mask]


def paired_stats(reference_values: np.ndarray, alternative_values: np.ndarray):
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

    differences = alternative_values - reference_values
    t_result = stats.ttest_rel(alternative_values, reference_values, nan_policy="omit")

    try:
        w_result = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
        w_p = float(w_result.pvalue)
    except Exception:
        w_p = np.nan

    return {
        "n_pairs": int(reference_values.size),
        "mean_reference": float(np.mean(reference_values)),
        "mean_alternative": float(np.mean(alternative_values)),
        "median_reference": float(np.median(reference_values)),
        "median_alternative": float(np.median(alternative_values)),
        "mean_difference_alternative_minus_reference": float(np.mean(differences)),
        "median_difference_alternative_minus_reference": float(np.median(differences)),
        "paired_t_pvalue": float(t_result.pvalue) if np.isfinite(t_result.pvalue) else np.nan,
        "wilcoxon_pvalue": w_p,
    }


def add_boxplot_subplot(
    ax,
    reference_values,
    alternative_values,
    title,
    y_label,
    stats_dict,
    class_color,
    use_ref_alt_colors=False,
    stats_position="top",
):
    """
    Draw REFERENCE vs ALTERNATIVE boxplot with jittered points and stats annotation.
    """
    bp = ax.boxplot(
        [reference_values, alternative_values],
        labels=["REFERENCE", "ALTERNATIVE"],
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.6),
        whiskerprops=dict(color=EDGE_COLOR, linewidth=1.4),
        capprops=dict(color=EDGE_COLOR, linewidth=1.4),
        boxprops=dict(edgecolor=EDGE_COLOR, linewidth=1.4),
    )

    if use_ref_alt_colors:
        # Example: log-likelihood row – highlight REFERENCE vs ALTERNATIVE with different colors
        bp["boxes"][0].set_facecolor(REF_COLOR_LL)
        bp["boxes"][0].set_alpha(REF_ALPHA)
        bp["boxes"][1].set_facecolor(class_color)
        bp["boxes"][1].set_alpha(ALT_ALPHA)
    else:
        # Example: cross-entropy row – both boxes use the class color, different alpha etc...
        bp["boxes"][0].set_facecolor(class_color)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(class_color)
        bp["boxes"][1].set_alpha(0.9)

    # Jittered points (neutral dark color)
    rng = np.random.default_rng(0)
    for i, values in enumerate([reference_values, alternative_values], start=1):
        if values.size == 0:
            continue
        x = rng.normal(i, 0.045, size=values.size)
        ax.scatter(
            x, values,
            s=10, alpha=0.2,
            color=EDGE_COLOR,
            linewidths=0,
            zorder=1,
        )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    txt = (
        f"n pairs = {stats_dict['n_pairs']}\n"
        f"Δmean (ALTERNATIVE - REFERENCE) = {stats_dict['mean_difference_alternative_minus_reference']:.4g}\n"
        f"Δmedian (ALTERNATIVE - REFERENCE) = {stats_dict['median_difference_alternative_minus_reference']:.4g}\n"
        f"paired t-test pvalue = {stats_dict['paired_t_pvalue']:.3g}\n"
        f"Wilcoxon pvalue = {stats_dict['wilcoxon_pvalue']:.3g}"
    )

    if stats_position == "bottom":
        x_pos, y_pos, va = 0.03, 0.03, "bottom"
    else:
        x_pos, y_pos, va = 0.03, 0.97, "top"

    ax.text(
        x_pos, y_pos, txt,
        transform=ax.transAxes,
        va=va, ha="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=BG_BOX_TEXT,
            alpha=0.9,
            edgecolor="none",
        ),
    )


def compute_global_ylim(arrays, pad_frac=0.08):
    if not arrays:
        return None, None
    all_vals = np.concatenate(arrays)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return None, None

    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    pad = (vmax - vmin) * pad_frac
    return vmin - pad, vmax + pad


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    ensure_directory_exists(OUTPUT_DIRECTORY)

    df = pd.read_csv(CSV_PATH, dtype=str).replace("NA", np.nan)

    if "label" not in df.columns:
        raise ValueError("CSV is expected to contain a 'label' column with A_only/B_only/both.")

    # Map labels to motif classes
    df["motif_class"] = df["label"].replace(
        {
            "A_only": "motif A",
            "B_only": "motif B",
            "both": "both motifs",
        }
    )

    numeric_cols = ["ref_sum_motif", "alt_sum_motif", "ce_ref_motif", "ce_alt_motif"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = convert_series_to_numeric_safe(df[col])

    classes = ["motif A", "motif B", "both motifs"]

    stats_records = []
    ll_all_for_ylim = []
    ce_all_for_ylim = []
    per_class_data = {}

    for cls in classes:
        sub = df[df["motif_class"] == cls]

        ref_ll, alt_ll = paired_clean(sub["ref_sum_motif"], sub["alt_sum_motif"])
        ref_ce, alt_ce = paired_clean(sub["ce_ref_motif"], sub["ce_alt_motif"])

        ll_stats = paired_stats(ref_ll, alt_ll)
        ce_stats = paired_stats(ref_ce, alt_ce)

        per_class_data[cls] = {
            "ref_ll": ref_ll,
            "alt_ll": alt_ll,
            "ll_stats": ll_stats,
            "ref_ce": ref_ce,
            "alt_ce": alt_ce,
            "ce_stats": ce_stats,
        }

        ll_all_for_ylim.extend([ref_ll, alt_ll])
        ce_all_for_ylim.extend([ref_ce, alt_ce])

        stats_records.append(
            {
                "motif_class": cls,
                "metric": "motif_only_loglik_sum",
                **ll_stats,
            }
        )
        stats_records.append(
            {
                "motif_class": cls,
                "metric": "motif_only_cross_entropy",
                **ce_stats,
            }
        )

    ll_ymin, ll_ymax = compute_global_ylim(ll_all_for_ylim, pad_frac=0.10)
    ce_ymin, ce_ymax = compute_global_ylim(ce_all_for_ylim, pad_frac=0.10)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)

    # Row 1: motif-only log-likelihood sum (REFERENCE blue, ALTERNATIVE class color, stats at bottom)
    for j, cls in enumerate(classes):
        ax = axes[0, j]
        data = per_class_data[cls]
        add_boxplot_subplot(
            ax,
            data["ref_ll"],
            data["alt_ll"],
            title=f"{cls}: REF vs ALT motif-only log-likelihood sum",
            y_label="log-likelihood sum (motif-only)" if j == 0 else "",
            stats_dict=data["ll_stats"],
            class_color=CLASS_COLORS[cls],
            use_ref_alt_colors=True,
            stats_position="bottom",
        )
        if ll_ymin is not None:
            ax.set_ylim(ll_ymin, ll_ymax)

    # Row 2: motif-only cross-entropy (both boxes class color, stats at top)
    for j, cls in enumerate(classes):
        ax = axes[1, j]
        data = per_class_data[cls]
        add_boxplot_subplot(
            ax,
            data["ref_ce"],
            data["alt_ce"],
            title=f"{cls}: REF vs ALT motif-only cross-entropy",
            y_label="cross-entropy (motif-only)" if j == 0 else "",
            stats_dict=data["ce_stats"],
            class_color=CLASS_COLORS[cls],
            use_ref_alt_colors=False,
            stats_position="top",
        )
        if ce_ymin is not None:
            ax.set_ylim(ce_ymin, ce_ymax)

    axes[0, 0].set_ylabel("log-likelihood sum (motif-only)")
    axes[1, 0].set_ylabel("cross-entropy (motif-only)")

    fig.suptitle(
        "Motif-only model comparison: REFERENCE vs ALTERNATIVE across motif A, motif B, and both motifs\n"
        "Paired statistics (t-test, Wilcoxon) per motif class and metric",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    plot_path = os.path.join(
        OUTPUT_DIRECTORY,
        "boxplots_reference_alternative_motif_only_loglik_ce.png",
    )
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    stats_df = pd.DataFrame(stats_records)
    stats_path = os.path.join(
        OUTPUT_DIRECTORY,
        "paired_stats_reference_vs_alternative__motif_only.csv",
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved combined plot: {plot_path}")
    print(f"Saved stats table: {stats_path}")


if __name__ == "__main__":
    main()
