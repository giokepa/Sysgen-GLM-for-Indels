"""
Motif and deletion statistics for FASTA sequences
=================================================

This script reads a FASTA file where each header encodes:

- Which motifs are present (`label`)
  - `both`
  - `A_only`
  - `B_only`
  - `no_motif`
- Motif positions (`posAmotif`, `posBmotif`)
- Distance between motifs (`gaplength`)
- Total deletions reported in the header (`deletions`)

From these sequences:

1. Loads all sequences and metadata into a pandas DataFrame.
2. Counts how many sequences fall into each label.
3. Checks that the deletion count in the header matches the deletions in the sequence.
4. For `A_only` and `B_only` sequences:
   - Summarizes motif start positions.
   - Counts total deletions per sequence.
   - Splits deletions into those before and after the motif.
5. For `both` sequences:
   - Summarizes the start positions of motif A and motif B.
   - Counts total deletions per sequence.
   - Counts deletions between motif A and motif B (from end of A to start of B).
6. Saves:
   - Several plots into `PLOT_DIR`.
   - CSV files with per-sequence statistics into `out_dir`.

Run from the command line:

    python3 stats.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

FASTA = Path(
    "/Users/amelielaura/Documents/Project6/data/"
    "augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta"
)

PLOT_DIR = Path("/Users/amelielaura/Documents/Project6/plot_results")

MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
LEN_A = len(MOTIF_A)
LEN_B = len(MOTIF_B)

# Color-blind-friendly palette
CBF_COLORS: Dict[str, str] = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "pink": "#f781bf",
    "brown": "#a65628",
    "purple": "#984ea3",
    "gray": "#999999",
    "red": "#e41a1c",
    "yellow": "#dede00",
}


def set_friendly_style() -> None:
    """grid-based style with readable fonts."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


set_friendly_style()


# -------------------------------------------------------------------
# FASTA parsing
# -------------------------------------------------------------------


def parse_header(header_line: str) -> Dict[str, Any]:
    """
    Parse a FASTA header containing metadata into a dictionary.

    Example format:
    >seq0001|label=both|posAmotif=12|posBmotif=45|gaplength=30|deletions=14
    """
    raw_fields = header_line.lstrip(">").split("|")
    record: Dict[str, Any] = {"id": raw_fields[0]}

    for field in raw_fields[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            record[key] = value

    def to_int_or_none(value: Optional[str]) -> Optional[int]:
        if value in (None, "None", ""):
            return None
        return int(value)

    record["label"] = record.get("label")
    record["posA"] = to_int_or_none(record.get("posAmotif"))
    record["posB"] = to_int_or_none(record.get("posBmotif"))
    record["gap"] = to_int_or_none(record.get("gaplength"))
    record["del_total_header"] = to_int_or_none(record.get("deletions"))
    return record


def read_fasta_with_metadata(path: Path) -> pd.DataFrame:
    """Read a FASTA file with headers into a DataFrame."""
    records: list[Dict[str, Any]] = []
    current_header: Optional[str] = None
    current_seq_lines: list[str] = []

    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    record = parse_header(current_header)
                    record["seq"] = "".join(current_seq_lines)
                    records.append(record)
                current_header = line
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

        if current_header is not None:
            record = parse_header(current_header)
            record["seq"] = "".join(current_seq_lines)
            records.append(record)

    return pd.DataFrame(records)


# -------------------------------------------------------------------
# Deletion helpers
# -------------------------------------------------------------------


def count_deletions(sequence: str) -> int:
    """Count how many deletion characters ('-') appear in the full sequence."""
    return sequence.count("-")


def deletions_before_motif(sequence: str, motif_start: int) -> int:
    """Count deletions that occur before the motif starts."""
    return sequence[:motif_start].count("-")


def deletions_after_motif(sequence: str, motif_start: int, motif_length: int) -> int:
    """Count deletions that occur after the motif ends."""
    motif_end = motif_start + motif_length
    return sequence[motif_end:].count("-")


def deletions_between_motifs(sequence: str, posA: int, posB: int) -> float:
    """
    Count deletions between the end of motif A and the start of motif B.

    Returns numpy.nan if motif B starts before motif A ends.
    """
    start = posA + LEN_A
    end = posB
    if end < start:
        return float(np.nan)
    return float(sequence[start:end].count("-"))


# -------------------------------------------------------------------
# Text summaries
# -------------------------------------------------------------------


def describe_series(label: str, values: np.ndarray) -> None:
    """Print a summary for a 1D numeric array."""
    if values.size == 0:
        print(f"\n{label}: no values to summarize.")
        return

    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    print(f"\n{label}:")
    print(f"  Number of values : {values.size}")
    print(f"  Mean             : {mean_val:.3f}")
    print(f"  Median           : {median_val:.3f}")
    print(f"  Range (min–max)  : {min_val:.3f} – {max_val:.3f}")


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------


def plot_bar(
    ax,
    x_values,
    heights,
    color,
    title: str,
    xlabel: str,
    ylabel: str,
    rotation: int = 15,
) -> None:
    """Draw a simple, good bar chart."""
    ax.bar(x_values, heights, color=color, edgecolor="#333333")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)


def plot_hist(
    ax,
    data,
    bins,
    color,
    label: Optional[str] = None,
    alpha: float = 0.7,
    xlabel: str = "",
    ylabel: str = "Count",
    title: str = "",
) -> None:
    """Draw a histogram with defaults."""
    ax.hist(
        data,
        bins=bins,
        color=color,
        alpha=alpha,
        label=label,
        edgecolor="white",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure into PLOT_DIR and report the destination."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PLOT_DIR / filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {outpath}")


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------


def main() -> None:
    """Run the full analysis on the FASTA file."""
    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA file not found at: {FASTA}")

    df = read_fasta_with_metadata(FASTA)

    # Basic per-sequence information
    df["length"] = df["seq"].str.len()
    df["del_total_calc"] = df["seq"].str.count("-")
    df["del_header_matches"] = df["del_total_calc"].eq(df["del_total_header"])

    # How many sequences of each motif pattern?
    counts = (
        df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    counts["percent"] = (counts["count"] / len(df) * 100).round(2)

    print("\n=== Overview: sequences by motif pattern ===")
    print(counts.to_string(index=False))

    # Quick sanity checks
    print("\n=== Checks ===")
    print(f"- Total number of sequences             : {len(df)}")
    lengths = sorted(df["length"].unique().tolist())
    print(f"- Sequence lengths observed (first 10)  : {lengths[:10]}")
    print(f"- Number of distinct sequence lengths   : {df['length'].nunique()}")
    match_rate = df["del_header_matches"].mean() * 100
    print(
        f"- Headers where deletion count matches  : "
        f"{match_rate:.2f}% of sequences"
    )

    # Split by motif label
    a_only = df[df["label"] == "A_only"].copy()
    b_only = df[df["label"] == "B_only"].copy()
    both_motifs = df[df["label"] == "both"].copy()

    # Deletion counts relative to motifs
    def count_deletions_before_A(row: pd.Series) -> int:
        return deletions_before_motif(row["seq"], int(row["posA"]))

    def count_deletions_after_A(row: pd.Series) -> int:
        return deletions_after_motif(row["seq"], int(row["posA"]), LEN_A)

    def count_deletions_before_B(row: pd.Series) -> int:
        return deletions_before_motif(row["seq"], int(row["posB"]))

    def count_deletions_after_B(row: pd.Series) -> int:
        return deletions_after_motif(row["seq"], int(row["posB"]), LEN_B)

    def count_deletions_between(row: pd.Series) -> float:
        return deletions_between_motifs(row["seq"], int(row["posA"]), int(row["posB"]))

    if len(a_only) > 0:
        a_only["del_before_A"] = a_only.apply(count_deletions_before_A, axis=1)
        a_only["del_after_A"] = a_only.apply(count_deletions_after_A, axis=1)

    if len(b_only) > 0:
        b_only["del_before_B"] = b_only.apply(count_deletions_before_B, axis=1)
        b_only["del_after_B"] = b_only.apply(count_deletions_after_B, axis=1)

    if len(both_motifs) > 0:
        both_motifs["del_between"] = both_motifs.apply(count_deletions_between, axis=1)

    # Text summaries
    print("\n=== Deletion summaries (per sequence) ===")
    describe_series(
        "A_only: total deletions in sequence",
        a_only["del_total_calc"].to_numpy() if len(a_only) else np.array([]),
    )
    describe_series(
        "B_only: total deletions in sequence",
        b_only["del_total_calc"].to_numpy() if len(b_only) else np.array([]),
    )
    describe_series(
        "both: total deletions in sequence",
        both_motifs["del_total_calc"].to_numpy() if len(both_motifs) else np.array([]),
    )
    describe_series(
        "no_motif: total deletions in sequence",
        df[df["label"] == "no_motif"]["del_total_calc"].to_numpy(),
    )

    if len(a_only):
        describe_series(
            "A_only: deletions BEFORE motif A",
            a_only["del_before_A"].to_numpy(),
        )
        describe_series(
            "A_only: deletions AFTER motif A",
            a_only["del_after_A"].to_numpy(),
        )
    if len(b_only):
        describe_series(
            "B_only: deletions BEFORE motif B",
            b_only["del_before_B"].to_numpy(),
        )
        describe_series(
            "B_only: deletions AFTER motif B",
            b_only["del_after_B"].to_numpy(),
        )
    if len(both_motifs):
        describe_series(
            "both: deletions BETWEEN motif A and motif B",
            both_motifs["del_between"].dropna().to_numpy(),
        )

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------

    # 1) label_counts.png
    fig, ax = plt.subplots()
    plot_bar(
        ax=ax,
        x_values=counts["label"],
        heights=counts["count"],
        color=CBF_COLORS["blue"],
        title="How many sequences per motif pattern?",
        xlabel="Motif pattern in sequence",
        ylabel="Number of sequences",
    )
    plt.tight_layout()
    save_figure(fig, "label_counts.png")
    plt.show()

    # Binning for motif positions
    seq_length = int(df["length"].iloc[0])
    pos_bins = np.arange(0, seq_length + 1, 5)

    # 2) motif_positions_single.png
    if len(a_only) and len(b_only):
        fig, ax = plt.subplots()
        plot_hist(
            ax=ax,
            data=a_only["posA"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["blue"],
            label="Motif A (A_only)",
            xlabel="Motif start position (0-based)",
            ylabel="Number of sequences",
            title="Motif start positions in sequences with a single motif",
        )
        plot_hist(
            ax=ax,
            data=b_only["posB"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["orange"],
            label="Motif B (B_only)",
        )
        ax.legend(title="Motif type")
        plt.tight_layout()
        save_figure(fig, "motif_positions_single.png")
        plt.show()

    # 3) motif_positions_both.png
    if len(both_motifs):
        fig, ax = plt.subplots()
        plot_hist(
            ax=ax,
            data=both_motifs["posA"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["green"],
            label="Motif A (both)",
            xlabel="Motif start position (0-based)",
            ylabel="Number of sequences",
            title="Motif start positions when both motifs are present",
        )
        plot_hist(
            ax=ax,
            data=both_motifs["posB"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["purple"],
            label="Motif B (both)",
        )
        ax.legend(title="Motif")
        plt.tight_layout()
        save_figure(fig, "motif_positions_both.png")
        plt.show()

    # 4) total_deletions_by_label.png
    fig, ax = plt.subplots()
    label_colors = {
        "both": CBF_COLORS["blue"],
        "A_only": CBF_COLORS["orange"],
        "B_only": CBF_COLORS["green"],
        "no_motif": CBF_COLORS["gray"],
    }
    for label, color in label_colors.items():
        subset = df[df["label"] == label]["del_total_calc"]
        if not len(subset):
            continue
        bins = np.arange(subset.min(), subset.max() + 2) - 0.5
        ax.hist(
            subset,
            bins=bins,
            alpha=0.6,
            label=label.replace("_", " "),
            color=color,
            edgecolor="white",
        )
    ax.set_xlabel("Total deletions per sequence (count of '-')")
    ax.set_ylabel("Number of sequences")
    ax.set_title("Total deletions per sequence for each motif pattern")
    ax.legend(title="Motif pattern")
    plt.tight_layout()
    save_figure(fig, "total_deletions_by_label.png")
    plt.show()

    # 5) A_only_before_after.png
    if len(a_only):
        fig, ax = plt.subplots()
        bins_before = (
            np.arange(a_only["del_before_A"].min(), a_only["del_before_A"].max() + 2)
            - 0.5
        )
        bins_after = (
            np.arange(a_only["del_after_A"].min(), a_only["del_after_A"].max() + 2)
            - 0.5
        )
        ax.hist(
            a_only["del_before_A"],
            bins=bins_before,
            alpha=0.7,
            label="Before motif A",
            color=CBF_COLORS["blue"],
            edgecolor="white",
        )
        ax.hist(
            a_only["del_after_A"],
            bins=bins_after,
            alpha=0.7,
            label="After motif A",
            color=CBF_COLORS["orange"],
            edgecolor="white",
        )
        ax.set_xlabel("Number of deletions")
        ax.set_ylabel("Number of sequences")
        ax.set_title("A_only sequences: deletions before vs after motif A")
        ax.legend()
        plt.tight_layout()
        save_figure(fig, "A_only_before_after.png")
        plt.show()

    # 6) B_only_before_after.png
    if len(b_only):
        fig, ax = plt.subplots()
        bins_before = (
            np.arange(b_only["del_before_B"].min(), b_only["del_before_B"].max() + 2)
            - 0.5
        )
        bins_after = (
            np.arange(b_only["del_after_B"].min(), b_only["del_after_B"].max() + 2)
            - 0.5
        )
        ax.hist(
            b_only["del_before_B"],
            bins=bins_before,
            alpha=0.7,
            label="Before motif B",
            color=CBF_COLORS["green"],
            edgecolor="white",
        )
        ax.hist(
            b_only["del_after_B"],
            bins=bins_after,
            alpha=0.7,
            label="After motif B",
            color=CBF_COLORS["purple"],
            edgecolor="white",
        )
        ax.set_xlabel("Number of deletions")
        ax.set_ylabel("Number of sequences")
        ax.set_title("B_only sequences: deletions before vs after motif B")
        ax.legend()
        plt.tight_layout()
        save_figure(fig, "B_only_before_after.png")
        plt.show()

    # 7) both_deletions_between.png
    if len(both_motifs):
        between = both_motifs["del_between"].dropna()
        fig, ax = plt.subplots()
        bins = np.arange(between.min(), between.max() + 2) - 0.5
        plot_hist(
            ax=ax,
            data=between,
            bins=bins,
            color=CBF_COLORS["red"],
            xlabel="Deletions between motif A and motif B",
            ylabel="Number of sequences",
            title="Deletions between motifs in sequences that contain both motifs",
        )
        plt.tight_layout()
        save_figure(fig, "both_deletions_between.png")
        plt.show()

    # ----------------------------------------------------------------
    # CSV outputs
    # ----------------------------------------------------------------

    out_dir = Path("/Users/amelielaura/Documents/Project6/data/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts.to_csv(out_dir / "label_counts.csv", index=False)
    if len(a_only):
        a_only[["id", "posA", "del_total_calc", "del_before_A", "del_after_A"]].to_csv(
            out_dir / "A_only_stats.csv", index=False
        )
    if len(b_only):
        b_only[["id", "posB", "del_total_calc", "del_before_B", "del_after_B"]].to_csv(
            out_dir / "B_only_stats.csv", index=False
        )
    if len(both_motifs):
        both_motifs[
            ["id", "posA", "posB", "gap", "del_total_calc", "del_between"]
        ].to_csv(out_dir / "both_stats.csv", index=False)

    print(f"\nAll CSV summaries have been saved in: {out_dir.resolve()}")
    print(f"Finito! All plots saved in: {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
