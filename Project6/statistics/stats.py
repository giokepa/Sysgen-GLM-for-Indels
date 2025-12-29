"""
What this script does
---------------------

1) Reads a FASTA file where each header stores:
      - label (which motifs are present)
      - motif positions (posAmotif, posBmotif)
      - gaplength
      - deletions (total deletions reported in header)

2) Counts how many sequences fall into each label:
      - both
      - A_only
      - B_only
      - no_motif

3) For A_only and B_only:
      - distribution of motif start positions
      - total deletions per sequence
      - deletions BEFORE the motif and AFTER the motif
        (so we see where deletions tend to cluster)

4) For both:
      - motif A and motif B start position distributions
      - total deletions per sequence
      - deletions BETWEEN motifs
        (region from end of A to start of B)

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
FASTA = Path("/Users/amelielaura/Documents/Project6/data/augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta")

MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
LEN_A = len(MOTIF_A)
LEN_B = len(MOTIF_B)

# Color‑blind‑friendly palette from internet
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


def friendly_style() -> None:
    """Set a light, grid‑based style with good fonts."""
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


friendly_style()

# -------------------------------------------------------------------
# FASTA parsing
# -------------------------------------------------------------------


def parse_header(header_line: str) -> Dict[str, Any]:
    """
    Turn a FASTA header with embedded metadata into a readable dictionary.

    Expected format (one example):
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
    """Read a FASTA file with metadata‑rich headers into a DataFrame."""
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
    """Print a compact summary for a 1D numeric array."""
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
    """Draw a simple, readable bar chart."""
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
    """Draw a histogram with gentle defaults."""
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


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------


def main() -> None:
    """Run the full analysis on the FASTA file."""
    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA file not found at: {FASTA}")

    df = read_fasta_with_metadata(FASTA)

    df["length"] = df["seq"].str.len()
    df["del_total_calc"] = df["seq"].str.count("-")
    df["del_header_matches"] = df["del_total_calc"].eq(df["del_total_header"])

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

    print("\n=== Quick sanity checks ===")
    print(f"- Total number of sequences             : {len(df)}")
    lengths = sorted(df["length"].unique().tolist())
    print(f"- Sequence lengths observed (first 10)  : {lengths[:10]}")
    print(f"- Number of distinct sequence lengths   : {df['length'].nunique()}")
    match_rate = df["del_header_matches"].mean() * 100
    print(
        f"- Headers where deletion count matches  : "
        f"{match_rate:.2f}% of sequences"
    )

    Aonly = df[df["label"] == "A_only"].copy()
    Bonly = df[df["label"] == "B_only"].copy()
    BOTH = df[df["label"] == "both"].copy()

    def compute_del_before_A(row: pd.Series) -> int:
        return deletions_before_motif(row["seq"], int(row["posA"]))

    def compute_del_after_A(row: pd.Series) -> int:
        return deletions_after_motif(row["seq"], int(row["posA"]), LEN_A)

    def compute_del_before_B(row: pd.Series) -> int:
        return deletions_before_motif(row["seq"], int(row["posB"]))

    def compute_del_after_B(row: pd.Series) -> int:
        return deletions_after_motif(row["seq"], int(row["posB"]), LEN_B)

    def compute_del_between(row: pd.Series) -> float:
        return deletions_between_motifs(row["seq"], int(row["posA"]), int(row["posB"]))

    if len(Aonly) > 0:
        Aonly["del_before_A"] = Aonly.apply(compute_del_before_A, axis=1)
        Aonly["del_after_A"] = Aonly.apply(compute_del_after_A, axis=1)

    if len(Bonly) > 0:
        Bonly["del_before_B"] = Bonly.apply(compute_del_before_B, axis=1)
        Bonly["del_after_B"] = Bonly.apply(compute_del_after_B, axis=1)

    if len(BOTH) > 0:
        BOTH["del_between"] = BOTH.apply(compute_del_between, axis=1)

    print("\n=== Deletion summaries (per sequence) ===")
    describe_series(
        "A_only: total deletions in sequence",
        Aonly["del_total_calc"].to_numpy() if len(Aonly) else np.array([]),
    )
    describe_series(
        "B_only: total deletions in sequence",
        Bonly["del_total_calc"].to_numpy() if len(Bonly) else np.array([]),
    )
    describe_series(
        "both: total deletions in sequence",
        BOTH["del_total_calc"].to_numpy() if len(BOTH) else np.array([]),
    )
    describe_series(
        "no_motif: total deletions in sequence",
        df[df["label"] == "no_motif"]["del_total_calc"].to_numpy(),
    )

    if len(Aonly):
        describe_series(
            "A_only: deletions BEFORE motif A",
            Aonly["del_before_A"].to_numpy(),
        )
        describe_series(
            "A_only: deletions AFTER motif A",
            Aonly["del_after_A"].to_numpy(),
        )
    if len(Bonly):
        describe_series(
            "B_only: deletions BEFORE motif B",
            Bonly["del_before_B"].to_numpy(),
        )
        describe_series(
            "B_only: deletions AFTER motif B",
            Bonly["del_after_B"].to_numpy(),
        )
    if len(BOTH):
        describe_series(
            "both: deletions BETWEEN motif A and motif B",
            BOTH["del_between"].dropna().to_numpy(),
        )

    # Plots
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
    plt.show()

    L = int(df["length"].iloc[0])
    pos_bins = np.arange(0, L + 1, 5)

    if len(Aonly) and len(Bonly):
        fig, ax = plt.subplots()
        plot_hist(
            ax=ax,
            data=Aonly["posA"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["blue"],
            label="Motif A (A_only)",
            xlabel="Motif start position (0-based)",
            ylabel="Number of sequences",
            title="Motif start positions in sequences with a single motif",
        )
        plot_hist(
            ax=ax,
            data=Bonly["posB"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["orange"],
            label="Motif B (B_only)",
        )
        ax.legend(title="Motif type")
        plt.tight_layout()
        plt.show()

    if len(BOTH):
        fig, ax = plt.subplots()
        plot_hist(
            ax=ax,
            data=BOTH["posA"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["green"],
            label="Motif A (both)",
            xlabel="Motif start position (0-based)",
            ylabel="Number of sequences",
            title="Motif start positions when both motifs are present",
        )
        plot_hist(
            ax=ax,
            data=BOTH["posB"].astype(int),
            bins=pos_bins,
            color=CBF_COLORS["purple"],
            label="Motif B (both)",
        )
        ax.legend(title="Motif")
        plt.tight_layout()
        plt.show()

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
    plt.show()

    if len(Aonly):
        fig, ax = plt.subplots()
        bins_before = (
            np.arange(Aonly["del_before_A"].min(), Aonly["del_before_A"].max() + 2) - 0.5
        )
        bins_after = (
            np.arange(Aonly["del_after_A"].min(), Aonly["del_after_A"].max() + 2) - 0.5
        )
        ax.hist(
            Aonly["del_before_A"],
            bins=bins_before,
            alpha=0.7,
            label="Before motif A",
            color=CBF_COLORS["blue"],
            edgecolor="white",
        )
        ax.hist(
            Aonly["del_after_A"],
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
        plt.show()

    if len(Bonly):
        fig, ax = plt.subplots()
        bins_before = (
            np.arange(Bonly["del_before_B"].min(), Bonly["del_before_B"].max() + 2) - 0.5
        )
        bins_after = (
            np.arange(Bonly["del_after_B"].min(), Bonly["del_after_B"].max() + 2) - 0.5
        )
        ax.hist(
            Bonly["del_before_B"],
            bins=bins_before,
            alpha=0.7,
            label="Before motif B",
            color=CBF_COLORS["green"],
            edgecolor="white",
        )
        ax.hist(
            Bonly["del_after_B"],
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
        plt.show()

    if len(BOTH):
        between = BOTH["del_between"].dropna()
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
        plt.show()

    out_dir = Path("/Users/amelielaura/Documents/Project6/data/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts.to_csv(out_dir / "label_counts.csv", index=False)
    if len(Aonly):
        Aonly[["id", "posA", "del_total_calc", "del_before_A", "del_after_A"]].to_csv(
            out_dir / "A_only_stats.csv", index=False
        )
    if len(Bonly):
        Bonly[["id", "posB", "del_total_calc", "del_before_B", "del_after_B"]].to_csv(
            out_dir / "B_only_stats.csv", index=False
        )
    if len(BOTH):
        BOTH[["id", "posA", "posB", "gap", "del_total_calc", "del_between"]].to_csv(
            out_dir / "both_stats.csv", index=False
        )

    print(f"\nAll CSV summaries have been saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
