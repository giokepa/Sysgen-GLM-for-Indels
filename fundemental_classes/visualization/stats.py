"""
What this script does
---------------------

1) Reads a FASTA file where each header line contains:
   - A label describing which motifs are present
   - The positions of the motifs (posAmotif, posBmotif)
   - The gap length between motifs
   - The total number of deletions reported in the header

2) Counts how many sequences fall into each label:
   - both
   - A_only
   - B_only
   - no_motif

3) For A_only and B_only:
   - Distribution of motif start positions
   - Total deletions per sequence
   - Deletions BEFORE and AFTER the motif
     (to see where deletions tend to cluster along the sequence)

4) For both:
   - Start position distributions for motif A and motif B
   - Total deletions per sequence
   - Deletions BETWEEN motifs
     (from the end of motif A to the start of motif B)

Run:
   python3 stats.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
# Input FASTA (augmented sequences with deletions encoded as "-")
FASTA = Path(
    "/Users/amelielaura/Documents/Project6/data/"
    "augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta"
)

# Output locations
OUT_DIR = Path("/Users/amelielaura/Documents/Project6/data/out")         # CSVs
PLOT_DIR = Path("/Users/amelielaura/Documents/Project6/plotresults")    # plots

# Motif definitions
MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
LEN_A = len(MOTIF_A)
LEN_B = len(MOTIF_B)

# Color-blind-friendly palette
CBF_COLORS: Dict[str, str] = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "purple": "#984ea3",
    "gray": "#999999",
    "red": "#e41a1c",
}

# labels for plotting
LABEL_PRETTY = {
    "both": "both motifs",
    "A_only": "only motif A",
    "B_only": "only motif B",
    "no_motif": "none",
}


def friendly_style() -> None:
    """Use a good Matplotlib style that is easy to read."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
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
    Parse a FASTA header line that encodes motif metadata.

    Example:
        seq0001|label=both|posAmotif=12|posBmotif=45|gaplength=30|deletions=14
    """
    raw_fields = header_line.lstrip(">").split("|")
    record: Dict[str, Any] = {"id": raw_fields[0]}

    # Everything after the ID is key=value
    for field in raw_fields[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            record[key] = value

    def to_int_or_none(value: Optional[str]) -> Optional[int]:
        if value in (None, "None", ""):
            return None
        return int(value)

    record["label"] = record.get("label")
    record["position_of_motif_A"] = to_int_or_none(record.get("posAmotif"))
    record["position_of_motif_B"] = to_int_or_none(record.get("posBmotif"))
    record["gaplength"] = to_int_or_none(record.get("gaplength"))
    record["deletions_header"] = to_int_or_none(record.get("deletions"))
    return record


def read_fasta_with_metadata(path: Path) -> pd.DataFrame:
    """
    Read a FASTA file where headers contain motif metadata.

    Returns a DataFrame with one row per sequence:
    - id, label, motif positions, gaplength, deletions_header, seq
    """
    records: list[Dict[str, Any]] = []
    current_header: Optional[str] = None
    current_seq_lines: list[str] = []

    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # If there is a previous record, finalize it
                if current_header is not None:
                    record = parse_header(current_header)
                    record["seq"] = "".join(current_seq_lines)
                    records.append(record)
                current_header = line
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

        # Add the final record (if any)
        if current_header is not None:
            record = parse_header(current_header)
            record["seq"] = "".join(current_seq_lines)
            records.append(record)

    return pd.DataFrame(records)


# -------------------------------------------------------------------
# Deletion helpers
# -------------------------------------------------------------------
def deletions_before_motif(sequence: str, motif_start: int) -> int:
    """Count how many deletions occur before a motif start."""
    return sequence[:motif_start].count("-")


def deletions_after_motif(sequence: str, motif_start: int, motif_length: int) -> int:
    """Count how many deletions occur after a motif, from its end to the end of the sequence."""
    motif_end = motif_start + motif_length
    return sequence[motif_end:].count("-")


def deletions_between_motifs(sequence: str, pos_a: int, pos_b: int) -> float:
    """
    Count deletions between motif A and motif B.

    We look at the region from the end of motif A up to the start of motif B:
        [pos_a + LEN_A : pos_b]

    Assumes motif A appears before motif B.
    """
    start = pos_a + LEN_A
    end = pos_b
    if end < start:
        return float(np.nan)
    return float(sequence[start:end].count("-"))


# -------------------------------------------------------------------
# Text summary helper
# -------------------------------------------------------------------
def describe_series(label: str, values: np.ndarray) -> None:
    """Print a numeric summary."""
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
# Main analysis
# -------------------------------------------------------------------
def main() -> None:
    # Basic checks and directory setup
    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA file not found at: {FASTA}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Read sequences + metadata
    df = read_fasta_with_metadata(FASTA)

    # Core derived columns
    df["sequence_length"] = df["seq"].str.len()
    df["total_deletions_per_sequence"] = df["seq"].str.count("-").astype(int)
    df["del_header_matches"] = df["total_deletions_per_sequence"].eq(
        df["deletions_header"]
    )

    # How many sequences per label?
    counts = (
        df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    counts["percent"] = (counts["count"] / len(df) * 100).round(2)

    # ---------------- Overview prints ----------------
    print("\n=== Overview: sequences by motif pattern ===")
    print(counts.to_string(index=False))

    print("\n=== Quick sanity checks ===")
    print(f"- Total number of sequences             : {len(df)}")
    lengths = sorted(df["sequence_length"].unique().tolist())
    print(f"- Sequence lengths observed (first 10)  : {lengths[:10]}")
    print(
        f"- Number of distinct sequence lengths   : "
        f"{df['sequence_length'].nunique()}"
    )
    match_rate = df["del_header_matches"].mean() * 100
    print(
        f"- Headers where deletion count matches  : "
        f"{match_rate:.2f}% of sequences"
    )

    # Split into groups by label
    only_a = df[df["label"] == "A_only"].copy()
    only_b = df[df["label"] == "B_only"].copy()
    both = df[df["label"] == "both"].copy()
    none = df[df["label"] == "no_motif"].copy()

    # For A_only: deletions before/after motif A
    if len(only_a) > 0:
        only_a["del_before_A"] = only_a.apply(
            lambda r: deletions_before_motif(
                r["seq"], int(r["position_of_motif_A"])
            ),
            axis=1,
        )
        only_a["del_after_A"] = only_a.apply(
            lambda r: deletions_after_motif(
                r["seq"], int(r["position_of_motif_A"]), LEN_A
            ),
            axis=1,
        )

    # For B_only: deletions before/after motif B
    if len(only_b) > 0:
        only_b["del_before_B"] = only_b.apply(
            lambda r: deletions_before_motif(
                r["seq"], int(r["position_of_motif_B"])
            ),
            axis=1,
        )
        only_b["del_after_B"] = only_b.apply(
            lambda r: deletions_after_motif(
                r["seq"], int(r["position_of_motif_B"]), LEN_B
            ),
            axis=1,
        )

    # For both motifs: keep only sequences where A is before B
    both_valid = both.dropna(
        subset=["position_of_motif_A", "position_of_motif_B"]
    ).copy()
    both_valid = both_valid[
        both_valid["position_of_motif_A"] < both_valid["position_of_motif_B"]
    ]

    if len(both_valid) > 0:
        both_valid["del_between"] = both_valid.apply(
            lambda r: deletions_between_motifs(
                r["seq"],
                int(r["position_of_motif_A"]),
                int(r["position_of_motif_B"]),
            ),
            axis=1,
        )

    # ---------------- Numeric summaries ----------------
    print("\n=== Deletion summaries (per sequence) ===")
    describe_series(
        "only motif A: total deletions in sequence",
        only_a["total_deletions_per_sequence"].to_numpy()
        if len(only_a)
        else np.array([]),
    )
    describe_series(
        "only motif B: total deletions in sequence",
        only_b["total_deletions_per_sequence"].to_numpy()
        if len(only_b)
        else np.array([]),
    )
    describe_series(
        "both motifs: total deletions in sequence",
        both["total_deletions_per_sequence"].to_numpy()
        if len(both)
        else np.array([]),
    )
    describe_series(
        "none: total deletions in sequence",
        none["total_deletions_per_sequence"].to_numpy()
        if len(none)
        else np.array([]),
    )

    if len(only_a):
        describe_series(
            "only motif A: deletions BEFORE motif A",
            only_a["del_before_A"].to_numpy(),
        )
        describe_series(
            "only motif A: deletions AFTER  motif A",
            only_a["del_after_A"].to_numpy(),
        )

    if len(only_b):
        describe_series(
            "only motif B: deletions BEFORE motif B",
            only_b["del_before_B"].to_numpy(),
        )
        describe_series(
            "only motif B: deletions AFTER  motif B",
            only_b["del_after_B"].to_numpy(),
        )

    if len(both_valid):
        describe_series(
            "both motifs: deletions BETWEEN motif A and motif B",
            both_valid["del_between"].dropna().to_numpy(),
        )

    # ---------------- Save CSVs ----------------
    counts.to_csv(OUT_DIR / "label_counts.csv", index=False)

    if len(only_a):
        only_a[
            [
                "id",
                "position_of_motif_A",
                "total_deletions_per_sequence",
                "del_before_A",
                "del_after_A",
            ]
        ].to_csv(OUT_DIR / "A_only_stats.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "id",
                "position_of_motif_A",
                "total_deletions_per_sequence",
                "del_before_A",
                "del_after_A",
            ]
        ).to_csv(OUT_DIR / "A_only_stats.csv", index=False)

    if len(only_b):
        only_b[
            [
                "id",
                "position_of_motif_B",
                "total_deletions_per_sequence",
                "del_before_B",
                "del_after_B",
            ]
        ].to_csv(OUT_DIR / "B_only_stats.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "id",
                "position_of_motif_B",
                "total_deletions_per_sequence",
                "del_before_B",
                "del_after_B",
            ]
        ).to_csv(OUT_DIR / "B_only_stats.csv", index=False)

    if len(both_valid):
        both_valid[
            [
                "id",
                "position_of_motif_A",
                "position_of_motif_B",
                "gaplength",
                "total_deletions_per_sequence",
                "del_between",
            ]
        ].to_csv(OUT_DIR / "both_stats.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "id",
                "position_of_motif_A",
                "position_of_motif_B",
                "gaplength",
                "total_deletions_per_sequence",
                "del_between",
            ]
        ).to_csv(OUT_DIR / "both_stats.csv", index=False)

    # ---------------- 4 plots in one figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: label counts (with friendly x-axis labels)
    ax = axes[0, 0]
    plot_counts = counts.copy()
    plot_counts["label_pretty"] = plot_counts["label"].map(LABEL_PRETTY).fillna(
        plot_counts["label"]
    )
    ax.bar(
        plot_counts["label_pretty"],
        plot_counts["count"],
        color=CBF_COLORS["blue"],
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_title("Label counts (motif classes)")
    ax.set_xlabel("Motif class")
    ax.set_ylabel("Number of sequences")
    ax.tick_params(axis="x", rotation=15)

    # Plot 2: motif start positions for single-motif sequences
    ax = axes[0, 1]
    L = int(df["sequence_length"].mode().iloc[0])
    # Use coarse bins for readability; details are in the CSVs
    edges = np.arange(0, L + 5, 5)
    centers = (edges[:-1] + edges[1:]) / 2.0

    a_counts = (
        np.histogram(only_a["position_of_motif_A"].astype(int), bins=edges)[0]
        if len(only_a)
        else np.zeros_like(centers, dtype=int)
    )
    b_counts = (
        np.histogram(only_b["position_of_motif_B"].astype(int), bins=edges)[0]
        if len(only_b)
        else np.zeros_like(centers, dtype=int)
    )

    bar_width = 2.0
    ax.bar(
        centers - bar_width / 2,
        a_counts,
        width=bar_width,
        color=CBF_COLORS["blue"],
        edgecolor="white",
        label="only motif A",
    )
    ax.bar(
        centers + bar_width / 2,
        b_counts,
        width=bar_width,
        color=CBF_COLORS["orange"],
        edgecolor="white",
        label="only motif B",
    )
    ax.set_title("Motif start positions (single-motif sequences)")
    ax.set_xlabel("Motif start position (0-based)")
    ax.set_ylabel("Count")
    ax.legend()

    # Plot 3: total deletions per sequence by class
    ax = axes[1, 0]
    label_order = ["both", "A_only", "B_only", "no_motif"]
    legend_names = {
        "both": "both motifs",
        "A_only": "only motif A",
        "B_only": "only motif B",
        "no_motif": "none",
    }
    colors_by_label = {
        "both": CBF_COLORS["green"],
        "A_only": CBF_COLORS["blue"],
        "B_only": CBF_COLORS["orange"],
        "no_motif": CBF_COLORS["gray"],
    }

    max_del = int(df["total_deletions_per_sequence"].max())
    x = np.arange(0, max_del + 1).astype(float)

    counts_by_label: Dict[str, np.ndarray] = {}
    for lab in label_order:
        subset = df[df["label"].eq(lab)][
            "total_deletions_per_sequence"
        ].to_numpy()
        counts_by_label[lab] = (
            np.bincount(subset, minlength=max_del + 1)
            if subset.size
            else np.zeros(max_del + 1, dtype=int)
        )

    n = len(label_order)
    width = 0.18
    offsets = (np.arange(n) - (n - 1) / 2.0) * width
    for i, lab in enumerate(label_order):
        ax.bar(
            x + offsets[i],
            counts_by_label[lab],
            width=width * 0.95,
            color=colors_by_label[lab],
            edgecolor="white",
            linewidth=0.5,
            label=legend_names[lab],
        )

    ax.set_title("Total deletions per sequence by motif class")
    ax.set_xlabel("Total deletions per sequence (count of '-')")
    ax.set_ylabel("Count")
    ax.set_xticks(
        np.arange(0, max_del + 1, 1)
        if max_del <= 25
        else np.arange(0, max_del + 1, 2)
    )
    ax.legend()

    # Plot 4: deletions between motif A and motif B (both class)
    ax = axes[1, 1]
    if len(both_valid):
        between = both_valid["del_between"].dropna().astype(int).to_numpy()
        max_between = int(between.max()) if between.size else 0
        xb = np.arange(0, max_between + 1)
        yb = (
            np.bincount(between, minlength=max_between + 1)
            if between.size
            else np.zeros(max_between + 1, dtype=int)
        )
        ax.bar(
            xb, yb, color=CBF_COLORS["red"], edgecolor="white", linewidth=0.6
        )
        ax.set_title("Deletions between motif A and motif B")
        ax.set_xlabel("Deletions between motifs (end of motif A → start of motif B)")
        ax.set_ylabel("Count")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid 'both' sequences\n(A before B)",
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    fig.suptitle(
        "Motif and deletion statistics of our whole dataset (4 key plots)",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()

    out_png = PLOT_DIR / "summary_4plots.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.show()

    # Final paths printout
    print(f"\nSaved figure: {out_png}")
    print(f"Saved CSVs in: {OUT_DIR.resolve()}")
    print(" - label_counts.csv")
    print(" - A_only_stats.csv")
    print(" - B_only_stats.csv")
    print(" - both_stats.csv")


if __name__ == "__main__":
    main()

