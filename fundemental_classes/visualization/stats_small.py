from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
FASTA = Path(
    "/Users/amelielaura/Documents/new_augumented_sequence_size5000_length100_deletions0.2_nodeletionseq0.05.fasta"
)

PLOT_DIR = Path("/Users/amelielaura/Documents/Project6/data/plotresults_indel")

MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
LEN_A = len(MOTIF_A)
LEN_B = len(MOTIF_B)

MOTIFCOUNT_ORDER = ["0", "1", "2", "3", "4", "5plus"]
MOTIFCOUNT_PRETTY: Dict[str, str] = {
    "0": "0 motifs total",
    "1": "1 motif total",
    "2": "2 motifs total",
    "3": "3 motifs total",
    "4": "4 motifs total",
    "5plus": "≥5 motifs total",
}
MOTIFCOUNT_COLOR: Dict[str, str] = {
    "0": "#999999",
    "1": "#377eb8",
    "2": "#ff7f00",
    "3": "#4daf4a",
    "4": "#984ea3",
    "5plus": "#e41a1c",
}


def set_friendly_style() -> None:
    """Set a consistent, readable matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
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

set_friendly_style()

def parse_pos_list(value: Optional[str]) -> List[int]:
    if value in (None, "None", ""):
        return []
    positions: List[int] = []
    for raw_entry in str(value).split(","):
        cleaned = raw_entry.strip()
        if cleaned:
            positions.append(int(cleaned))
    return positions


def to_int_or_none(value: Optional[str]) -> Optional[int]:
    """Convert a header field to int or None, using only the first comma-separated part."""
    if value in (None, "None", ""):
        return None
    first_part = str(value).split(",", 1)[0]
    return int(first_part)


def parse_header(header_line: str) -> Dict[str, Any]:
    stripped = header_line.lstrip(">")
    raw_fields = stripped.split("|")
    record: Dict[str, Any] = {"id": raw_fields[0]}

    for field in raw_fields[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            record[key] = value

    record["posA_list"] = parse_pos_list(record.get("posAmotif"))
    record["posB_list"] = parse_pos_list(record.get("posBmotif"))
    record["gaplength"] = to_int_or_none(record.get("gaplength"))
    record["deletions_header"] = to_int_or_none(record.get("deletions"))
    return record


def read_fasta_with_metadata(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    current_header: Optional[str] = None
    current_sequence_lines: List[str] = []

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    record = parse_header(current_header)
                    record["seq"] = "".join(current_sequence_lines)
                    records.append(record)
                current_header = line
                current_sequence_lines = []
            else:
                current_sequence_lines.append(line)

        if current_header is not None:
            record = parse_header(current_header)
            record["seq"] = "".join(current_sequence_lines)
            records.append(record)

    return pd.DataFrame(records)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def flatten_lists(series_of_lists: pd.Series) -> np.ndarray:
    flat: List[int] = []
    for inner_list in series_of_lists:
        flat.extend(inner_list)
    if flat:
        return np.array(flat, dtype=int)
    return np.array([], dtype=int)


def total_motif_group(total_count: int) -> str:
    if total_count == 0:
        return "0"
    if total_count == 1:
        return "1"
    if total_count == 2:
        return "2"
    if total_count == 3:
        return "3"
    if total_count == 4:
        return "4"
    return "5plus"


# -------------------------------------------------------------------
# Main: motif-position plot grouped by total motif count
# -------------------------------------------------------------------
def main() -> None:
    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA file not found at: {FASTA}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load sequences and motif positions
    df = read_fasta_with_metadata(FASTA)
    if df.empty:
        raise ValueError("No sequences found in FASTA.")

    df["seq"] = df["seq"].astype(str)
    df["sequence_length"] = df["seq"].str.len()

    # Counts of motif A and B per sequence
    df["A_count"] = df["posA_list"].apply(len)
    df["B_count"] = df["posB_list"].apply(len)

    # Total motif count per sequence (A + B)
    df["total_motifs"] = df["A_count"] + df["B_count"]

    # Group label per sequence based on total_motifs (0,1,2,3,4,≥5)
    df["motif_group"] = df["total_motifs"].apply(total_motif_group)

    # 2) Define position bins along the sequence
    main_length = int(df["sequence_length"].mode().iloc[0])
    bin_edges = np.arange(0, main_length + 1, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # 3) Build histograms of motif start positions for each total-motif group
    histA_by_group: Dict[str, np.ndarray] = {}
    histB_by_group: Dict[str, np.ndarray] = {}

    for group in MOTIFCOUNT_ORDER:
        a_positions_all = flatten_lists(
            df.loc[df["motif_group"] == group, "posA_list"]
        )
        b_positions_all = flatten_lists(
            df.loc[df["motif_group"] == group, "posB_list"]
        )

        if a_positions_all.size > 0:
            histA_by_group[group] = np.histogram(a_positions_all, bins=bin_edges)[0]
        else:
            histA_by_group[group] = np.zeros(len(bin_centers), dtype=int)

        if b_positions_all.size > 0:
            histB_by_group[group] = np.histogram(b_positions_all, bins=bin_edges)[0]
        else:
            histB_by_group[group] = np.zeros(len(bin_centers), dtype=int)

    # 4) Create the motif-position plot
    fig, ax = plt.subplots(figsize=(10, 5))

    for group in MOTIFCOUNT_ORDER:
        hist_a = histA_by_group[group]
        hist_b = histB_by_group[group]

        if hist_a.sum() > 0:
            ax.plot(
                bin_centers,
                hist_a,
                color=MOTIFCOUNT_COLOR[group],
                label=f"A | {MOTIFCOUNT_PRETTY[group]}",
            )
        if hist_b.sum() > 0:
            ax.plot(
                bin_centers,
                hist_b,
                color=MOTIFCOUNT_COLOR[group],
                linestyle="--",
                label=f"B | {MOTIFCOUNT_PRETTY[group]}",
            )

    ax.set_title(
        "Motif start positions along sequences\n"
        "Grouped by total number of motifs per sequence\n"
        "(A: solid, B: dashed; groups: 0,1,2,3,4,≥5 motifs total)"
    )
    ax.set_xlabel("Motif start position (0-based, binned in steps of 5)")
    ax.set_ylabel(
        "Number of motif starts in bin\n"
        "(across all sequences in the same total-motif group)"
    )
    ax.set_xlim(0, main_length)
    ax.legend(ncol=2)

    fig.tight_layout()
    out_png = PLOT_DIR / "plot2_motif_start_positions_by_total_motifs.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
