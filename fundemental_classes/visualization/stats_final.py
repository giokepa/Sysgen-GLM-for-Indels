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


# -------------------------------------------------------------------
# Plot style
# -------------------------------------------------------------------
def set_friendly_style() -> None:
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


# -------------------------------------------------------------------
# FASTA parsing
# -------------------------------------------------------------------
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
# Helper
# -------------------------------------------------------------------
def flatten_lists(series_of_lists: pd.Series) -> np.ndarray:
    flat: List[int] = []
    for inner_list in series_of_lists:
        flat.extend(inner_list)
    if flat:
        return np.array(flat, dtype=int)
    return np.array([], dtype=int)


# -------------------------------------------------------------------
# Main: only start positions of motif A and motif B
# -------------------------------------------------------------------
def main() -> None:
    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA file not found at: {FASTA}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequences and motif positions
    df = read_fasta_with_metadata(FASTA)
    if df.empty:
        raise ValueError("No sequences found in FASTA.")

    df["seq"] = df["seq"].astype(str)
    df["sequence_length"] = df["seq"].str.len()

    main_length = int(df["sequence_length"].mode().iloc[0])
    bin_edges = np.arange(0, main_length + 1, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Collect all A and all B start positions across all sequences
    all_A_positions = flatten_lists(df["posA_list"])
    all_B_positions = flatten_lists(df["posB_list"])

    # Histograms of A and B start positions
    if all_A_positions.size > 0:
        histA = np.histogram(all_A_positions, bins=bin_edges)[0]
    else:
        histA = np.zeros(len(bin_centers), dtype=int)

    if all_B_positions.size > 0:
        histB = np.histogram(all_B_positions, bins=bin_edges)[0]
    else:
        histB = np.zeros(len(bin_centers), dtype=int)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    if histA.sum() > 0:
        ax.plot(
            bin_centers,
            histA,
            color="#377eb8",
            label="Motif A (start positions)",
        )
    if histB.sum() > 0:
        ax.plot(
            bin_centers,
            histB,
            color="#ff7f00",
            linestyle="--",
            label="Motif B (start positions)",
        )

    ax.set_title("Motif start positions along sequences")
    ax.set_xlabel("Motif start position")
    ax.set_ylabel("Number of motif starts across all sequences")
    ax.set_xlim(0, main_length)
    ax.legend()

    fig.tight_layout()
    out_png = PLOT_DIR / "motif_AB_start_positions_all_sequences.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
