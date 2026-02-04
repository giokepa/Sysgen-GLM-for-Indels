from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
FASTA_PATH = Path(
    "/Users/amelielaura/Documents/new_augumented_sequence_size5000_length100_deletions0.2_nodeletionseq0.05.fasta"
)

MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
MOTIF_A_LENGTH = len(MOTIF_A)
MOTIF_B_LENGTH = len(MOTIF_B)


# -------------------------------------------------------------------
# FASTA parsing
# -------------------------------------------------------------------
def parse_position_list(value: Optional[str]) -> List[int]:
    if value in (None, "None", ""):
        return []
    positions: List[int] = []
    for raw_entry in str(value).split(","):
        cleaned = raw_entry.strip()
        if cleaned:
            positions.append(int(cleaned))
    return positions


def to_integer_or_none(value: Optional[str]) -> Optional[int]:
    if value in (None, "None", ""):
        return None
    first_part = str(value).split(",", 1)[0]
    return int(first_part)


def parse_header_line(header_line: str) -> Dict[str, Any]:
    stripped = header_line.lstrip(">")
    raw_fields = stripped.split("|")
    record: Dict[str, Any] = {"sequence_id": raw_fields[0]}

    for field in raw_fields[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            record[key] = value

    record["motif_A_positions"] = parse_position_list(record.get("posAmotif"))
    record["motif_B_positions"] = parse_position_list(record.get("posBmotif"))
    record["gap_length"] = to_integer_or_none(record.get("gaplength"))
    record["number_of_deletions"] = to_integer_or_none(record.get("deletions"))
    return record


def read_fasta_with_metadata(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    current_header_line: Optional[str] = None
    current_sequence_lines: List[str] = []

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header_line is not None:
                    record = parse_header_line(current_header_line)
                    record["sequence"] = "".join(current_sequence_lines)
                    records.append(record)
                current_header_line = line
                current_sequence_lines = []
            else:
                current_sequence_lines.append(line)

        if current_header_line is not None:
            record = parse_header_line(current_header_line)
            record["sequence"] = "".join(current_sequence_lines)
            records.append(record)

    return pd.DataFrame(records)


# -------------------------------------------------------------------
# Percentage of sequences by total motif counts (A + B)
# -------------------------------------------------------------------
def main() -> None:
    if not FASTA_PATH.exists():
        raise FileNotFoundError(f"FASTA file not found at: {FASTA_PATH}")

    dataframe = read_fasta_with_metadata(FASTA_PATH)
    if dataframe.empty:
        raise ValueError("No sequences found in FASTA file.")

    # Number of motif A and motif B hits per sequence
    motif_A_counts_per_sequence = dataframe["motif_A_positions"].apply(len)
    motif_B_counts_per_sequence = dataframe["motif_B_positions"].apply(len)

    # Total motif count per sequence (motif A plus motif B)
    total_motif_counts_per_sequence = (
        motif_A_counts_per_sequence + motif_B_counts_per_sequence
    )

    # Bins: 0, 1, 2, 3, 4, 5 or more motifs in total
    bin_values = [0, 1, 2, 3, 4, 5]
    bin_labels = [
        "0 motifs",
        "1 motif",
        "2 motifs",
        "3 motifs",
        "4 motifs",
        "5 or more motifs",
    ]

    # better layout
    binned_total_motif_counts = total_motif_counts_per_sequence.clip(upper=5)

    # Count frequencies per bin and reorder according to bin_values
    total_motif_bin_frequencies = binned_total_motif_counts.value_counts().reindex(
        bin_values, fill_value=0
    )

    # Convert to percentages
    number_of_sequences = len(dataframe)
    total_motif_percentages = (
        total_motif_bin_frequencies / number_of_sequences * 100
    )

    result_dataframe = pd.DataFrame(
        {
            "motif_count_category": bin_labels,
            "total_motif_percentage_of_sequences": total_motif_percentages.values,
        }
    )

    print(result_dataframe.to_string(index=False))

    # ----------------------------------------------------------------
    # Bar plot of percentages
    # ----------------------------------------------------------------
    fig, axis = plt.subplots(figsize=(8, 5))
    x_positions = range(len(bin_labels))

    bar_colors = [
        "#1b9e77",
        "#40b5ad",
        "#67c8d0",
        "#8dd3c7",
        "#b574c9",
        "#d95f02", 
    ]

    axis.bar(
        x_positions,
        result_dataframe["total_motif_percentage_of_sequences"],
        color=bar_colors[: len(bin_labels)],
        edgecolor="#333333",
        linewidth=1.1,
    )

    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(bin_labels, rotation=45, ha="right")

    axis.set_ylabel("Percentage of sequences")
    axis.set_xlabel("Total number of motifs (motif A plus motif B)")
    axis.set_title("Distribution of total motif counts per sequence")

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
