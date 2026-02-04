from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
FASTA = Path(
    "/Users/amelielaura/Documents/new_augumented_sequence_size5000_length100_deletions0.2_nodeletionseq0.05.fasta"
)

OUT_DIR = Path("/Users/amelielaura/Documents/Project6/data/out_baseline")
PLOT_DIR = Path("/Users/amelielaura/Documents/Project6/data/plotresults_baseline")

# Motifs and their lengths
MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
LEN_A = len(MOTIF_A)
LEN_B = len(MOTIF_B)

# 5-class label scheme
CLASS_ORDER = ["none", "only_A", "only_B", "both_exact_once", "multimotifs"]
CLASS_PRETTY: Dict[str, str] = {
    "none": "none",
    "only_A": "only motif A once",
    "only_B": "only motif B once",
    "both_exact_once": "both exactly once",
    "multimotifs": "multimotifs (>1 total)",
}
CLASS_COLOR: Dict[str, str] = {
    "none": "#999999",
    "only_A": "#377eb8",
    "only_B": "#ff7f00",
    "both_exact_once": "#4daf4a",
    "multimotifs": "#984ea3",
}

# Motif transition types in multimotif sequences
TRANS_ORDER = ["A->B", "B->A", "A->A", "B->B"]
TRANS_COLOR: Dict[str, str] = {
    "A->B": "#e41a1c",
    "B->A": "#377eb8",
    "A->A": "#999999",
    "B->B": "#ff7f00",
}

# Total motif count groups (A_count + B_count)
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


# -------------------------------------------------------------------
# Plot style
# -------------------------------------------------------------------
def set_friendly_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (16, 12),
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
# Classification and helpers
# -------------------------------------------------------------------
def classify_sequence(a_count: int, b_count: int) -> str:
    if a_count == 0 and b_count == 0:
        return "none"
    if a_count == 1 and b_count == 0:
        return "only_A"
    if a_count == 0 and b_count == 1:
        return "only_B"
    if a_count == 1 and b_count == 1:
        return "both_exact_once"
    return "multimotifs"


def total_motif_group(total_count: int) -> str:
    """
    Map total motif count (A + B) to the groups 0,1,2,3,4,≥5.
    """
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


def count_deletions_between(
    sequence: str, start_pos: int, start_len: int, end_pos: int
) -> float:
    start_index = start_pos + start_len
    end_index = end_pos
    if end_index < start_index:
        return float(np.nan)
    subseq = sequence[start_index:end_index]
    return float(subseq.count("-"))


def event_sort_key(entry: Tuple[int, str, int]) -> int:
    return entry[0]


def build_events(posA: List[int], posB: List[int]) -> List[Tuple[int, str, int]]:
    events: List[Tuple[int, str, int]] = []
    for p in posA:
        events.append((p, "A", LEN_A))
    for p in posB:
        events.append((p, "B", LEN_B))
    events.sort(key=event_sort_key)
    return events


def flatten_lists(series_of_lists: pd.Series) -> np.ndarray:
    flat: List[int] = []
    for inner_list in series_of_lists:
        flat.extend(inner_list)
    if flat:
        return np.array(flat, dtype=int)
    return np.array([], dtype=int)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA file not found at: {FASTA}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load data and basic columns
    # ---------------------------------------------------------------
    df = read_fasta_with_metadata(FASTA)
    if df.empty:
        raise ValueError("No sequences found in FASTA.")

    df["seq"] = df["seq"].astype(str)
    df["sequence_length"] = df["seq"].str.len()
    df["total_deletions"] = df["seq"].str.count("-").astype(int)

    # counts of motif A and B per sequence
    df["A_count"] = df["posA_list"].apply(len)
    df["B_count"] = df["posB_list"].apply(len)

    # total motifs (A+B) and motif-count groups
    df["total_motifs"] = df["A_count"] + df["B_count"]
    df["motif_group"] = df["total_motifs"].apply(total_motif_group)

    # 5‑class labels
    df["class5"] = [
        classify_sequence(int(a), int(b)) for a, b in zip(df["A_count"], df["B_count"])
    ]

    # class counts
    class_counts = (
        df["class5"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename_axis("class5")
        .reset_index(name="count")
    )
    class_counts["percent"] = (class_counts["count"] / len(df) * 100).round(2)
    class_counts.to_csv(OUT_DIR / "class5_counts.csv", index=False)

    # ---------------------------------------------------------------
    # both_exact_once: deletions between motifs
    # ---------------------------------------------------------------
    both_exact_once_df = df[df["class5"] == "both_exact_once"].copy()
    both_between = np.array([], dtype=int)

    if not both_exact_once_df.empty:
        deletions_between_list: List[int] = []
        for row in both_exact_once_df.itertuples(index=False):
            if not row.posA_list or not row.posB_list:
                continue

            first_a = int(row.posA_list[0])
            first_b = int(row.posB_list[0])

            if first_a <= first_b:
                start_pos = first_a
                end_pos = first_b
                start_len = LEN_A
            else:
                start_pos = first_b
                end_pos = first_a
                start_len = LEN_B

            deletions_between = count_deletions_between(
                row.seq, start_pos, start_len, end_pos
            )
            if not np.isnan(deletions_between):
                deletions_between_list.append(int(deletions_between))

        if deletions_between_list:
            both_between = np.array(deletions_between_list, dtype=int)

    # ---------------------------------------------------------------
    # multimotifs: motif transitions and deletions between them
    # ---------------------------------------------------------------
    multimotifs_df = df[df["class5"] == "multimotifs"].copy()
    transition_rows: List[Dict[str, Any]] = []

    if not multimotifs_df.empty:
        for row in multimotifs_df.itertuples(index=False):
            events = build_events(row.posA_list, row.posB_list)
            sequence_string = row.seq
            seq_id = row.id

            for idx in range(len(events) - 1):
                position_1, type_1, length_1 = events[idx]
                position_2, type_2, _ = events[idx + 1]

                deletions_between = count_deletions_between(
                    sequence_string,
                    int(position_1),
                    int(length_1),
                    int(position_2),
                )
                transition_rows.append(
                    {
                        "id": seq_id,
                        "transition": f"{type_1}->{type_2}",
                        "del_between": deletions_between,
                    }
                )

    transitions_df = pd.DataFrame(transition_rows)
    transitions_df.to_csv(OUT_DIR / "multimotif_transitions.csv", index=False)

    # ---------------------------------------------------------------
    # Histograms and transition stats
    # ---------------------------------------------------------------
    max_total_deletions = int(df["total_deletions"].max())
    main_length = int(df["sequence_length"].mode().iloc[0])
    bin_edges = np.arange(0, main_length + 1, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # motif positions by class5 (for plots using class5)
    histA_by_class: Dict[str, np.ndarray] = {}
    histB_by_class: Dict[str, np.ndarray] = {}
    for class_name in CLASS_ORDER:
        a_positions_all = flatten_lists(
            df.loc[df["class5"] == class_name, "posA_list"]
        )
        b_positions_all = flatten_lists(
            df.loc[df["class5"] == class_name, "posB_list"]
        )

        if a_positions_all.size > 0:
            histA_by_class[class_name] = np.histogram(
                a_positions_all, bins=bin_edges
            )[0]
        else:
            histA_by_class[class_name] = np.zeros(len(bin_centers), dtype=int)

        if b_positions_all.size > 0:
            histB_by_class[class_name] = np.histogram(
                b_positions_all, bins=bin_edges
            )[0]
        else:
            histB_by_class[class_name] = np.zeros(len(bin_centers), dtype=int)

    # motif positions by total motif group (for plot 2)
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

    transition_counts: Dict[str, int] = {}
    transition_values: Dict[str, np.ndarray] = {}
    for t in TRANS_ORDER:
        transition_counts[t] = 0
        transition_values[t] = np.array([], dtype=int)

    if not transitions_df.empty:
        for t in TRANS_ORDER:
            subset = transitions_df.loc[
                transitions_df["transition"] == t, "del_between"
            ].dropna()
            transition_counts[t] = int(subset.shape[0])
            if subset.shape[0] > 0:
                transition_values[t] = subset.astype(int).to_numpy()

    # ---------------------------------------------------------------
    # Case 1: no deletions → only motif-based plots
    # ---------------------------------------------------------------
    if max_total_deletions == 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_classes = axes[0]
        ax_positions = axes[1]

        # Plot 1: class counts (class5)
        pretty_labels: List[str] = []
        count_values: List[int] = []
        color_values: List[str] = []

        for class_name in CLASS_ORDER:
            pretty_labels.append(CLASS_PRETTY[class_name])
            count_value = int(
                class_counts.loc[class_counts["class5"] == class_name, "count"].iloc[0]
            )
            count_values.append(count_value)
            color_values.append(CLASS_COLOR[class_name])

        ax_classes.bar(
            pretty_labels,
            count_values,
            color=color_values,
            edgecolor="#333333",
            linewidth=0.6,
        )
        ax_classes.set_title("Class counts (5-class scheme)")
        ax_classes.set_xlabel("Class")
        ax_classes.set_ylabel("Number of sequences")
        ax_classes.tick_params(axis="x", rotation=15)

        # Plot 2: motif positions by total motif count group
        for group in MOTIFCOUNT_ORDER:
            hist_a = histA_by_group[group]
            hist_b = histB_by_group[group]

            if hist_a.sum() > 0:
                ax_positions.plot(
                    bin_centers,
                    hist_a,
                    color=MOTIFCOUNT_COLOR[group],
                    label=f"A | {MOTIFCOUNT_PRETTY[group]}",
                )
            if hist_b.sum() > 0:
                ax_positions.plot(
                    bin_centers,
                    hist_b,
                    color=MOTIFCOUNT_COLOR[group],
                    linestyle="--",
                    label=f"B | {MOTIFCOUNT_PRETTY[group]}",
                )

        ax_positions.set_title(
            "Motif start positions by total motif count\n"
            "(A solid, B dashed; groups 0,1,2,3,4,≥5)"
        )
        ax_positions.set_xlabel("Start position (0-based, binned by 5)")
        ax_positions.set_ylabel("Count")
        ax_positions.set_xlim(0, main_length)
        ax_positions.legend(ncol=2)

        fig.suptitle("Motif statistics (no deletions present)", y=0.98, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_png = PLOT_DIR / "summary_motif_only_no_deletions.png"
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.show()
        return

    # ---------------------------------------------------------------
    # Case 2: deletions present → 6 plots
    # ---------------------------------------------------------------
    fig, axes_array = plt.subplots(3, 2, figsize=(16, 12))
    axes_array = axes_array.reshape(3, 2)

    # (1) class counts by class5
    ax_classes = axes_array[0, 0]
    pretty_labels = []
    count_values = []
    color_values = []

    for class_name in CLASS_ORDER:
        pretty_labels.append(CLASS_PRETTY[class_name])
        count_value = int(
            class_counts.loc[class_counts["class5"] == class_name, "count"].iloc[0]
        )
        count_values.append(count_value)
        color_values.append(CLASS_COLOR[class_name])

    ax_classes.bar(
        pretty_labels,
        count_values,
        color=color_values,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax_classes.set_title("1) Class counts (5-class scheme)")
    ax_classes.set_xlabel("Class")
    ax_classes.set_ylabel("Number of sequences")
    ax_classes.tick_params(axis="x", rotation=15)

    # (2) motif positions by total motif count group
    ax_positions = axes_array[0, 1]
    for group in MOTIFCOUNT_ORDER:
        hist_a = histA_by_group[group]
        hist_b = histB_by_group[group]

        if hist_a.sum() > 0:
            ax_positions.plot(
                bin_centers,
                hist_a,
                color=MOTIFCOUNT_COLOR[group],
                label=f"A | {MOTIFCOUNT_PRETTY[group]}",
            )
        if hist_b.sum() > 0:
            ax_positions.plot(
                bin_centers,
                hist_b,
                color=MOTIFCOUNT_COLOR[group],
                linestyle="--",
                label=f"B | {MOTIFCOUNT_PRETTY[group]}",
            )

    ax_positions.set_title(
        "2) Motif start positions by total motifs\n"
        "(A solid, B dashed; groups 0,1,2,3,4,≥5)"
    )
    ax_positions.set_xlabel("Start position (0-based, binned by 5)")
    ax_positions.set_ylabel("Count")
    ax_positions.set_xlim(0, main_length)
    ax_positions.legend(ncol=2)

    # (3) total deletions per sequence by class5
    ax_total_del = axes_array[1, 0]
    x_vals = np.arange(0, max_total_deletions + 1).astype(float)
    bar_width = 0.16

    offsets: List[float] = []
    for idx in range(len(CLASS_ORDER)):
        offsets.append((idx - (len(CLASS_ORDER) - 1) / 2.0) * bar_width)

    any_nonzero_total = False
    for class_idx, class_name in enumerate(CLASS_ORDER):
        subset_vals = df.loc[df["class5"] == class_name, "total_deletions"].to_numpy()

        if subset_vals.size > 0:
            counts = np.bincount(subset_vals, minlength=max_total_deletions + 1)
        else:
            counts = np.zeros(max_total_deletions + 1, dtype=int)

        if counts[1:].sum() > 0:
            any_nonzero_total = True

        ax_total_del.bar(
            x_vals + offsets[class_idx],
            counts,
            width=bar_width * 0.95,
            color=CLASS_COLOR[class_name],
            edgecolor="white",
            linewidth=0.5,
            label=CLASS_PRETTY[class_name],
        )

    if any_nonzero_total:
        ax_total_del.set_title("3) Total deletions per sequence by class")
        ax_total_del.set_xlabel("Total deletions (count of '-') per sequence")
        ax_total_del.set_ylabel("Number of sequences")
        if max_total_deletions <= 25:
            ax_total_del.set_xticks(np.arange(0, max_total_deletions + 1, 1))
        else:
            ax_total_del.set_xticks(np.arange(0, max_total_deletions + 1, 2))
        ax_total_del.legend()
    else:
        ax_total_del.set_axis_off()

    # (4) both_exact_once: deletions between motifs
    ax_between_both = axes_array[1, 1]
    if both_between.size > 0:
        max_between = int(both_between.max())
        x_between = np.arange(0, max_between + 1)

        counts_between = np.bincount(both_between, minlength=max_between + 1)
        total_pairs = float(both_between.size)
        prob_between = counts_between / total_pairs

        ax_between_both.bar(
            x_between,
            prob_between,
            color=CLASS_COLOR["both_exact_once"],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
        )

        mean_between = float(both_between.mean())
        median_between = float(np.median(both_between))

        ax_between_both.axvline(
            mean_between,
            color="#555555",
            linestyle=":",
            linewidth=1.1,
        )

        ax_between_both.set_title("4) both_exact_once: deletions between motifs")
        ax_between_both.set_xlabel("Number of deletions between motifs")
        ax_between_both.set_ylabel("Fraction of motif pairs")

        if max_between <= 25:
            ax_between_both.set_xticks(x_between)
        else:
            ax_between_both.set_xticks(np.arange(0, max_between + 1, 2))

        ax_between_both.text(
            0.98,
            0.95,
            f"mean = {mean_between:.2f}\nmedian = {median_between:.2f}",
            ha="right",
            va="top",
            transform=ax_between_both.transAxes,
            fontsize=8,
        )
    else:
        ax_between_both.set_axis_off()

    # (5) transition counts
    ax_trans_counts = axes_array[2, 0]
    nonzero_transitions: List[str] = []
    nonzero_heights: List[int] = []

    for t in TRANS_ORDER:
        count_val = transition_counts[t]
        if count_val > 0:
            nonzero_transitions.append(t)
            nonzero_heights.append(count_val)

    if nonzero_transitions:
        total_transitions = float(sum(nonzero_heights))
        x_positions = np.arange(len(nonzero_transitions))

        bars = ax_trans_counts.bar(
            x_positions,
            nonzero_heights,
            color=[TRANS_COLOR[t] for t in nonzero_transitions],
            edgecolor="white",
            linewidth=0.6,
        )

        ax_trans_counts.set_title("5) multimotifs: transition counts")
        ax_trans_counts.set_xlabel("Transition (consecutive motifs)")
        ax_trans_counts.set_ylabel("Number of transitions")
        ax_trans_counts.set_xticks(x_positions)
        ax_trans_counts.set_xticklabels(nonzero_transitions)

        for i, bar in enumerate(bars):
            height = nonzero_heights[i]
            fraction = height / total_transitions * 100.0
            x_center = bar.get_x() + bar.get_width() / 2.0
            ax_trans_counts.text(
                x_center,
                height,
                f"{height}\n({fraction:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        ax_trans_counts.set_axis_off()

    # (6) deletions between consecutive motifs
    ax_trans_del = axes_array[2, 1]
    total_transition_values = sum(arr.size for arr in transition_values.values())

    if total_transition_values > 0:
        global_max = 0
        for t in TRANS_ORDER:
            arr = transition_values[t]
            if arr.size > 0:
                local_max = int(arr.max())
                if local_max > global_max:
                    global_max = local_max

        if global_max > 0:
            x_bins = np.arange(0, global_max + 1)

            for t in TRANS_ORDER:
                arr = transition_values[t]
                if arr.size == 0:
                    continue

                counts = np.bincount(arr, minlength=global_max + 1)
                total_for_t = float(arr.size)
                fractions = counts / total_for_t
                mean_for_t = float(arr.mean())

                label_text = f"{t} (mean {mean_for_t:.2f})"

                ax_trans_del.plot(
                    x_bins,
                    fractions,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=label_text,
                )

            ax_trans_del.set_title(
                "6) multimotifs: deletions between consecutive motifs"
            )
            ax_trans_del.set_xlabel("Deletions between consecutive motifs")
            ax_trans_del.set_ylabel("Fraction of transitions")
            if global_max <= 25:
                ax_trans_del.set_xticks(x_bins)
            ax_trans_del.legend()
        else:
            ax_trans_del.set_axis_off()
    else:
        ax_trans_del.set_axis_off()

    fig.suptitle("Motif and deletion statistics — 6 plots", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = PLOT_DIR / "summary_6plots_class5_and_totmotif.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
