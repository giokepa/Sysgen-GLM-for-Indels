from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
#FASTA = Path(
    #"/Users/amelielaura/Documents/Project6/new_augumented_sequence_size10000_length150_deletions0.2_nodeletionseq0.05.fasta"
#)

FASTA = Path(
    "/Users/amelielaura/Documents/Project6/data/new_augumented_sequence_size10000_length150_deletions0_nodeletionseq0.05.fasta"
)

OUT_DIR = Path("/Users/amelielaura/Documents/Project6/data/out_baseline")
PLOT_DIR = Path("/Users/amelielaura/Documents/Project6/data/plotresults_baseline")

MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
LEN_A = len(MOTIF_A)
LEN_B = len(MOTIF_B)

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

TRANS_ORDER = ["A->B", "B->A", "A->A", "B->B"]
TRANS_COLOR: Dict[str, str] = {
    "A->B": "#e41a1c",
    "B->A": "#377eb8",
    "A->A": "#999999",
    "B->B": "#ff7f00",
}


# -------------------------------------------------------------------
# Plot style
# -------------------------------------------------------------------
def friendly_style() -> None:
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


friendly_style()


# -------------------------------------------------------------------
# FASTA parsing
# -------------------------------------------------------------------
def parse_pos_list(value: Optional[str]) -> List[int]:
    if value in (None, "None", ""):
        return []
    positions: List[int] = []
    string_value = str(value)
    parts = string_value.split(",")
    for raw_entry in parts:
        cleaned_entry = raw_entry.strip()
        if cleaned_entry:
            positions.append(int(cleaned_entry))
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


def count_del_between(sequence: str, start_pos: int, start_len: int, end_pos: int) -> float:
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
    for position_a in posA:
        events.append((position_a, "A", LEN_A))
    for position_b in posB:
        events.append((position_b, "B", LEN_B))
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

    dataframe = read_fasta_with_metadata(FASTA)
    if dataframe.empty:
        raise ValueError("No sequences found in FASTA.")

    # basic derived columns
    dataframe["sequence_length"] = dataframe["seq"].str.len()
    dataframe["total_deletions"] = dataframe["seq"].str.count("-").astype(int)

    # counts of motif A and B per sequence
    a_counts: List[int] = []
    b_counts: List[int] = []
    for row in dataframe.itertuples(index=False):
        a_counts.append(len(row.posA_list))
        b_counts.append(len(row.posB_list))
    dataframe["A_count"] = a_counts
    dataframe["B_count"] = b_counts

    # 5‑class labels
    class5_list: List[str] = []
    for a_value, b_value in zip(dataframe["A_count"], dataframe["B_count"]):
        class5_list.append(classify_sequence(int(a_value), int(b_value)))
    dataframe["class5"] = class5_list

    # class counts
    class_counts = (
        dataframe["class5"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename_axis("class5")
        .reset_index(name="count")
    )
    class_counts["percent"] = (class_counts["count"] / len(dataframe) * 100).round(2)
    class_counts.to_csv(OUT_DIR / "class5_counts.csv", index=False)

    # both_exact_once: deletions between the two motifs
    both_exact_once_df = dataframe[dataframe["class5"] == "both_exact_once"].copy()
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
            deletions_between = count_del_between(row.seq, start_pos, start_len, end_pos)
            if not np.isnan(deletions_between):
                deletions_between_list.append(int(deletions_between))
        if deletions_between_list:
            both_between = np.array(deletions_between_list, dtype=int)

    # multimotifs: transitions and deletions between motifs
    multimotifs_df = dataframe[dataframe["class5"] == "multimotifs"].copy()
    transition_rows: List[Dict[str, Any]] = []
    if not multimotifs_df.empty:
        for row in multimotifs_df.itertuples(index=False):
            events = build_events(row.posA_list, row.posB_list)
            sequence_string = row.seq
            sequence_id = row.id
            last_index = len(events) - 1
            event_index = 0
            while event_index < last_index:
                position_1, type_1, length_1 = events[event_index]
                position_2, type_2, length_2_unused = events[event_index + 1]
                deletions_between = count_del_between(
                    sequence_string,
                    int(position_1),
                    int(length_1),
                    int(position_2),
                )
                transition_rows.append(
                    {
                        "id": sequence_id,
                        "transition": f"{type_1}->{type_2}",
                        "del_between": deletions_between,
                    }
                )
                event_index += 1

    transitions_dataframe = pd.DataFrame(transition_rows)
    transitions_dataframe.to_csv(OUT_DIR / "multimotif_transitions.csv", index=False)

    # ---- prepare data for plots ----
    max_total_deletions = int(dataframe["total_deletions"].max())

    main_length = int(dataframe["sequence_length"].mode().iloc[0])
    bin_edges = np.arange(0, main_length + 1, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    histA_by_class: Dict[str, np.ndarray] = {}
    histB_by_class: Dict[str, np.ndarray] = {}
    for class_name in CLASS_ORDER:
        a_positions_all = flatten_lists(
            dataframe.loc[dataframe["class5"] == class_name, "posA_list"]
        )
        b_positions_all = flatten_lists(
            dataframe.loc[dataframe["class5"] == class_name, "posB_list"]
        )

        if a_positions_all.size > 0:
            histA_by_class[class_name] = np.histogram(a_positions_all, bins=bin_edges)[0]
        else:
            histA_by_class[class_name] = np.zeros(len(bin_centers), dtype=int)

        if b_positions_all.size > 0:
            histB_by_class[class_name] = np.histogram(b_positions_all, bins=bin_edges)[0]
        else:
            histB_by_class[class_name] = np.zeros(len(bin_centers), dtype=int)

    transition_counts: Dict[str, int] = {}
    transition_values: Dict[str, np.ndarray] = {}
    for transition_name in TRANS_ORDER:
        transition_counts[transition_name] = 0
        transition_values[transition_name] = np.array([], dtype=int)

    if not transitions_dataframe.empty:
        for transition_name in TRANS_ORDER:
            subset_series = transitions_dataframe.loc[
                transitions_dataframe["transition"] == transition_name, "del_between"
            ].dropna()
            transition_counts[transition_name] = int(subset_series.shape[0])
            if subset_series.shape[0] > 0:
                transition_values[transition_name] = subset_series.astype(int).to_numpy()

    # -------------------------------------------------------------------
    # CASE 1: no deletions -> only motif‑based plots
    # -------------------------------------------------------------------
    if max_total_deletions == 0:
        figure, axes_array = plt.subplots(1, 2, figsize=(12, 5))
        axis_classes = axes_array[0]
        axis_positions = axes_array[1]

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
        axis_classes.bar(
            pretty_labels,
            count_values,
            color=color_values,
            edgecolor="#333333",
            linewidth=0.6,
        )
        axis_classes.set_title("Class counts (5-class scheme)")
        axis_classes.set_xlabel("Class")
        axis_classes.set_ylabel("Number of sequences")
        axis_classes.tick_params(axis="x", rotation=15)

        for class_name in CLASS_ORDER:
            hist_a = histA_by_class[class_name]
            hist_b = histB_by_class[class_name]
            if hist_a.sum() > 0:
                axis_positions.plot(
                    bin_centers, hist_a, label=f"A | {CLASS_PRETTY[class_name]}"
                )
            if hist_b.sum() > 0:
                axis_positions.plot(
                    bin_centers,
                    hist_b,
                    linestyle="--",
                    label=f"B | {CLASS_PRETTY[class_name]}",
                )
        axis_positions.set_title("Motif start positions (A solid, B dashed)")
        axis_positions.set_xlabel("Start position (0-based, binned by 5)")
        axis_positions.set_ylabel("Count")
        axis_positions.set_xlim(0, main_length)
        axis_positions.legend(ncol=2)

        figure.suptitle("Motif statistics (no deletions present)", y=0.98, fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.95])

        output_png = PLOT_DIR / "summary_motif_only_no_deletions.png"
        figure.savefig(output_png, dpi=220, bbox_inches="tight")
        plt.show()
        return

    # -------------------------------------------------------------------
    # CASE 2: deletions present -> 6 plots
    # -------------------------------------------------------------------
    figure, axes_array = plt.subplots(3, 2, figsize=(16, 12))
    axes_array = axes_array.reshape(3, 2)

    # (1) class counts
    axis_classes = axes_array[0, 0]
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
    axis_classes.bar(
        pretty_labels,
        count_values,
        color=color_values,
        edgecolor="#333333",
        linewidth=0.6,
    )
    axis_classes.set_title("1) Class counts (5-class scheme)")
    axis_classes.set_xlabel("Class")
    axis_classes.set_ylabel("Number of sequences")
    axis_classes.tick_params(axis="x", rotation=15)

    # (2) motif positions
    axis_positions = axes_array[0, 1]
    for class_name in CLASS_ORDER:
        hist_a = histA_by_class[class_name]
        hist_b = histB_by_class[class_name]
        if hist_a.sum() > 0:
            axis_positions.plot(
                bin_centers, hist_a, label=f"A | {CLASS_PRETTY[class_name]}"
            )
        if hist_b.sum() > 0:
            axis_positions.plot(
                bin_centers,
                hist_b,
                linestyle="--",
                label=f"B | {CLASS_PRETTY[class_name]}",
            )
    axis_positions.set_title("2) Motif start positions (A solid, B dashed)")
    axis_positions.set_xlabel("Start position (0-based, binned by 5)")
    axis_positions.set_ylabel("Count")
    axis_positions.set_xlim(0, main_length)
    axis_positions.legend(ncol=2)

    # (3) total deletions per sequence by class
    axis_total_del = axes_array[1, 0]
    x_values = np.arange(0, max_total_deletions + 1).astype(float)
    bar_width = 0.16
    offsets: List[float] = []
    offset_index = 0
    while offset_index < len(CLASS_ORDER):
        value = (offset_index - (len(CLASS_ORDER) - 1) / 2.0) * bar_width
        offsets.append(value)
        offset_index += 1

    any_nonzero_total = False
    for class_index, class_name in enumerate(CLASS_ORDER):
        subset_values = dataframe.loc[
            dataframe["class5"] == class_name, "total_deletions"
        ].to_numpy()
        if subset_values.size > 0:
            counts = np.bincount(subset_values, minlength=max_total_deletions + 1)
        else:
            counts = np.zeros(max_total_deletions + 1, dtype=int)
        if counts[1:].sum() > 0:
            any_nonzero_total = True
        axis_total_del.bar(
            x_values + offsets[class_index],
            counts,
            width=bar_width * 0.95,
            color=CLASS_COLOR[class_name],
            edgecolor="white",
            linewidth=0.5,
            label=CLASS_PRETTY[class_name],
        )
    if any_nonzero_total:
        axis_total_del.set_title("3) Total deletions per sequence by class")
        axis_total_del.set_xlabel("Total deletions (count of '-')")
        axis_total_del.set_ylabel("Count")
        if max_total_deletions <= 25:
            axis_total_del.set_xticks(np.arange(0, max_total_deletions + 1, 1))
        else:
            axis_total_del.set_xticks(np.arange(0, max_total_deletions + 1, 2))
        axis_total_del.legend()
    else:
        axis_total_del.set_axis_off()

    # (4) both_exact_once: deletions between motifs (clean PMF)
    axis_between_both = axes_array[1, 1]
    if both_between.size > 0:
        max_between = int(both_between.max())
        x_between = np.arange(0, max_between + 1)
        counts_between = np.bincount(both_between, minlength=max_between + 1)
        total_pairs = float(both_between.size)
        prob_between = counts_between / total_pairs

        axis_between_both.bar(
            x_between,
            prob_between,
            color=CLASS_COLOR["both_exact_once"],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
        )

        mean_between = float(both_between.mean())
        median_between = float(np.median(both_between))
        axis_between_both.axvline(
            mean_between,
            color="#555555",
            linestyle=":",
            linewidth=1.1,
        )

        axis_between_both.set_title("4) both_exact_once: deletions between motifs")
        axis_between_both.set_xlabel("Number of deletions between motifs")
        axis_between_both.set_ylabel("Fraction of motif pairs")

        if max_between <= 25:
            axis_between_both.set_xticks(x_between)
        else:
            axis_between_both.set_xticks(np.arange(0, max_between + 1, 2))

        axis_between_both.text(
            0.98,
            0.95,
            f"mean = {mean_between:.2f}\nmedian = {median_between:.2f}",
            ha="right",
            va="top",
            transform=axis_between_both.transAxes,
            fontsize=8,
        )
    else:
        axis_between_both.set_axis_off()

    # (5) multimotifs: transition counts (compact labels)
    axis_trans_counts = axes_array[2, 0]
    nonzero_transitions: List[str] = []
    nonzero_heights: List[int] = []
    for transition_name in TRANS_ORDER:
        count_value = transition_counts[transition_name]
        if count_value > 0:
            nonzero_transitions.append(transition_name)
            nonzero_heights.append(count_value)

    if nonzero_transitions:
        total_transitions = float(sum(nonzero_heights))
        x_positions = np.arange(len(nonzero_transitions))
        bar_list = axis_trans_counts.bar(
            x_positions,
            nonzero_heights,
            color=[TRANS_COLOR[t] for t in nonzero_transitions],
            edgecolor="white",
            linewidth=0.6,
        )
        axis_trans_counts.set_title("5) multimotifs: transition counts")
        axis_trans_counts.set_xlabel("Transition (consecutive motifs)")
        axis_trans_counts.set_ylabel("Number of transitions")
        axis_trans_counts.set_xticks(x_positions)
        axis_trans_counts.set_xticklabels(nonzero_transitions)

        for index, bar in enumerate(bar_list):
            height = nonzero_heights[index]
            fraction = height / total_transitions * 100.0
            x_center = bar.get_x() + bar.get_width() / 2.0
            axis_trans_counts.text(
                x_center,
                height,
                f"{height}\n({fraction:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        axis_trans_counts.set_axis_off()

    # (6) multimotifs: deletions between consecutive motifs
    axis_trans_del = axes_array[2, 1]
    total_transition_values = 0
    for value_array in transition_values.values():
        total_transition_values += value_array.size

    if total_transition_values > 0:
        global_max = 0
        for transition_name in TRANS_ORDER:
            array_for_transition = transition_values[transition_name]
            if array_for_transition.size > 0:
                local_max = int(array_for_transition.max())
                if local_max > global_max:
                    global_max = local_max

        if global_max > 0:
            x_bins = np.arange(0, global_max + 1)
            for transition_name in TRANS_ORDER:
                values_for_transition = transition_values[transition_name]
                if values_for_transition.size == 0:
                    continue

                counts_for_transition = np.bincount(
                    values_for_transition, minlength=global_max + 1
                )
                total_for_transition = float(values_for_transition.size)
                fractions_for_transition = counts_for_transition / total_for_transition
                mean_for_transition = float(values_for_transition.mean())
                label_text = f"{transition_name} (mean {mean_for_transition:.2f})"

                axis_trans_del.plot(
                    x_bins,
                    fractions_for_transition,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=label_text,
                )

            axis_trans_del.set_title(
                "6) multimotifs: deletions between consecutive motifs"
            )
            axis_trans_del.set_xlabel("Deletions between consecutive motifs")
            axis_trans_del.set_ylabel("Fraction of transitions")
            if global_max <= 25:
                axis_trans_del.set_xticks(x_bins)
            axis_trans_del.legend()
        else:
            axis_trans_del.set_axis_off()
    else:
        axis_trans_del.set_axis_off()

    figure.suptitle(
        "Motif and deletion statistics — 6 plots", y=0.995, fontsize=16
    )
    figure.tight_layout(rect=[0, 0, 1, 0.98])

    output_png = PLOT_DIR / "summary_6plots_class5.png"
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
