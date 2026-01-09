import os
import csv
import random
from typing import List, Tuple

from glm_model_new import GLMModel
from dependency_map import DependencyMapGenerator

PROJECT_ROOT = "/Users/amelielaura/Documents/Project6"

FASTA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta"
)
MODEL_OUT = os.path.join(PROJECT_ROOT, "model_out")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "dependency_maps")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")


# -------------------------
# Helpers
# -------------------------
def get_label_from_header(header: str) -> str:
    if "label=" not in header:
        return "unknown"
    # header might look like: seq0001|label=A_only|posAmotif=...
    parts = header.split("|")
    for p in parts:
        if p.startswith("label="):
            return p.split("=", 1)[1].strip()
    return "unknown"


def save_input_sequence(out_dir: str, idx: int, header: str, ref_seq: str) -> str:
    """Save the exact sequence used for the dependency map."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seq_{idx}__input.txt")
    with open(path, "w") as f:
        f.write(f"INDEX: {idx}\n")
        f.write(f"HEADER: {header}\n")
        f.write(f"LABEL: {get_label_from_header(header)}\n")
        f.write(f"LENGTH: {len(ref_seq)}\n\n")
        f.write(ref_seq + "\n")
    return path


def choose_indices_by_label(
    headers: List[str],
    seqs: List[str],
    wanted_label: str,
    n: int,
    rng: random.Random
) -> List[int]:
    """
    Pick n indices where header label==wanted_label AND sequence is clean (no '-').
    """
    candidates = [
        i for i, (h, s) in enumerate(zip(headers, seqs))
        if ("-" not in s) and (get_label_from_header(h) == wanted_label)
    ]
    rng.shuffle(candidates)
    return candidates[: min(n, len(candidates))]


def choose_clean_fillup(
    headers: List[str],
    seqs: List[str],
    already_chosen: set,
    n: int,
    rng: random.Random
) -> List[int]:
    """
    Fill up with any clean sequences (no '-') not already chosen.
    """
    candidates = [
        i for i, s in enumerate(seqs)
        if ("-" not in s) and (i not in already_chosen)
    ]
    rng.shuffle(candidates)
    return candidates[: min(n, len(candidates))]


def main(
    n_total: int = 50,
    n_a_only: int = 25,
    n_b_only: int = 25,
    metric: str = "tv",
    seed: int = 727,
    batch_size: int = 64,
    fill_up_if_missing: bool = True,
):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(seed)

    glm = GLMModel(model_path=MODEL_OUT, fasta_file=FASTA_PATH, max_seq_length=256)
    dep = DependencyMapGenerator(glm)

    headers = glm.dataset.headers
    seqs = glm.dataset.seqs

    # Pick A_only and B_only
    chosen_a = choose_indices_by_label(headers, seqs, "A_only", n_a_only, rng)
    chosen_b = choose_indices_by_label(headers, seqs, "B_only", n_b_only, rng)

    chosen = chosen_a + chosen_b
    chosen_set = set(chosen)

    # If missing: fill up with any clean sequences
    missing = n_total - len(chosen)
    if missing > 0 and fill_up_if_missing:
        extra = choose_clean_fillup(headers, seqs, chosen_set, missing, rng)
        chosen += extra
        chosen_set.update(extra)

    chosen = chosen[: min(n_total, len(chosen))]

    if len(chosen) == 0:
        raise RuntimeError(
            "No sequences selected. Check that your FASTA headers contain label=A_only / label=B_only "
            "and that you have clean sequences without '-'."
        )

    # Create label subfolders
    out_a = os.path.join(OUT_DIR, "A_only")
    out_b = os.path.join(OUT_DIR, "B_only")
    out_other = os.path.join(OUT_DIR, "other")

    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)
    os.makedirs(out_other, exist_ok=True)

    rows = []
    for k, i in enumerate(chosen, start=1):
        header = headers[i]
        ref_seq = seqs[i]
        label = get_label_from_header(header)

        # Put into label-specific directory
        if label == "A_only":
            out_here = out_a
        elif label == "B_only":
            out_here = out_b
        else:
            out_here = out_other

        print(f"[{k}/{len(chosen)}] idx={i} label={label} len={len(ref_seq)}")

        # Compute dependency map
        result = dep.compute_dependency_map(
            ref_seq=ref_seq,
            metric=metric,
            batch_size=batch_size,
            set_diagonal_nan=True,
        )

        prefix = f"seq_{i}__dep"
        outs = dep.save_outputs(result, out_dir=out_here, prefix=prefix, make_heatmap=True)

        # Save input sequence next to the map
        input_txt = save_input_sequence(out_here, i, header, ref_seq)

        rows.append({
            "index": i,
            "label": label,
            "header": header,
            "length": len(ref_seq),
            "metric": metric,
            "out_dir": out_here,
            "input_txt": input_txt,
            "npy_path": outs.get("npy", ""),
            "png_path": outs.get("png", ""),
        })

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nDone")
    print("Output root:", OUT_DIR)
    print("Manifest:", MANIFEST)
    print(f"Counts: A_only={sum(r['label']=='A_only' for r in rows)}, "
          f"B_only={sum(r['label']=='B_only' for r in rows)}, "
          f"other={sum(r['label'] not in ['A_only','B_only'] for r in rows)}")


if __name__ == "__main__":
    main(
        n_total=50,
        n_a_only=25,
        n_b_only=25,
        metric="tv",
        seed=727,
        batch_size=64,
        fill_up_if_missing=True,
    )
