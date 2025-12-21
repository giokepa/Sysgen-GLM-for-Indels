import os
import random
import csv

from glm_model_new import GLMModel
from sequence_plotter import SequenceLogoPlotter

PROJECT_ROOT = "/Users/amelielaura/Documents/Project6"

FASTA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta"
)
MODEL_OUT = os.path.join(PROJECT_ROOT, "model_out")

OUT_SCORES = os.path.join(PROJECT_ROOT, "outputs", "scores", "scores.csv")
OUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

def make_deletion_alt(ref_seq: str, deletion_rate: float = 0.10) -> str:
    s = list(ref_seq)
    for i in range(len(s)):
        if random.random() < deletion_rate:
            s[i] = "-"
    return "".join(s)

def main():
    os.makedirs(os.path.dirname(OUT_SCORES), exist_ok=True)
    os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

    glm = GLMModel(model_path=MODEL_OUT, fasta_file=FASTA_PATH, max_seq_length=256)
    plotter = SequenceLogoPlotter()

    headers = glm.dataset.headers
    seqs = glm.dataset.seqs

    # Use clean sequences (no '-') as reference
    clean_idx = [i for i, s in enumerate(seqs) if "-" not in s]
    if not clean_idx:
        raise RuntimeError("No clean sequences found. Need some sequences without '-' as reference.")

    random.shuffle(clean_idx)

    n_eval = min(200, len(clean_idx))   # increase later
    n_plots = 5                         # just a few for testing

    rows = []
    for k in range(n_eval):
        i = clean_idx[k]
        header = headers[i]
        ref = seqs[i]
        alt = make_deletion_alt(ref, deletion_rate=0.10)

        # Method 1
        delta_fast = glm.delta_likelihood_fast(ref, alt)["delta"]

        # Method 2
        infl = glm.influence_probability_shift(ref, alt)["influence_score"]

        rows.append([i, header, delta_fast, infl])

        if k < n_plots:
            probs_ref = glm.get_full_reconstruction_probs(ref)
            probs_alt = glm.get_full_reconstruction_probs(alt)

            plotter.plot_to_file(header, ref, probs_ref, os.path.join(OUT_PLOTS_DIR, f"ref_{i}.png"))
            plotter.plot_to_file(header, alt, probs_alt, os.path.join(OUT_PLOTS_DIR, f"alt_{i}.png"))

    with open(OUT_SCORES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "header", "delta_fast", "influence_score"])
        w.writerows(rows)

    print("Done.")
    print("Scores:", OUT_SCORES)
    print("Plots:", OUT_PLOTS_DIR)

if __name__ == "__main__":
    main()
