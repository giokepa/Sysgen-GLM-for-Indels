import os
import sys
import random
import csv

PROJECT_ROOT = "/Users/amelielaura/Documents/Project6"
sys.path.insert(0, PROJECT_ROOT)

from lib.glm_model_new import GLMModel
from lib.sequence_plotter import SequenceLogoPlotter


def make_deletion_alt(ref_seq: str, deletion_rate: float = 0.10) -> str:
    s = list(ref_seq)
    for i in range(len(s)):
        if random.random() < deletion_rate:
            s[i] = "-"
    return "".join(s)


def main():
    fasta_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "augumented_sequence_size100000_length150_deletions0.1_nodeletionseq0.25.fasta"
    )
    model_out = os.path.join(PROJECT_ROOT, "model_out")

    out_scores = os.path.join(PROJECT_ROOT, "outputs", "scores.csv")
    out_plots = os.path.join(PROJECT_ROOT, "outputs", "plots")

    os.makedirs(os.path.dirname(out_scores), exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    glm = GLMModel(model_path=model_out, fasta_file=fasta_path, max_seq_length=256)

    headers = glm.dataset.headers
    seqs = glm.dataset.seqs

    clean_idx = [i for i, s in enumerate(seqs) if "-" not in s]
    random.shuffle(clean_idx)

    n_eval = min(200, len(clean_idx))  # raise later
    n_plots = 5

    plotter = SequenceLogoPlotter()

    with open(out_scores, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "header", "delta_fast", "influence_score"])

        for k in range(n_eval):
            i = clean_idx[k]
            header = headers[i]
            ref = seqs[i]
            alt = make_deletion_alt(ref, deletion_rate=0.10)

            d = glm.delta_likelihood_fast(ref, alt)["delta"]
            infl = glm.influence_probability_shift(ref, alt)["influence_score"]

            w.writerow([i, header, d, infl])

            if k < n_plots:
                probs_ref = glm.get_full_reconstruction_probs(ref)
                probs_alt = glm.get_full_reconstruction_probs(alt)

                plotter.plot_to_file(header, ref, probs_ref, os.path.join(out_plots, f"ref_{i}.png"))
                plotter.plot_to_file(header, alt, probs_alt, os.path.join(out_plots, f"alt_{i}.png"))

    print("Done.")
    print("Scores:", out_scores)
    print("Plots :", out_plots)


if __name__ == "__main__":
    main()
