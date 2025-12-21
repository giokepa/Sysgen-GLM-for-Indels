import os
import csv
import random
import argparse

from glm_model_new import GLMModel
from sequence_plotter import SequenceLogoPlotter


def make_deletion_alt(ref_seq: str, deletion_rate: float = 0.10) -> str:
    s = list(ref_seq)
    for i in range(len(s)):
        if random.random() < deletion_rate:
            s[i] = "-"
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_plots", type=int, default=10)
    ap.add_argument("--deletion_rate", type=float, default=0.10)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    out_scores = os.path.join(args.out_dir, "scores", "scores.csv")
    out_plots = os.path.join(args.out_dir, "plots")
    os.makedirs(os.path.dirname(out_scores), exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    glm = GLMModel(model_path=args.model_out, fasta_file=args.fasta, max_seq_length=args.max_len)

    headers = glm.dataset.headers
    seqs = glm.dataset.seqs

    # clean sequences = no '-' inside
    clean_idx = [i for i, s in enumerate(seqs) if "-" not in s]
    if not clean_idx:
        raise RuntimeError("No clean sequences found. Need at least some refs without '-'.")

    random.shuffle(clean_idx)
    n_eval = min(args.n_eval, len(clean_idx))

    plotter = SequenceLogoPlotter()

    with open(out_scores, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "header", "delta_fast", "influence_score"])

        for k in range(n_eval):
            i = clean_idx[k]
            header = headers[i]
            ref = seqs[i]
            alt = make_deletion_alt(ref, deletion_rate=args.deletion_rate)

            delta_fast = glm.delta_likelihood_fast(ref, alt)["delta"]
            influence = glm.influence_probability_shift(ref, alt, metric="max_abs_logodds", reduce="mean")["influence_score"]

            w.writerow([i, header, delta_fast, influence])

            if k < args.n_plots:
                probs_ref = glm.get_full_reconstruction_probs(ref)
                probs_alt = glm.get_full_reconstruction_probs(alt)
                plotter.plot_to_file(header, ref, probs_ref, os.path.join(out_plots, f"ref_{i}.png"))
                plotter.plot_to_file(header, alt, probs_alt, os.path.join(out_plots, f"alt_{i}.png"))

    print("Done.")
    print("Scores:", out_scores)
    print("Plots :", out_plots)


if __name__ == "__main__":
    main()
