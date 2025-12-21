import os
import random
import csv

from glm_model_new import GLMModel
from sequence_plotter import SequenceLogoPlotter


def make_deletion_alt(ref_seq: str, deletion_rate: float = 0.10) -> str:
    s = list(ref_seq)
    for i in range(len(s)):
        if random.random() < deletion_rate:
            s[i] = "-"
    return "".join(s)


def main():
    PROJECT_ROOT = "/Users/amelielaura/Documents/Project6"

    FASTA = os.path.join(
        PROJECT_ROOT,
        "data",
        "augumented_sequence_size100000_length150_deletions0.1_nodeletionseq0.25.fasta",
    )

    MODEL_OUT = os.path.join(PROJECT_ROOT, "model_out")
    OUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "scores", "scores.csv")
    OUT_PLOTS = os.path.join(PROJECT_ROOT, "outputs", "plots")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(OUT_PLOTS, exist_ok=True)

    glm = GLMModel(model_path=MODEL_OUT, fasta_file=FASTA, max_seq_length=256)

    # If no trained model exists yet, train quickly for local testing.
    if glm.model is None:
        print("No trained model found in model_out/. Training a small test model now...")
        glm.train(epochs=1, batch_size=16, lr=2e-4)

    headers = glm.dataset.headers
    seqs = glm.dataset.seqs

    # We need clean REF sequences (no '-')
    clean_idx = [i for i, s in enumerate(seqs) if "-" not in s]
    if len(clean_idx) == 0:
        raise RuntimeError("No clean sequences found (no '-' tokens). Need some references without deletions.")

    random.shuffle(clean_idx)

    n_eval = min(200, len(clean_idx))   # increase later (e.g. 5000)
    n_plots = 10                        # plots for meeting

    plotter = SequenceLogoPlotter()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index",
            "header",
            "ref_len",
            "deletion_rate",
            "delta_fast",
            "influence_score",
            "n_queries"
        ])

        for k in range(n_eval):
            i = clean_idx[k]
            header = headers[i]
            ref = seqs[i]
            alt = make_deletion_alt(ref, deletion_rate=0.10)

            # Method 1
            delta_fast = glm.delta_likelihood_fast(ref, alt)["delta"]

            # Method 2
            infl = glm.influence_probability_shift(
                ref, alt,
                query_positions=None,
                target_window=None,
                metric="max_abs_logodds",
                reduce="mean"
            )
            infl_score = infl["influence_score"]
            n_queries = infl["n_queries"]

            w.writerow([i, header, len(ref), 0.10, delta_fast, infl_score, n_queries])

            # Plots (reconstruction logos) for a few examples
            if k < n_plots:
                probs_ref = glm.get_full_reconstruction_probs(ref)
                probs_alt = glm.get_full_reconstruction_probs(alt)

                plotter.plot_to_file(
                    header=header,
                    sequence=ref,
                    prob_matrix=probs_ref,
                    out_png=os.path.join(OUT_PLOTS, f"ref_{i}.png")
                )
                plotter.plot_to_file(
                    header=header,
                    sequence=alt,
                    prob_matrix=probs_alt,
                    out_png=os.path.join(OUT_PLOTS, f"alt_{i}.png")
                )

    print("DONE")
    print("Scores CSV:", OUT_CSV)
    print("Plots dir :", OUT_PLOTS)


if __name__ == "__main__":
    main()
