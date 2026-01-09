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
OUT_SEQS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plot_sequences")


def make_deletion_alt(ref_seq: str, deletion_rate: float = 0.10) -> str:
   s = list(ref_seq)
   for i in range(len(s)):
       if random.random() < deletion_rate:
           s[i] = "-"
   return "".join(s)


def save_plot_sequence(index, header, ref, alt):
   os.makedirs(OUT_SEQS_DIR, exist_ok=True)
   out_txt = os.path.join(OUT_SEQS_DIR, f"seq_{index}.txt")

   with open(out_txt, "w") as f:
       f.write(f"INDEX: {index}\n")
       f.write(f"HEADER: {header}\n\n")
       f.write("REFERENCE (ref):\n")
       f.write(ref + "\n\n")
       f.write("PERTURBED (alt):\n")
       f.write(alt + "\n")

   return out_txt


def main():
   os.makedirs(os.path.dirname(OUT_SCORES), exist_ok=True)
   os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
   os.makedirs(OUT_SEQS_DIR, exist_ok=True)

   glm = GLMModel(model_path=MODEL_OUT, fasta_file=FASTA_PATH, max_seq_length=256)

   # Must have trained model
   if glm.model is None:
       raise RuntimeError("No trained model found. Run train_and_quality.py first.")

   # Ensure split exists (creates if missing)
   glm.ensure_split(seed=727, val_ratio=0.2)

   plotter = SequenceLogoPlotter()

   # ---- IMPORTANT: use VAL split (not training set) ----
   split = glm.get_split_indices()
   val_idx = split["val_idx"].tolist()

   headers = glm.dataset.headers
   seqs = glm.dataset.seqs

   # Use clean sequences (no '-') as reference candidates from VAL
   clean_val_idx = [i for i in val_idx if "-" not in seqs[i]]
   if not clean_val_idx:
       raise RuntimeError("No clean sequences found in VAL. Need sequences without '-' as reference.")

   random.shuffle(clean_val_idx)

   # overall model quality score (VAL) - appended into every row for convenience
   quality = glm.evaluate_mlm_quality_on_val(n_samples=500)
   val_loss = quality["val_mlm_loss"]
   val_ppl = quality["val_perplexity"]

   n_eval = min(200, len(clean_val_idx))
   n_plots = 5

   rows = []
   for k in range(n_eval):
       i = clean_val_idx[k]
       header = headers[i]
       ref = seqs[i]
       alt = make_deletion_alt(ref, deletion_rate=0.10)

       delta_fast = glm.delta_likelihood_fast(ref, alt)["delta"]
       infl = glm.influence_probability_shift(ref, alt)["influence_score"]

       rows.append([i, header, delta_fast, infl, val_loss, val_ppl])

       if k < n_plots:
           probs_ref = glm.get_full_reconstruction_probs(ref)
           probs_alt = glm.get_full_reconstruction_probs(alt)

           ref_png = os.path.join(OUT_PLOTS_DIR, f"ref_{i}.png")
           alt_png = os.path.join(OUT_PLOTS_DIR, f"alt_{i}.png")

           plotter.plot_to_file(header, ref, probs_ref, ref_png)
           plotter.plot_to_file(header, alt, probs_alt, alt_png)

           seq_txt = save_plot_sequence(i, header, ref, alt)

           print(f"[PLOT {k+1}/{n_plots}] saved:")
           print(f"  ref plot: {ref_png}")
           print(f"  alt plot: {alt_png}")
           print(f"  sequences: {seq_txt}")

   with open(OUT_SCORES, "w", newline="") as f:
       w = csv.writer(f)
       w.writerow(["index", "header", "delta_fast", "influence_score", "val_mlm_loss", "val_perplexity"])
       w.writerows(rows)

   print("Done.")
   print("Scores:", OUT_SCORES)
   print("Plots:", OUT_PLOTS_DIR)
   print("Plotted sequences:", OUT_SEQS_DIR)


if __name__ == "__main__":
   main()
