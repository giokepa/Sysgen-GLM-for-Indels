import os
import sys
import random
import csv
from fundemental_classes.model_related.glm_model_new import GLMModel
from fundemental_classes.visualization.sequence_plotter import SequenceLogoPlotter

NOTEBOOK_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, ".."))

if PROJECT_ROOT not in sys.path:
   sys.path.insert(0, PROJECT_ROOT)

print("NOTEBOOK_DIR:", NOTEBOOK_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)

FASTA_PATH = os.path.join(
   PROJECT_ROOT, "data",
   "augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta"
)

##########################################################################################################################

MODEL_OUT = os.path.join(PROJECT_ROOT, "model_out")

OUT_MODEL_QUALITY = os.path.join(PROJECT_ROOT, "outputs", "model_quality", "model_quality.csv")
OUT_SCORES = os.path.join(PROJECT_ROOT, "outputs", "scores", "scores.csv")

OUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
OUT_SEQS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plot_sequences")

print("FASTA_PATH:", FASTA_PATH)
print("MODEL_OUT:", MODEL_OUT)
#############################################################################################################################
os.makedirs(MODEL_OUT, exist_ok=True)

glm = GLMModel(
   model_path=MODEL_OUT,
   fasta_file=FASTA_PATH,
   max_seq_length=256,
   force_retrain=False,
)

glm.train(
   epochs=30,
   batch_size=16,
   lr=2e-4,
   seed=727,
   val_ratio=0.2,
   save_split=True
)
#%%
os.makedirs(os.path.dirname(OUT_MODEL_QUALITY), exist_ok=True)

q = glm.evaluate_mlm_quality_on_val(n_samples=500, mlm_probability=0.15, seed=0)
print("VAL MLM quality:", q)

with open(OUT_MODEL_QUALITY, "w", newline="") as f:
   w = csv.writer(f)
   w.writerow(["val_mlm_loss", "val_perplexity", "n_samples"])
   w.writerow([q["val_mlm_loss"], q["val_perplexity"], q["n_samples"]])

print("Saved model quality CSV:", OUT_MODEL_QUALITY)
##########################################################################################################
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

################################################################################################################################
os.makedirs(os.path.dirname(OUT_SCORES), exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
os.makedirs(OUT_SEQS_DIR, exist_ok=True)

plotter = SequenceLogoPlotter()

# --- IMPORTANT: evaluate on validation set only ---
split = glm.get_split_indices()
val_idx = split["val_idx"].tolist()

headers = glm.dataset.headers
seqs = glm.dataset.seqs

clean_val_idx = [i for i in val_idx if "-" not in seqs[i]]
if not clean_val_idx:
   raise RuntimeError("No clean sequences found in VAL (no '-')")

random.shuffle(clean_val_idx)

n_eval = min(200, len(clean_val_idx))
n_plots = 5

rows = []
for k in range(n_eval):
   i = clean_val_idx[k]
   header = headers[i]
   ref = seqs[i]
   alt = make_deletion_alt(ref, deletion_rate=0.10)

   delta_fast = glm.delta_likelihood_fast(ref, alt)["delta"]
   infl = glm.influence_probability_shift(
       ref, alt,
       metric="max_abs_logodds",
       reduce="mean"
   )["influence_score"]

   rows.append([i, header, delta_fast, infl, q["val_mlm_loss"], q["val_perplexity"]])

   if k < n_plots:
       probs_ref = glm.get_full_reconstruction_probs(ref)
       probs_alt = glm.get_full_reconstruction_probs(alt)

       ref_png = os.path.join(OUT_PLOTS_DIR, f"ref_{i}.png")
       alt_png = os.path.join(OUT_PLOTS_DIR, f"alt_{i}.png")

       plotter.plot_to_file(header, ref, probs_ref, ref_png)
       plotter.plot_to_file(header, alt, probs_alt, alt_png)

       seq_txt = save_plot_sequence(i, header, ref, alt)

       print(f"[PLOT {k+1}/{n_plots}] ref={ref_png} alt={alt_png} seq={seq_txt}")

with open(OUT_SCORES, "w", newline="") as f:
   w = csv.writer(f)
   w.writerow(["index", "header", "delta_fast", "influence_score", "val_mlm_loss", "val_perplexity"])
   w.writerows(rows)

print("Saved scores:", OUT_SCORES)
print("Saved plots:", OUT_PLOTS_DIR)
print("Saved plotted sequences:", OUT_SEQS_DIR)
