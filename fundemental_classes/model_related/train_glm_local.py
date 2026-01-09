import os
import csv

from glm_model_new import GLMModel

PROJECT_ROOT = "/Users/amelielaura/Documents/Project6"

FASTA_PATH = os.path.join(
   PROJECT_ROOT,
   "data",
   "augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta"
)

MODEL_OUT = os.path.join(PROJECT_ROOT, "model_out")
OUT_MODEL_QUALITY = os.path.join(PROJECT_ROOT, "outputs", "model_quality", "model_quality.csv")


def main():
   os.makedirs(os.path.dirname(OUT_MODEL_QUALITY), exist_ok=True)

   glm = GLMModel(model_path=MODEL_OUT, fasta_file=FASTA_PATH, max_seq_length=256)

   # Train only if no trained model is available
   if glm.model is None:
       glm.train(epochs=30, batch_size=16, lr=2e-4, seed=727, val_ratio=0.2)
   else:
       print("Model already exists -> skipping training.")

   # Overall model quality on VAL
   q = glm.evaluate_mlm_quality_on_val(n_samples=500, mlm_probability=0.15, seed=0)
   print("VAL MLM quality:", q)

   # Save to CSV (single row)
   with open(OUT_MODEL_QUALITY, "w", newline="") as f:
       w = csv.writer(f)
       w.writerow(["val_mlm_loss", "val_perplexity", "n_samples"])
       w.writerow([q["val_mlm_loss"], q["val_perplexity"], q["n_samples"]])

   print("Saved model quality CSV:", OUT_MODEL_QUALITY)


if __name__ == "__main__":
   main()
