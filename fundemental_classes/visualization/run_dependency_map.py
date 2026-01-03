import os
import csv
import random

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
def save_input_sequence(out_dir: str, idx: int, header: str, ref_seq: str) -> str:
   """Save the exact sequence used for the dependency map."""
   os.makedirs(out_dir, exist_ok=True)
   path = os.path.join(out_dir, f"seq_{idx}__input.txt")
   with open(path, "w") as f:
       f.write(f"INDEX: {idx}\n")
       f.write(f"HEADER: {header}\n")
       f.write(f"LENGTH: {len(ref_seq)}\n\n")
       f.write(ref_seq + "\n")
   return path


def main(
   n_sequences: int = 10,
   metric: str = "tv",
   seed: int = 727,
   batch_size: int = 64,
):
   os.makedirs(OUT_DIR, exist_ok=True)
   random.seed(seed)

   glm = GLMModel(model_path=MODEL_OUT, fasta_file=FASTA_PATH, max_seq_length=256)
   dep = DependencyMapGenerator(glm)

   headers = glm.dataset.headers
   seqs = glm.dataset.seqs

   # choose clean sequences (no '-')
   clean_idx = [i for i, s in enumerate(seqs) if "-" not in s]
   if len(clean_idx) == 0:
       raise RuntimeError("No clean sequences found (no '-' in seq). Cannot build reference deletion scans.")

   random.shuffle(clean_idx)
   chosen = clean_idx[: min(n_sequences, len(clean_idx))]

   rows = []
   for k, i in enumerate(chosen, start=1):
       header = headers[i]
       ref_seq = seqs[i]

       print(f"[{k}/{len(chosen)}] seq index={i} len={len(ref_seq)}")

       # compute dependency map
       result = dep.compute_dependency_map(
           ref_seq=ref_seq,
           metric=metric,
           batch_size=batch_size,
           set_diagonal_nan=True,
       )

       prefix = f"seq_{i}__dep"
       outs = dep.save_outputs(result, out_dir=OUT_DIR, prefix=prefix, make_heatmap=True)

       # save input sequence next to the map
       input_txt = save_input_sequence(OUT_DIR, i, header, ref_seq)

       rows.append({
           "index": i,
           "header": header,
           "length": len(ref_seq),
           "metric": metric,
           "input_txt": input_txt,
           "npy_path": outs.get("npy", ""),
           "png_path": outs.get("png", ""),
       })

   # write manifest
   with open(MANIFEST, "w", newline="") as f:
       w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
       w.writeheader()
       w.writerows(rows)

   print("\nDone")
   print("Output directory:", OUT_DIR)
   print("Manifest:", MANIFEST)


if __name__ == "__main__":
   main(n_sequences=10, metric="tv", seed=727, batch_size=64)
