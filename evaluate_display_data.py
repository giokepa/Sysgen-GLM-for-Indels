import os
import csv
import random
from pathlib import Path

import numpy as np

from fundamental_classes.model_related.glm_model_new import GLMModel
from  fundamental_classes.visualization.dependency_map import DependencyMapGenerator
############################################

# --- ROOT of project ---

PROJECT_ROOT = os.path.abspath(os.path.join( ".."))

# --- inputs ---
FASTA_PATH = PROJECT_ROOT / "data" / "augumented_sequence_size10000_length150_deletions0.1_nodeletionseq0.25.fasta"
MODEL_OUT  = PROJECT_ROOT / "model_out"

# --- outputs ---
OUT_DIR    = PROJECT_ROOT / "outputs" / "dependency_maps"
MANIFEST   = OUT_DIR / "manifest.csv"

# --- run settings ---
SEED = 727
METRIC = "tv"          # "tv" | "kl_ref_alt" | "max_abs_logodds"
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 256

# --- how many sequences per label ---
N_A_ONLY = 50
N_B_ONLY = 50
MAKE_HEATMAPS = True

###############################################
def get_label_from_header(header: str, key: str = "label") -> str:
   """
   Expected header format like:
     seq0001|label=A_only|posAmotif=12|...
   """
   h = header[1:] if header.startswith(">") else header
   parts = h.split("|")
   for p in parts:
       if p.startswith(f"{key}="):
           return p.split("=", 1)[1].strip()
   return "unknown"


def is_clean(seq: str) -> bool:
   return "-" not in seq


def sample_indices_by_label(headers, seqs, label: str, n: int, rng: random.Random):
   candidates = [
       i for i, (h, s) in enumerate(zip(headers, seqs))
       if is_clean(s) and get_label_from_header(h) == label
   ]
   rng.shuffle(candidates)
   return candidates[: min(n, len(candidates))]

###############################################

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)

glm = GLMModel(model_path=str(MODEL_OUT), fasta_file=str(FASTA_PATH), max_seq_length=MAX_SEQ_LENGTH)
dep = DependencyMapGenerator(glm)

headers = glm.dataset.headers
seqs = glm.dataset.seqs

print("Loaded sequences:", len(seqs))
print("Example header:", headers[0])

####################################

idx_A = sample_indices_by_label(headers, seqs, "A_only", N_A_ONLY, rng)
idx_B = sample_indices_by_label(headers, seqs, "B_only", N_B_ONLY, rng)

print("Picked A_only:", len(idx_A))
print("Picked B_only:", len(idx_B))

if len(idx_A) == 0 and len(idx_B) == 0:
   raise RuntimeError("No sequences selected. Check if headers contain label=A_only / label=B_only.")

#####################################

def save_input_sequence(out_dir: Path, idx: int, header: str, ref_seq: str) -> Path:
   out_dir.mkdir(parents=True, exist_ok=True)
   p = out_dir / f"seq_{idx}__input.txt"
   with open(p, "w") as f:
       f.write(f"INDEX: {idx}\n")
       f.write(f"HEADER: {header}\n")
       f.write(f"LABEL: {get_label_from_header(header)}\n")
       f.write(f"LENGTH: {len(ref_seq)}\n\n")
       f.write(ref_seq + "\n")
   return p


def run_for_indices(indices, label_name: str):
   rows = []
   label_dir = OUT_DIR / label_name
   label_dir.mkdir(parents=True, exist_ok=True)

   for k, idx in enumerate(indices, start=1):
       header = headers[idx]
       ref_seq = seqs[idx]

       print(f"[{label_name}] {k}/{len(indices)} idx={idx} len={len(ref_seq)}")

       result = dep.compute_dependency_map(
           ref_seq=ref_seq,
           metric=METRIC,
           batch_size=BATCH_SIZE,
           set_diagonal_nan=True
       )

       prefix = f"seq_{idx}__dep"
       outs = dep.save_outputs(
           result=result,
           out_dir=str(label_dir),
           prefix=prefix,
           make_heatmap=MAKE_HEATMAPS
       )

       input_txt = save_input_sequence(label_dir, idx, header, ref_seq)

       rows.append({
           "index": idx,
           "label": label_name,
           "header": header,
           "length": len(ref_seq),
           "metric": METRIC,
           "input_txt": str(input_txt),
           "npy_path": outs.get("npy", ""),
           "png_path": outs.get("png", ""),
           "out_dir": str(label_dir),
       })

   return rows


rows = []
rows += run_for_indices(idx_A, "A_only")
rows += run_for_indices(idx_B, "B_only")

# write manifest
with open(MANIFEST, "w", newline="") as f:
   w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
   w.writeheader()
   w.writerows(rows)

print("\nDONE")
print("OUT_DIR:", OUT_DIR)
print("MANIFEST:", MANIFEST)
