#!/usr/bin/env python3
import os
import csv
import math
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling

from glm_model import GLMModel
from sequence_plotter import plot as plot_reconstruction


# ============================================================
# FASTA reader
# ============================================================
def load_fasta(path: str) -> Tuple[List[str], List[str]]:
   headers: List[str] = []
   seqs: List[str] = []
   with open(path) as f:
       h = None
       buf = []
       for line in f:
           line = line.strip()
           if not line:
               continue
           if line.startswith(">"):
               if h is not None:
                   headers.append(h)
                   seqs.append("".join(buf))
               h = line[1:]
               buf = []
           else:
               buf.append(line)
       if h is not None:
           headers.append(h)
           seqs.append("".join(buf))
   return headers, seqs


def get_label_from_header(header: str) -> Optional[str]:
   for p in header.split("|"):
       if p.startswith("label="):
           return p.split("=", 1)[1]
   return None


def save_current_figure(png_path: str, dpi: int = 200) -> None:
   os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
   fig = plt.gcf()
   fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
   plt.close(fig)
   if not os.path.exists(png_path):
       raise RuntimeError(f"Plot not saved: {png_path}")
   size = os.path.getsize(png_path)
   if size < 10_000:
       raise RuntimeError(f"Plot saved but suspiciously small ({size} bytes): {png_path}")
   print(f"[OK] Saved: {png_path} ({size} bytes)")


# ============================================================
# Split helper (reproducible)
# ============================================================
def make_split(n: int, val_ratio: float = 0.2, seed: int = 727) -> Dict[str, np.ndarray]:
   idx = np.arange(n)
   rng = np.random.default_rng(seed)
   rng.shuffle(idx)
   n_val = int(round(val_ratio * n))
   val_idx = idx[:n_val]
   train_idx = idx[n_val:]
   return {"train_idx": train_idx, "val_idx": val_idx}


# ============================================================
# Eval 1: MLM quality on VAL (loss + perplexity)
# ============================================================
def evaluate_mlm_quality_on_val(
   glm: GLMModel,
   fasta_path: str,
   val_idx: List[int],
   n_samples: int = 500,
   mlm_probability: float = 0.15,
   seed: int = 0,
) -> Dict[str, float]:
   if glm.model is None:
       raise RuntimeError("Model not loaded.")

   headers, seqs = load_fasta(fasta_path)

   rng = random.Random(seed)
   idx = val_idx.copy()
   rng.shuffle(idx)
   idx = idx[: min(n_samples, len(idx))]

   data_collator = DataCollatorForLanguageModeling(
       tokenizer=glm.tokenizer, mlm=True, mlm_probability=mlm_probability
   )

   glm.model.eval()
   losses = []
   with torch.no_grad():
       for i in idx:
           # tokenizer returns dict; collator expects "input_ids" etc.
           item = glm.tokenizer(seqs[i], return_tensors=None)
           batch = data_collator([item])
           batch = {k: v.to(glm.device) for k, v in batch.items()}
           out = glm.model(**batch)
           losses.append(float(out.loss.item()))

   mean_loss = float(np.mean(losses)) if losses else float("nan")
   ppl = float(math.exp(mean_loss)) if (losses and mean_loss < 50) else float("nan")

   return {"val_mlm_loss": mean_loss, "val_perplexity": ppl, "n_samples": len(losses)}


# ============================================================
# Eval 2: delta_likelihood_fast
# ============================================================
def delta_likelihood_fast(
   glm: GLMModel,
   reference_sequence: str,
   perturbed_sequence: str,
   region: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
   if glm.model is None:
       raise RuntimeError("Model not loaded.")

   if len(reference_sequence) != len(perturbed_sequence):
       raise ValueError("ref and alt must have same length (use '-' for deletions).")

   if region is None:
       start, end = 0, len(reference_sequence)
   else:
       start, end = int(region[0]), int(region[1])
       start = max(0, start)
       end = min(len(reference_sequence), end)
       if end <= start:
           raise ValueError("region must satisfy end > start")

   ref_inputs = glm.tokenizer(reference_sequence, return_tensors="pt").to(glm.device)
   alt_inputs = glm.tokenizer(perturbed_sequence, return_tensors="pt").to(glm.device)

   with torch.no_grad():
       ref_logits = glm.model(**ref_inputs).logits[0]
       alt_logits = glm.model(**alt_inputs).logits[0]

   ref_logp = F.log_softmax(ref_logits, dim=-1)
   alt_logp = F.log_softmax(alt_logits, dim=-1)

   # token offset: +1 because [CLS] (or BOS) at position 0
   tok_start = start + 1
   tok_end = end + 1

   max_token_index = ref_inputs.input_ids.shape[1] - 2  # exclude last [SEP]
   tok_start = max(1, min(tok_start, max_token_index))
   tok_end = max(tok_start, min(tok_end, max_token_index + 1))

   ref_ids = ref_inputs.input_ids[0]
   alt_ids = alt_inputs.input_ids[0]
   idx = torch.arange(tok_start, tok_end, device=glm.device)

   ref_sum = float(ref_logp[idx, ref_ids[idx]].sum().item())
   alt_sum = float(alt_logp[idx, alt_ids[idx]].sum().item())
   delta = alt_sum - ref_sum

   return {"delta": delta, "reference_sum": ref_sum, "perturbed_sum": alt_sum, "region": (start, end)}


# ============================================================
# Eval 3: influence_probability_shift
# ============================================================
def influence_probability_shift(
   glm: GLMModel,
   reference_sequence: str,
   perturbed_sequence: str,
   query_positions: Optional[List[int]] = None,
   target_window: Optional[Tuple[int, int]] = None,
   metric: str = "max_abs_logodds",   # "tv" | "kl_ref_alt" | "max_abs_logodds"
   reduce: str = "mean",
   eps: float = 1e-9,
) -> Dict[str, Any]:
   if glm.model is None:
       raise RuntimeError("Model not loaded.")

   if len(reference_sequence) != len(perturbed_sequence):
       raise ValueError("ref and alt must have same length (use '-' for deletions).")

   if query_positions is None:
       query_positions = [i for i, (a, b) in enumerate(zip(reference_sequence, perturbed_sequence)) if a != b]

   if target_window is None:
       t0, t1 = 0, len(reference_sequence)
   else:
       t0, t1 = int(target_window[0]), int(target_window[1])
       t0 = max(0, t0)
       t1 = min(len(reference_sequence), t1)
       if t1 <= t0:
           raise ValueError("target_window must satisfy end > start")

   targets = list(range(t0, t1))
   relevant_chars = ["A", "C", "G", "T", "-"]
   relevant_ids = torch.tensor([glm.tokenizer.vocab[c] for c in relevant_chars], device=glm.device)

   def masked_probs(seq: str, j: int) -> torch.Tensor:
       s = list(seq)
       s[j] = "[MASK]"
       inputs = glm.tokenizer("".join(s), return_tensors="pt").to(glm.device)

       mask_id = glm.tokenizer.mask_token_id
       input_ids = inputs["input_ids"][0]
       mpos = (input_ids == mask_id).nonzero(as_tuple=True)[0]
       if len(mpos) != 1:
           raise RuntimeError(f"Expected 1 mask, found {len(mpos)}")
       mask_pos = int(mpos[0].item())

       with torch.no_grad():
           logits = glm.model(**inputs).logits[0, mask_pos]

       p = F.softmax(logits, dim=-1)[relevant_ids]
       p = p / (p.sum() + eps)
       return p

   def shift_score(p_ref: torch.Tensor, p_alt: torch.Tensor) -> torch.Tensor:
       if metric == "max_abs_logodds":
           return torch.max(torch.abs(torch.log(p_alt + eps) - torch.log(p_ref + eps)))
       if metric == "kl_ref_alt":
           return torch.sum(p_ref * (torch.log(p_ref + eps) - torch.log(p_alt + eps)))
       if metric == "tv":
           return 0.5 * torch.sum(torch.abs(p_alt - p_ref))
       raise ValueError(f"Unknown metric: {metric}")

   total = 0.0
   for q in query_positions:
       per_target = []
       for j in targets:
           if j == q:
               continue
           p_ref = masked_probs(reference_sequence, j)
           p_alt = masked_probs(perturbed_sequence, j)
           per_target.append(float(shift_score(p_ref, p_alt).item()))
       q_score = 0.0 if not per_target else (float(np.mean(per_target)) if reduce == "mean" else float(np.sum(per_target)))
       total += q_score

   return {
       "influence_score": float(total),
       "query_positions": query_positions,
       "target_window": (t0, t1),
       "metric": metric,
       "reduce": reduce,
   }


# ============================================================
# Dependency Map Generator (class)
# ============================================================
class DependencyMapGenerator:
   def __init__(self, glm: GLMModel):
       if glm.model is None:
           raise RuntimeError("GLMModel has no loaded model.")
       self.glm = glm
       self.device = glm.device
       self.relevant_chars = ["A", "C", "G", "T", "-"]
       self.relevant_ids = torch.tensor(
           [self.glm.tokenizer.vocab[c] for c in self.relevant_chars],
           device=self.device,
       )

   def _mask_probs(self, seq: str, j: int, eps: float = 1e-9) -> torch.Tensor:
       s = list(seq)
       s[j] = "[MASK]"
       inputs = self.glm.tokenizer("".join(s), return_tensors="pt").to(self.device)

       mask_id = self.glm.tokenizer.mask_token_id
       input_ids = inputs["input_ids"][0]
       mpos = (input_ids == mask_id).nonzero(as_tuple=True)[0]
       if len(mpos) != 1:
           raise RuntimeError(f"Expected exactly 1 mask, found {len(mpos)}")
       mask_pos = int(mpos[0].item())

       with torch.no_grad():
           logits = self.glm.model(**inputs).logits[0, mask_pos]

       p = F.softmax(logits, dim=-1)[self.relevant_ids]
       p = p / (p.sum() + eps)
       return p

   @staticmethod
   def _shift_score(p_ref: torch.Tensor, p_alt: torch.Tensor, metric: str, eps: float = 1e-9) -> torch.Tensor:
       if metric == "tv":
           return 0.5 * torch.sum(torch.abs(p_alt - p_ref))
       if metric == "kl_ref_alt":
           return torch.sum(p_ref * (torch.log(p_ref + eps) - torch.log(p_alt + eps)))
       if metric == "max_abs_logodds":
           return torch.max(torch.abs(torch.log(p_alt + eps) - torch.log(p_ref + eps)))
       raise ValueError(f"Unknown metric: {metric}")

   def compute_dependency_map(self, ref_seq: str, metric: str = "tv", set_diagonal_nan: bool = True) -> Dict[str, Any]:
       L = len(ref_seq)
       M = np.zeros((L, L), dtype=np.float32)

       ref_probs_by_j = [self._mask_probs(ref_seq, j) for j in range(L)]

       for i in range(L):
           alt = list(ref_seq)
           alt[i] = "-"
           alt_seq = "".join(alt)

           for j in range(L):
               if j == i:
                   M[i, j] = np.nan if set_diagonal_nan else 0.0
                   continue
               p_ref = ref_probs_by_j[j]
               p_alt = self._mask_probs(alt_seq, j)
               M[i, j] = float(self._shift_score(p_ref, p_alt, metric=metric).item())

       return {"ref_seq": ref_seq, "metric": metric, "map": M}

   @staticmethod
   def save_outputs(result: Dict[str, Any], out_dir: str, prefix: str, make_heatmap: bool = True) -> Dict[str, str]:
       os.makedirs(out_dir, exist_ok=True)
       M = result["map"]
       metric = result["metric"]

       npy_path = os.path.join(out_dir, f"{prefix}__dep_{metric}.npy")
       np.save(npy_path, M)
       outs = {"npy": npy_path}

       if make_heatmap:
           png_path = os.path.join(out_dir, f"{prefix}__dep_{metric}.png")
           plt.figure(figsize=(10, 8))
           plt.imshow(M, aspect="auto")
           plt.title(f"Dependency map ({metric})")
           plt.xlabel("target position j")
           plt.ylabel("deletion position i")
           plt.colorbar()
           plt.tight_layout()
           plt.savefig(png_path, dpi=220, bbox_inches="tight")
           plt.close()
           outs["png"] = png_path

       return outs


# ============================================================
# MAIN
# ============================================================
def main():
   MODEL_DIR = "/Users/amelielaura/Documents/dna_bert_final"
   FASTA_FILE = "/Users/amelielaura/Documents/Project6/data/augumented_sequence_size10000_length150_deletions0.2_nodeletionseq0.1.fasta"
   OUT_DIR = "/Users/amelielaura/Documents/Project6/outputs/all_results"

   os.makedirs(OUT_DIR, exist_ok=True)

   # ---- load model
   print("Loading GLMModel...")
   glm = GLMModel(model_path=MODEL_DIR, fasta_file=FASTA_FILE, force_retrain=False)

   # ---- load fasta for selection + split
   headers, seqs = load_fasta(FASTA_FILE)
   split = make_split(len(seqs), val_ratio=0.2, seed=727)
   val_idx = split["val_idx"].tolist()

   # ============================================================
   # 0) Reconstruction plot (example)
   # ============================================================
   print("Running reconstruction...")
   header = "seq0082|label=both|posAmotif=17|posBmotif=58|gaplength=30|deletions=24"
   sequence = "GTAT---TTAGTGTGGCATATTCACTACTC-TTCGGACCATTG-TACG-AAAAC-ACCGTACTGCG-TGA-TCCCCTCATAG-CGCA-A-A-TGTGTGGTAGT-C-GC-C-G-GCC--GCTAAAAGG---GAATTGTGTGC-TCACTAGG"

   prob_matrix = glm.get_full_reconstruction_probs(sequence, debug=True)
   plot_reconstruction(header, sequence, prob_matrix, motif_length=7)
   save_current_figure(os.path.join(OUT_DIR, "reconstruction_seq0082.png"), dpi=200)

   # ============================================================
   # 1) MLM quality on VAL
   # ============================================================
   print("Running MLM quality on VAL...")
   q = evaluate_mlm_quality_on_val(
       glm=glm,
       fasta_path=FASTA_FILE,
       val_idx=val_idx,
       n_samples=500,
       mlm_probability=0.15,
       seed=0,
   )
   print("VAL MLM quality:", q)
   with open(os.path.join(OUT_DIR, "model_quality.csv"), "w", newline="") as f:
       w = csv.writer(f)
       w.writerow(["val_mlm_loss", "val_perplexity", "n_samples"])
       w.writerow([q["val_mlm_loss"], q["val_perplexity"], q["n_samples"]])

   # ============================================================
   # 2) delta_likelihood_fast + 3) influence_probability_shift
   #    We create a valid alt with SAME LENGTH by replacing some bases with '-'
   # ============================================================
   print("Running delta_likelihood_fast + influence_probability_shift...")

   # pick a clean (no '-') val sequence as reference
   clean_val = [i for i in val_idx if "-" not in seqs[i]]
   if not clean_val:
       raise RuntimeError("No clean sequences in VAL (without '-') to build ref/alt.")

   ref_i = clean_val[0]
   ref_header = headers[ref_i]
   ref_seq = seqs[ref_i]

   # make same-length alt by inserting '-' at random positions
   rng = random.Random(123)
   alt_list = list(ref_seq)
   for pos in range(len(alt_list)):
       if rng.random() < 0.10:
           alt_list[pos] = "-"
   alt_seq = "".join(alt_list)

   d = delta_likelihood_fast(glm, ref_seq, alt_seq)
   infl = influence_probability_shift(glm, ref_seq, alt_seq, metric="tv", reduce="mean")

   with open(os.path.join(OUT_DIR, "eval_ref_alt.csv"), "w", newline="") as f:
       w = csv.writer(f)
       w.writerow(["ref_index", "ref_header", "delta", "ref_sum", "alt_sum", "influence_score", "metric"])
       w.writerow([ref_i, ref_header, d["delta"], d["reference_sum"], d["perturbed_sum"], infl["influence_score"], infl["metric"]])

   print("[OK] Saved eval_ref_alt.csv")

   # ============================================================
   # 4) Dependency maps (50 A_only + 50 B_only) from CLEAN sequences
   # ============================================================
   print("Running dependency maps...")
   dep_dir = os.path.join(OUT_DIR, "dependency_maps")
   os.makedirs(dep_dir, exist_ok=True)
   manifest = os.path.join(dep_dir, "manifest.csv")

   clean = [(h, s) for h, s in zip(headers, seqs) if "-" not in s]
   A_only = [(h, s) for h, s in clean if get_label_from_header(h) == "A_only"]
   B_only = [(h, s) for h, s in clean if get_label_from_header(h) == "B_only"]

   random.seed(727)
   random.shuffle(A_only)
   random.shuffle(B_only)
   chosen = A_only[: min(50, len(A_only))] + B_only[: min(50, len(B_only))]

   dep = DependencyMapGenerator(glm)
   metric = "tv"

   rows: List[Dict[str, Any]] = []
   for k, (h, s) in enumerate(chosen, start=1):
       label = get_label_from_header(h) or "unknown"
       seq_id = h.split("|")[0]
       print(f"[{k}/{len(chosen)}] {label} {seq_id} len={len(s)}")

       res = dep.compute_dependency_map(s, metric=metric, set_diagonal_nan=True)
       prefix = f"{label}__{seq_id}"
       outs = dep.save_outputs(res, out_dir=dep_dir, prefix=prefix, make_heatmap=True)

       input_txt = os.path.join(dep_dir, f"{prefix}__input.txt")
       with open(input_txt, "w") as f:
           f.write(f"ID: {seq_id}\nLABEL: {label}\nHEADER: {h}\nLENGTH: {len(s)}\n\n{s}\n")

       rows.append({
           "id": seq_id,
           "label": label,
           "length": len(s),
           "metric": metric,
           "input_txt": input_txt,
           "npy_path": outs.get("npy", ""),
           "png_path": outs.get("png", ""),
       })

   with open(manifest, "w", newline="") as f:
       w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
       w.writeheader()
       w.writerows(rows)

   print("\nDONE")
   print("OUT_DIR:", OUT_DIR)


if __name__ == "__main__":
   main()
