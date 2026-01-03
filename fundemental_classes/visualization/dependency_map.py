import os
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from fundemental_classes.glm_model2 import GLMModel  


class DependencyMapGenerator:
   """
   Deletion-scan dependency map.

   For a given reference sequence (length L):
     - for each deletion position i:
         alt = ref with '-' at i  (keeps alignment)
         for each target position j:
             compare p_ref(. | mask j, context=ref) vs p_alt(. | mask j, context=alt)

   Output:
     map[i, j] = shift score (could be TV / KL / max-abs-logodds; in our case TV)

   This is a so called "scan-off heatmap":
     rows = deleted positions
     columns = affected target positions (masked)
   """

   def __init__(
       self,
       glm: GLMModel,
       relevant_chars: Optional[List[str]] = None,
       eps: float = 1e-9,
   ):
       self.glm = glm
       if getattr(self.glm, "model", None) is None:
           raise RuntimeError("GLMModel has no loaded model. Train/load it first.")

       self.device = self.glm.device
       self.eps = eps

       self.relevant_chars = relevant_chars or ["A", "C", "G", "T", "-"]
       vocab = self.glm.tokenizer.get_vocab()
       self.relevant_ids = torch.tensor([vocab[c] for c in self.relevant_chars], device=self.device)

   # ------------------------------------------------------------------
   # Batched masked probabilities over relevant tokens only
   # ------------------------------------------------------------------
   def _masked_probs_batch(self, seq: str, positions: List[int], batch_size: int = 64) -> np.ndarray:
       """
       For each target position j in `positions`:
         - create a copy of seq with seq[j] = [MASK]
         - run through model in batches
         - return probabilities over [A,C,G,T,'-'] only

       Returns: array shape (len(positions), 5)
       """
       results = []

       for start in range(0, len(positions), batch_size):
           batch_positions = positions[start:start + batch_size]

           masked_sequences = []
           for j in batch_positions:
               s = list(seq)
               s[j] = "[MASK]"
               masked_sequences.append("".join(s))

           # Important: avoid truncation
           tokenized = self.glm.tokenizer(
               masked_sequences,
               return_tensors="pt",
               padding=True,
               truncation=False,
           ).to(self.device)

           input_ids = tokenized["input_ids"]
           tok_len = input_ids.shape[1]

           # mask token position in token space ~ (j + 1) due to [CLS]
           mask_positions = [min(j + 1, tok_len - 2) for j in batch_positions]
           mask_positions = torch.tensor(mask_positions, device=self.device)

           with torch.no_grad():
               logits = self.glm.model(**tokenized).logits  # (B, T, V)

           b_idx = torch.arange(logits.shape[0], device=self.device)
           masked_logits = logits[b_idx, mask_positions]  # (B, V)

           probs = F.softmax(masked_logits, dim=-1)[:, self.relevant_ids]  # (B, 5)
           probs = probs / (probs.sum(dim=-1, keepdim=True) + self.eps)

           results.append(probs.detach().cpu().numpy())

       return np.vstack(results)

   # ------------------------------------------------------------------
   # Shift metrics
   # ------------------------------------------------------------------
   def _shift_score(self, p_ref: np.ndarray, p_alt: np.ndarray, metric: str) -> np.ndarray:
       """
       p_ref and p_alt shape: (N, 5)
       returns per-position shift score shape: (N,)
       """
       if metric == "tv":
           return 0.5 * np.sum(np.abs(p_alt - p_ref), axis=1)

       if metric == "kl_ref_alt":
           return np.sum(p_ref * (np.log(p_ref + self.eps) - np.log(p_alt + self.eps)), axis=1)

       if metric == "max_abs_logodds":
           return np.max(np.abs(np.log(p_alt + self.eps) - np.log(p_ref + self.eps)), axis=1)

       raise ValueError("Unknown metric. Use one of: tv | kl_ref_alt | max_abs_logodds")

   # ------------------------------------------------------------------
   # compute dependency map
   # ------------------------------------------------------------------
   def compute_dependency_map(
       self,
       ref_seq: str,
       deletion_positions: Optional[List[int]] = None,
       target_positions: Optional[List[int]] = None,
       metric: str = "tv",
       batch_size: int = 64,
       set_diagonal_nan: bool = True,
   ) -> Dict[str, Any]:
       """
       Compute a dependency map:
         rows = deleted positions
         cols = target positions (masked)

       Returns:
         {
           "map": (n_del, n_target) array,
           "deletion_positions": [...],
           "target_positions": [...],
           "metric": metric
         }
       """
       L = len(ref_seq)
       deletion_positions = deletion_positions or list(range(L))
       target_positions = target_positions or list(range(L))

       # Fast diagonal with the index
       pos_to_col = {p: c for c, p in enumerate(target_positions)}

       # Baseline probs (ref) for all targets once
       p_ref = self._masked_probs_batch(ref_seq, target_positions, batch_size=batch_size)

       M = np.zeros((len(deletion_positions), len(target_positions)), dtype=float)

       for r, q in enumerate(deletion_positions):
           alt = list(ref_seq)
           alt[q] = "-"  # represent deletion while keeping alignment
           alt = "".join(alt)

           p_alt = self._masked_probs_batch(alt, target_positions, batch_size=batch_size)
           M[r, :] = self._shift_score(p_ref, p_alt, metric=metric)

           if set_diagonal_nan and q in pos_to_col:
               M[r, pos_to_col[q]] = np.nan

       return {
           "map": M,
           "deletion_positions": deletion_positions,
           "target_positions": target_positions,
           "metric": metric,
       }

   # ------------------------------------------------------------------
   # Save map + heatmap
   # ------------------------------------------------------------------
   def save_outputs(
       self,
       result: Dict[str, Any],
       out_dir: str,
       prefix: str = "dependency_map",
       make_heatmap: bool = True,
   ) -> Dict[str, str]:
       """
       Save:
         - .npy matrix
         - optional .png heatmap
       """
       os.makedirs(out_dir, exist_ok=True)
       M = result["map"]
       metric = result["metric"]

       npy_path = os.path.join(out_dir, f"{prefix}_{metric}.npy")
       np.save(npy_path, M)

       outputs = {"npy": npy_path}

       if make_heatmap:
           png_path = os.path.join(out_dir, f"{prefix}_{metric}.png")
           plt.figure(figsize=(10, 6))
           plt.imshow(M, aspect="auto")
           plt.colorbar(label=f"Shift score ({metric})")
           plt.xlabel("Target position j (masked)")
           plt.ylabel("Deleted position i")
           plt.title("Deletion-scan dependency map")
           plt.tight_layout()
           plt.savefig(png_path, dpi=200)
           plt.close()
           outputs["png"] = png_path

       return outputs
