import os
import math
import random
from typing import Optional, Tuple, Dict, Any, List

import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, random_split

from transformers import (
   BertForMaskedLM,
   BertConfig,
   Trainer,
   TrainingArguments,
   DataCollatorForLanguageModeling,
   PreTrainedTokenizerFast,
)

from tokenizers import Tokenizer, models, pre_tokenizers
from dna_dataset import DNADataset


class GLMModel:
   """
   DNA Masked Language Model (MLM) wrapper.

   Supports:
     1) Training with TRAIN/VAL split (saved for reproducibility)
     2) Reconstruction probabilities (mask each position)
     3) Two deletion-effect scores:
        - Method 1: delta_likelihood_fast (global plausibility change)
        - Method 2: influence_probability_shift (distribution shift across positions)
     4) Model quality on VAL (MLM loss + perplexity)
   """

   SPLIT_FILE = "split_indices.npz"

   def __init__(self, model_path: str, fasta_file: str, max_seq_length: int = 122):
       self.model_path = model_path
       self.max_length = max_seq_length
       self.relevant_chars = ["A", "C", "G", "T", "-"]
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

       # --- Tokenizer ---
       if os.path.isdir(model_path) and (
           os.path.exists(os.path.join(model_path, "tokenizer.json"))
           or os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
       ):
           self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
       else:
           self.tokenizer = self.create_tokenizer(save_dir=model_path)

       # Make sure special tokens exist (sometimes HF loads but mask_token is None)
       if self.tokenizer.mask_token is None:
           # The token exists in vocab; just set the attribute.
           self.tokenizer.mask_token = "[MASK]"
       if self.tokenizer.cls_token is None:
           self.tokenizer.cls_token = "[CLS]"
       if self.tokenizer.sep_token is None:
           self.tokenizer.sep_token = "[SEP]"
       if self.tokenizer.pad_token is None:
           self.tokenizer.pad_token = "[PAD]"
       if self.tokenizer.unk_token is None:
           self.tokenizer.unk_token = "[UNK]"

       # --- Dataset ---
       self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)

       # --- Model ---
       if os.path.isdir(model_path) and (
           os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
           or os.path.exists(os.path.join(model_path, "model.safetensors"))
       ):
           self.model = BertForMaskedLM.from_pretrained(model_path).to(self.device)
           self.model.eval()
       else:
           self.model = None

   # -------------------------
   # Split handling (TRAIN/VAL)
   # -------------------------
   def _split_path(self) -> str:
       return os.path.join(self.model_path, self.SPLIT_FILE)

   def ensure_split(self, seed: int = 727, val_ratio: float = 0.2) -> Dict[str, np.ndarray]:
       """
       Create or load a train/val split.
       Stored in: model_out/split_indices.npz
       """
       os.makedirs(self.model_path, exist_ok=True)
       split_path = self._split_path()

       if os.path.exists(split_path):
           data = np.load(split_path)
           return {"train_idx": data["train_idx"], "val_idx": data["val_idx"]}

       n_total = len(self.dataset)
       idxs = np.arange(n_total)
       rng = np.random.default_rng(seed)
       rng.shuffle(idxs)

       n_val = int(n_total * val_ratio)
       val_idx = np.sort(idxs[:n_val])
       train_idx = np.sort(idxs[n_val:])

       np.savez(split_path, train_idx=train_idx, val_idx=val_idx)
       return {"train_idx": train_idx, "val_idx": val_idx}

   def get_split_indices(self) -> Dict[str, np.ndarray]:
       """
       For compatibility with our eval script.
       """
       split_path = self._split_path()
       if not os.path.exists(split_path):
           raise RuntimeError(
               f"No split indices stored yet at {split_path}.\n"
               f"Run training once (train_glm_local.py) to create them."
           )
       data = np.load(split_path)
       return {"train_idx": data["train_idx"], "val_idx": data["val_idx"]}

   # -------------------------
   # Train
   # -------------------------
   def _require_model(self):
       if self.model is None:
           raise RuntimeError(
               f"No trained model found in '{self.model_path}'.\n"
               f"Train first so model_out contains pytorch_model.bin/model.safetensors."
           )

   def _make_training_args(self, **kwargs):
       """
       Transformers versions differ: some use evaluation_strategy, some eval_strategy.
       We try one, and if TypeError occurs, we swap.
       """
       try:
           return TrainingArguments(**kwargs)
       except TypeError as e:
           msg = str(e)
           if "evaluation_strategy" in msg and "unexpected keyword" in msg:
               kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
               return TrainingArguments(**kwargs)
           raise

   def train(
       self,
       epochs: int = 30,
       batch_size: int = 16,
       lr: float = 2e-4,
       seed: int = 727,
       val_ratio: float = 0.2
   ):
       os.makedirs(self.model_path, exist_ok=True)

       # split indices (saved)
       split = self.ensure_split(seed=seed, val_ratio=val_ratio)
       train_ds = Subset(self.dataset, split["train_idx"].tolist())
       val_ds = Subset(self.dataset, split["val_idx"].tolist())

       data_collator = DataCollatorForLanguageModeling(
           tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
       )

       config = BertConfig(
           vocab_size=len(self.tokenizer.get_vocab()),
           hidden_size=256,
           num_hidden_layers=8,
           num_attention_heads=8,
           intermediate_size=1536,
           max_position_embeddings=512,
           type_vocab_size=1,
       )
       model = BertForMaskedLM(config)

       args = self._make_training_args(
           output_dir=self.model_path,
           overwrite_output_dir=True,
           num_train_epochs=epochs,
           per_device_train_batch_size=batch_size,
           per_device_eval_batch_size=batch_size,
           save_steps=500,
           logging_steps=200,
           evaluation_strategy="steps",
           eval_steps=500,
           save_strategy="steps",
           load_best_model_at_end=True,
           metric_for_best_model="eval_loss",
           report_to="none",
           learning_rate=lr,
           warmup_steps=100,
           dataloader_pin_memory=False,
           disable_tqdm=False,
       )

       trainer = Trainer(
           model=model,
           args=args,
           train_dataset=train_ds,
           eval_dataset=val_ds,
           data_collator=data_collator,
       )

       print("Starting training (train/val split saved in model_out)...")
       trainer.train()

       trainer.save_model(self.model_path)
       self.tokenizer.save_pretrained(self.model_path)

       self.model = model.to(self.device)
       self.model.eval()
       print("Training complete! Saved to:", self.model_path)

   # -------------------------
   # Model quality on VAL
   # -------------------------
   def evaluate_mlm_quality_on_val(
       self,
       n_samples: int = 500,
       mlm_probability: float = 0.15,
       seed: int = 0
   ) -> Dict[str, float]:
       """
       Overall model quality score (NOT deletion score):
       MLM loss + perplexity on VAL subset.
       """
       self._require_model()
       split = self.get_split_indices()
       val_idx = split["val_idx"].tolist()

       rng = random.Random(seed)
       rng.shuffle(val_idx)
       val_idx = val_idx[: min(n_samples, len(val_idx))]

       data_collator = DataCollatorForLanguageModeling(
           tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
       )

       self.model.eval()
       losses = []
       with torch.no_grad():
           for idx in val_idx:
               item = self.dataset[idx]
               batch = data_collator([item])
               batch = {k: v.to(self.device) for k, v in batch.items()}
               out = self.model(**batch)
               losses.append(float(out.loss.item()))

       mean_loss = float(np.mean(losses)) if losses else float("nan")
       ppl = float(math.exp(mean_loss)) if (losses and mean_loss < 50) else float("nan")

       return {"val_mlm_loss": mean_loss, "val_perplexity": ppl, "n_samples": len(losses)}

   # -------------------------
   # Reconstruction helpers
   # -------------------------
   def predict_position(self, sequence: str, position: int) -> np.ndarray:
       self._require_model()
       s = list(sequence)
       s[position] = "[MASK]"
       masked = "".join(s)

       inputs = self.tokenizer(masked, return_tensors="pt").to(self.device)
       mask_token_position = min(position + 1, inputs.input_ids.shape[1] - 2)

       with torch.no_grad():
           logits = self.model(**inputs).logits[0, mask_token_position]

       probs = torch.softmax(logits, dim=-1)
       return probs.detach().cpu().numpy()

   def get_full_reconstruction_probs(self, sequence_to_evaluate: str) -> np.ndarray:
       self._require_model()
       seq_len = len(sequence_to_evaluate)
       char_to_id = {c: self.tokenizer.vocab[c] for c in self.relevant_chars}
       prob_matrix = np.zeros((seq_len, len(self.relevant_chars)), dtype=float)

       for pos in range(seq_len):
           probs = self.predict_position(sequence_to_evaluate, pos)
           for j, c in enumerate(self.relevant_chars):
               prob_matrix[pos, j] = probs[char_to_id[c]]

       return prob_matrix

   # -------------------------
   # Method 1: delta_likelihood_fast
   # -------------------------
   def delta_likelihood_fast(
       self,
       reference_sequence: str,
       perturbed_sequence: str,
       region: Optional[Tuple[int, int]] = None
   ) -> Dict[str, Any]:
       self._require_model()

       if len(reference_sequence) != len(perturbed_sequence):
           raise ValueError("ref and alt must have the same length (use '-' for deletions).")

       if region is None:
           start, end = 0, len(reference_sequence)
       else:
           start, end = int(region[0]), int(region[1])
           start = max(0, start)
           end = min(len(reference_sequence), end)
           if end <= start:
               raise ValueError("region must satisfy end > start")

       ref_inputs = self.tokenizer(reference_sequence, return_tensors="pt").to(self.device)
       alt_inputs = self.tokenizer(perturbed_sequence, return_tensors="pt").to(self.device)

       with torch.no_grad():
           ref_logits = self.model(**ref_inputs).logits[0]
           alt_logits = self.model(**alt_inputs).logits[0]

       ref_logp = F.log_softmax(ref_logits, dim=-1)
       alt_logp = F.log_softmax(alt_logits, dim=-1)

       tok_start = start + 1
       tok_end = end + 1

       max_token_index = ref_inputs.input_ids.shape[1] - 2
       tok_start = max(1, min(tok_start, max_token_index))
       tok_end = max(tok_start, min(tok_end, max_token_index + 1))

       ref_ids = ref_inputs.input_ids[0]
       alt_ids = alt_inputs.input_ids[0]
       idx = torch.arange(tok_start, tok_end, device=self.device)

       ref_sum = float(ref_logp[idx, ref_ids[idx]].sum().item())
       alt_sum = float(alt_logp[idx, alt_ids[idx]].sum().item())
       delta = alt_sum - ref_sum

       return {
           "delta": delta,
           "reference_sum": ref_sum,
           "perturbed_sum": alt_sum,
           "region": (start, end)
       }

   # -------------------------
   # Method 2: influence_probability_shift
   # -------------------------
   def influence_probability_shift(
       self,
       reference_sequence: str,
       perturbed_sequence: str,
       query_positions: Optional[List[int]] = None,
       target_window: Optional[Tuple[int, int]] = None,
       metric: str = "max_abs_logodds",
       reduce: str = "mean",
       eps: float = 1e-9
   ) -> Dict[str, Any]:
       self._require_model()

       if len(reference_sequence) != len(perturbed_sequence):
           raise ValueError("ref and alt must have the same length (use '-' for deletions).")

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
       relevant_ids = torch.tensor([self.tokenizer.vocab[c] for c in self.relevant_chars], device=self.device)

       def masked_probs(seq: str, j: int) -> torch.Tensor:
           s = list(seq)
           s[j] = "[MASK]"
           inputs = self.tokenizer("".join(s), return_tensors="pt").to(self.device)
           mask_pos = min(j + 1, inputs.input_ids.shape[1] - 2)

           with torch.no_grad():
               logits = self.model(**inputs).logits[0, mask_pos]

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

           q_score = float(np.mean(per_target)) if (per_target and reduce == "mean") else float(np.sum(per_target)) if per_target else 0.0
           total += q_score

       return {
           "influence_score": float(total),
           "query_positions": query_positions,
           "target_window": (t0, t1),
           "metric": metric,
           "reduce": reduce,
       }

   # -------------------------
   # Tokenizer
   # -------------------------
   @staticmethod
   def create_tokenizer(save_dir: Optional[str] = None) -> PreTrainedTokenizerFast:
       vocab = {
           "[PAD]": 0,
           "[UNK]": 1,
           "[CLS]": 2,
           "[SEP]": 3,
           "[MASK]": 4,
           "A": 5,
           "C": 6,
           "G": 7,
           "T": 8,
           "-": 9,
       }

       tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
       tok.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")

       if save_dir is not None:
           os.makedirs(save_dir, exist_ok=True)
           path = os.path.join(save_dir, "tokenizer.json")
       else:
           path = "tokenizer.json"

       tok.save(path)

       return PreTrainedTokenizerFast(
           tokenizer_file=path,
           unk_token="[UNK]",
           sep_token="[SEP]",
           pad_token="[PAD]",
           cls_token="[CLS]",
           mask_token="[MASK]",
       )
