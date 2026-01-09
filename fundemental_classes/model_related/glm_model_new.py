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

   Extensions:
     - train/val split saved to split_indices.npz
     - HF TrainingArguments compatibility (evaluation_strategy vs eval_strategy)
     - evaluate_mlm_quality_on_val (loss + perplexity on VAL) + CSV export
     - delta_likelihood_fast
     - influence_probability_shift
   """

   SPLIT_FILE = "split_indices.npz"

   def __init__(self, model_path: str, fasta_file: str, max_seq_length: int = 122, force_retrain: bool = False):
       self.model_path = model_path
       self.meta_path = os.path.join(model_path, "training_metadata.json")
       self.max_length = max_seq_length
       self.relevant_chars = ["A", "C", "G", "T", "-"]
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

       if force_retrain:
           print("force_retrain=True: Clearing all model files")
           self._cleanup_all()

       load_success = False
       if os.path.exists(model_path) and not force_retrain:
           load_success = self._try_load_existing_model(fasta_file, max_seq_length)

       if not load_success:
           print("Initializing fresh model")

           # Tokenizer: if already present in model_path use it, else create
           if os.path.isdir(model_path) and (
               os.path.exists(os.path.join(model_path, "tokenizer.json"))
               or os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
           ):
               self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
           else:
               self.tokenizer = self.create_tokenizer(save_dir=model_path)

           self._ensure_special_tokens()
           self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)
           self.model = None
           print("No trained model loaded. Call train() to train the model.")

       self._ensure_special_tokens()

   # ============================================================
   # Cleanup / Loading / Metadata
   # ============================================================
   def _cleanup_all(self):
       """Delete model_path using only os.*"""
       if os.path.isdir(self.model_path):
           for root, dirs, files in os.walk(self.model_path, topdown=False):
               for name in files:
                   p = os.path.join(root, name)
                   try:
                       os.remove(p)
                   except Exception:
                       pass
               for name in dirs:
                   p = os.path.join(root, name)
                   try:
                       os.rmdir(p)
                   except Exception:
                       pass
           try:
               os.rmdir(self.model_path)
               print(f"Removed directory: {self.model_path}")
           except Exception as e:
               print(f"Could not fully remove {self.model_path}: {e}")

   def _try_load_existing_model(self, fasta_file: str, max_seq_length: int) -> bool:
       try:
           print(f"Checking for existing trained model in {self.model_path}")

           metadata = self._load_metadata()
           if not metadata.get("trained", False):
               print("No trained model found (metadata missing or trained=False)")
               return False

           required_files = {
               "config.json": os.path.join(self.model_path, "config.json"),
               "tokenizer.json": os.path.join(self.model_path, "tokenizer.json"),
           }

           weights_bin = os.path.join(self.model_path, "pytorch_model.bin")
           weights_safe = os.path.join(self.model_path, "model.safetensors")
           if os.path.exists(weights_bin):
               required_files["model weights"] = weights_bin
           else:
               required_files["model weights"] = weights_safe

           missing = [name for name, path in required_files.items() if not os.path.exists(path)]
           if missing:
               print(f"Model directory incomplete. Missing: {', '.join(missing)}")
               return False

           print("Loading tokenizer...")
           self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_path)
           self._ensure_special_tokens()
           print("Tokenizer loaded")

           print("Loading dataset...")
           self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)
           print("Dataset loaded")

           print("Loading model...")
           self.model = BertForMaskedLM.from_pretrained(self.model_path).to(self.device)
           try:
               self.model.tie_weights()
           except Exception:
               pass
           self.model.eval()
           print("Model loaded successfully!")

           print(f"Trained for {metadata.get('epochs_completed', 'N/A')} epochs")
           if "final_val_loss" in metadata:
               try:
                   print(f"Final validation loss: {float(metadata['final_val_loss']):.4f}")
               except Exception:
                   print(f"Final validation loss: {metadata['final_val_loss']}")

           return True

       except Exception as e:
           print(f"\nError loading model: {type(e).__name__}")
           print(f"Details: {str(e)[:200]}")
           print("\nAuto-cleanup: Removing corrupted model files...")
           self._cleanup_all()
           print("Cleanup complete. Will initialize fresh model.\n")
           return False

   def _load_metadata(self) -> Dict[str, Any]:
       if os.path.exists(self.meta_path):
           try:
               json_mod = __import__("json")
               with open(self.meta_path, "r") as f:
                   return json_mod.load(f)
           except Exception:
               return {}
       return {}

   def _save_metadata(self, epochs: int, final_train_loss: float, final_val_loss: float):
       try:
           json_mod = __import__("json")
           dt_mod = __import__("datetime")
           metadata = {
               "trained": True,
               "epochs_completed": int(epochs),
               "final_train_loss": float(final_train_loss),
               "final_val_loss": float(final_val_loss),
               "training_date": dt_mod.datetime.now().isoformat(),
               "dataset_size": int(len(self.dataset)),
           }
           os.makedirs(self.model_path, exist_ok=True)
           with open(self.meta_path, "w") as f:
               json_mod.dump(metadata, f, indent=2)
           print(f"Metadata saved to {self.meta_path}")
       except Exception as e:
           print(f"Could not save metadata: {e}")

   def _ensure_special_tokens(self):
       if getattr(self, "tokenizer", None) is None:
           return
       if self.tokenizer.mask_token is None:
           self.tokenizer.mask_token = "[MASK]"
       if self.tokenizer.cls_token is None:
           self.tokenizer.cls_token = "[CLS]"
       if self.tokenizer.sep_token is None:
           self.tokenizer.sep_token = "[SEP]"
       if self.tokenizer.pad_token is None:
           self.tokenizer.pad_token = "[PAD]"
       if self.tokenizer.unk_token is None:
           self.tokenizer.unk_token = "[UNK]"

   # ============================================================
   # Split
   # ============================================================
   def _split_path(self) -> str:
       return os.path.join(self.model_path, self.SPLIT_FILE)

   def ensure_split(self, seed: int = 727, val_ratio: float = 0.2) -> Dict[str, np.ndarray]:
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
       split_path = self._split_path()
       if not os.path.exists(split_path):
           raise RuntimeError(
               f"No split indices stored yet at {split_path}. "
               f"Train once with save_split=True or call ensure_split()."
           )
       data = np.load(split_path)
       return {"train_idx": data["train_idx"], "val_idx": data["val_idx"]}

   # ============================================================
   # HF TrainingArguments compatibility
   # ============================================================
   def _make_training_args(self, **kwargs) -> TrainingArguments:
       try:
           return TrainingArguments(**kwargs)
       except TypeError as e:
           msg = str(e)
           if "evaluation_strategy" in msg and "unexpected keyword" in msg:
               kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
               return TrainingArguments(**kwargs)
           raise

   # ============================================================
   # Train
   # ============================================================
   def train(
       self,
       epochs: int = 30,
       batch_size: int = 16,
       lr: float = 2e-4,
       seed: int = 727,
       val_ratio: float = 0.2,
       validation_split: Optional[float] = None,
       save_split: bool = True,
   ):
       os.makedirs(self.model_path, exist_ok=True)

       if validation_split is not None:
           val_ratio = validation_split

       if getattr(self, "dataset", None) is None:
           raise RuntimeError("Dataset not initialized. Check fasta_file and DNADataset initialization.")

       if save_split:
           split = self.ensure_split(seed=seed, val_ratio=val_ratio)
           train_ds = Subset(self.dataset, split["train_idx"].tolist())
           val_ds = Subset(self.dataset, split["val_idx"].tolist())
       else:
           g = torch.Generator().manual_seed(seed)
           n_total = len(self.dataset)
           n_val = int(n_total * val_ratio)
           n_train = n_total - n_val
           train_ds, val_ds = random_split(self.dataset, [n_train, n_val], generator=g)

       print(f"Dataset split: {len(train_ds)} training, {len(val_ds)} validation")

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
           tie_word_embeddings=True,
       )
       model = BertForMaskedLM(config)
       model.tie_weights()

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
           greater_is_better=False,       
           report_to="all",
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

       print("Starting Training")
       trainer.train()

       print("Saving model and tokenizer...")
       trainer.save_model(self.model_path)
       self.tokenizer.save_pretrained(self.model_path)
       print("Training complete")

       log_history = trainer.state.log_history
       train_losses = [log["loss"] for log in log_history if "loss" in log]
       eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]

       final_train_loss = train_losses[-1] if train_losses else float("nan")
       final_val_loss = eval_losses[-1] if eval_losses else float("nan")
       self._save_metadata(epochs, final_train_loss, final_val_loss)

       self.plot_training_and_validation_curves(
           log_history,
           save_path=os.path.join(self.model_path, "training_curves.png"),
       )

       self.model = trainer.model.to(self.device)
       try:
           self.model.tie_weights()
       except Exception:
           pass
       self.model.eval()

   # ============================================================
   # Model quality on VAL (+ CSV export)
   # ============================================================
   def evaluate_mlm_quality_on_val(
       self,
       n_samples: int = 500,
       mlm_probability: float = 0.15,
       seed: int = 0,
       write_csv: bool = True,
   ) -> Dict[str, float]:
       if self.model is None:
           raise RuntimeError("Model not loaded or trained. Call train() first or load a trained model.")

       # ensure split exists
       if not os.path.exists(self._split_path()):
           self.ensure_split(seed=727, val_ratio=0.2)

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

       result = {"val_mlm_loss": mean_loss, "val_perplexity": ppl, "n_samples": len(losses)}

       if write_csv:
           os.makedirs(self.model_path, exist_ok=True)
           csv_path = os.path.join(self.model_path, "model_quality.csv")
           with open(csv_path, "w") as f:
               f.write("val_mlm_loss,val_perplexity,n_samples\n")
               f.write(f"{mean_loss},{ppl},{len(losses)}\n")
           result["csv_path"] = csv_path

       return result

   # ============================================================
   # Reconstruction helpers (MASK POSITION)
   # ============================================================
   def predict_position(self, sequence: str, position: int) -> np.ndarray:
       """
       Mask ONE character at `position` and return full vocab probabilities.
       Uses token-id search to find the true [MASK] position (robust).
       """
       if self.model is None:
           raise RuntimeError("Model not loaded or trained. Call train() first or load a trained model.")

       if position < 0 or position >= len(sequence):
           raise ValueError(f"Position {position} out of range for sequence length {len(sequence)}")

       s = list(sequence)
       s[position] = "[MASK]"
       masked = "".join(s)

       inputs = self.tokenizer(masked, return_tensors="pt").to(self.device)

       mask_id = self.tokenizer.mask_token_id
       input_ids = inputs["input_ids"][0]
       mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]
       if len(mask_positions) != 1:
           raise RuntimeError(f"Expected exactly 1 [MASK] token, found {len(mask_positions)}")

       mask_pos = int(mask_positions[0].item())

       with torch.no_grad():
           logits = self.model(**inputs).logits[0, mask_pos]

       probs = torch.softmax(logits, dim=-1)
       return probs.detach().cpu().numpy()

   def get_full_reconstruction_probs(self, sequence_to_evaluate: str) -> np.ndarray:
       if self.model is None:
           raise RuntimeError("Model not loaded or trained. Call train() first or load a trained model.")

       seq_len = len(sequence_to_evaluate)
       char_to_id = {c: self.tokenizer.vocab[c] for c in self.relevant_chars}
       prob_matrix = np.zeros((seq_len, len(self.relevant_chars)), dtype=float)

       for pos in range(seq_len):
           probs = self.predict_position(sequence_to_evaluate, pos)
           for j, c in enumerate(self.relevant_chars):
               prob_matrix[pos, j] = probs[char_to_id[c]]

       return prob_matrix

   # ============================================================
   # Method 1: delta_likelihood_fast
   # ============================================================
   def delta_likelihood_fast(
       self,
       reference_sequence: str,
       perturbed_sequence: str,
       region: Optional[Tuple[int, int]] = None
   ) -> Dict[str, Any]:
       if self.model is None:
           raise RuntimeError("Model not loaded or trained. Call train() first or load a trained model.")

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

       return {"delta": delta, "reference_sum": ref_sum, "perturbed_sum": alt_sum, "region": (start, end)}

   # ============================================================
   # Method 2: influence_probability_shift
   # ============================================================
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
       if self.model is None:
           raise RuntimeError("Model not loaded or trained. Call train() first or load a trained model.")

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

           # robust mask position
           mask_id = self.tokenizer.mask_token_id
           input_ids = inputs["input_ids"][0]
           mpos = (input_ids == mask_id).nonzero(as_tuple=True)[0]
           if len(mpos) != 1:
               raise RuntimeError(f"Expected 1 mask, found {len(mpos)}")
           mask_pos = int(mpos[0].item())

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

           q_score = 0.0 if not per_target else (float(np.mean(per_target)) if reduce == "mean" else float(np.sum(per_target)))
           total += q_score

       return {"influence_score": float(total), "query_positions": query_positions, "target_window": (t0, t1), "metric": metric, "reduce": reduce}

   # ============================================================
   # Plotting
   # ============================================================
   @staticmethod
   def plot_training_and_validation_curves(log_history, save_path=None):
       import matplotlib.pyplot as plt

       if not log_history or len(log_history) < 2:
           print("No training logs available for plotting")
           return

       train_losses, train_steps = [], []
       eval_losses, eval_steps = [], []
       lrs, lr_steps = [], []

       for i, log in enumerate(log_history):
           if "loss" in log:
               train_losses.append(log["loss"])
               train_steps.append(log.get("step", i))
           if "eval_loss" in log:
               eval_losses.append(log["eval_loss"])
               eval_steps.append(log.get("step", i))
           if "learning_rate" in log:
               lrs.append(log["learning_rate"])
               lr_steps.append(log.get("step", i))

       if not train_losses:
           print("No loss data found in logs")
           return

       fig, axes = plt.subplots(1, 2, figsize=(14, 5))

       ax1 = axes[0]
       ax1.plot(train_steps, train_losses, "b-", linewidth=2, marker="o", markersize=4, label="Training Loss", alpha=0.7)
       if eval_losses:
           ax1.plot(eval_steps, eval_losses, "r-", linewidth=2, marker="s", markersize=4, label="Validation Loss", alpha=0.7)
       ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
       ax1.set_xlabel("Training Steps", fontsize=10)
       ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
       ax1.legend(loc="best", fontsize=9)
       ax1.grid(True, alpha=0.3, linestyle="--")

       ax2 = axes[1]
       if lrs:
           ax2.plot(lr_steps, lrs, "g-", linewidth=2, marker="d", markersize=4)
           ax2.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
           ax2.set_xlabel("Training Steps", fontsize=10)
           ax2.set_ylabel("Learning Rate", fontsize=10)
           ax2.grid(True, alpha=0.3, linestyle="--")
           ax2.set_yscale("log")
       else:
           ax2.text(0.5, 0.5, "No LR data logged", ha="center", va="center", transform=ax2.transAxes, fontsize=11)

       plt.tight_layout()
       if save_path:
           plt.savefig(save_path, dpi=150, bbox_inches="tight")
           print(f"Training curves saved to {save_path}")
       plt.show()

   # ============================================================
   # Tokenizer
   # ============================================================
   @staticmethod
   def create_tokenizer(save_dir: Optional[str] = None) -> PreTrainedTokenizerFast:
       vocab = {
           "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
           "A": 5, "C": 6, "G": 7, "T": 8, "-": 9,
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
