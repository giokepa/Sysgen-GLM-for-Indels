import os
import math
import random
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

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

   This is a DNA Masked Language Model (MLM) wrapper.

   It supports:
     (A) Training a small BERT-style MLM on DNA tokens (A,C,G,T, '-') using Georgi’s training approach.
     (B) Reconstructing sequences position-by-position (mask one position and predict probabilities).
     (C) Evaluating the effect of deletions using two new scores:
         1) A FAST "global disruption" score (delta_likelihood_fast)
         2) A slower but more informative "probability shift / influence" score (influence_probability_shift)

   Important:
     - Right now we only do deletions. That means: ref and alt must have equal length.
     - Deletions are encoded using '-' so the sequences stay aligned.
   """

   def __init__(self, model_path: str, fasta_file: str, max_seq_length: int = 122):
       self.model_path = model_path
       self.max_length = max_seq_length
       self.relevant_chars = ["A", "C", "G", "T", "-"]
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

       # 1) Tokenizer
       # If model_path contains a saved tokenizer, load it.
       # Otherwise create a minimal tokenizer and (optionally) save it.
       if os.path.isdir(model_path) and any(
           os.path.exists(os.path.join(model_path, f)) for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
       ):
           self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
       else:
           self.tokenizer = self.create_tokenizer(save_dir=model_path)

       # 2) Dataset
       self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)

       # 3) Model
       # If model files exist, load them. Otherwise set to None and expect train() later.
       if os.path.isdir(model_path) and (
           os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
           or os.path.exists(os.path.join(model_path, "model.safetensors"))
       ):
           self.model = BertForMaskedLM.from_pretrained(model_path).to(self.device)
           self.model.eval()
       else:
           self.model = None

   # ---------------------------------------------------------------------
   # Helper: require a trained model
   # ---------------------------------------------------------------------
   def _require_model(self):
       if self.model is None:
           raise RuntimeError(
               f"No trained model found in '{self.model_path}'.\n"
               f"Our folder must contain a trained checkpoint (pytorch_model.bin or model.safetensors).\n"
               f"Fix: call glm.train(...) first, or point model_path to the folder that already has the trained model."
           )

   # ---------------------------------------------------------------------
   # TRAINING
   # ---------------------------------------------------------------------
   def train(self, epochs: int = 30, batch_size: int = 16, lr: float = 2e-4):
       """
       Train a small BERT-style masked language model (MLM) on the FASTA sequences.

       What gets written into model_path:
         - pytorch_model.bin (or safetensors)
         - tokenizer files
         - trainer logs (optional)

       After training, self.model is ready for scoring/evaluation.
       """
       os.makedirs(self.model_path, exist_ok=True)

       data_collator = DataCollatorForLanguageModeling(
           tokenizer=self.tokenizer,
           mlm=True,
           mlm_probability=0.15
       )

       # small-ish model
       config = BertConfig(
           vocab_size=len(self.tokenizer.get_vocab()),
           hidden_size=256,
           num_hidden_layers=4,
           num_attention_heads=4,
           intermediate_size=512,
           max_position_embeddings=512,
           type_vocab_size=1
       )

       model = BertForMaskedLM(config)

       args = TrainingArguments(
           output_dir=self.model_path,
           overwrite_output_dir=True,
           num_train_epochs=epochs,
           per_device_train_batch_size=batch_size,
           save_steps=1000,
           logging_steps=50,
           report_to="none",
           learning_rate=lr,
           warmup_steps=100,
           dataloader_pin_memory=False,
           disable_tqdm=False
       )

       trainer = Trainer(
           model=model,
           args=args,
           train_dataset=self.dataset,
           data_collator=data_collator
       )

       print("Starting training...")
       trainer.train()

       trainer.save_model(self.model_path)
       self.tokenizer.save_pretrained(self.model_path)

       self.model = model.to(self.device)
       self.model.eval()

       print("Training complete! Saved model to:", self.model_path)

   # ---------------------------------------------------------------------
   # STEP 2 "model quality" score: MLM loss/perplexity on held-out sequences
   # ---------------------------------------------------------------------
   def evaluate_mlm_quality(
       self,
       n_samples: int = 500,
       mlm_probability: float = 0.15,
       seed: int = 0
   ) -> Dict[str, float]:
       """

       This is NOT a deletion score. This is a check:
       "Is the masked language model itself learning something reasonable?"

       We take a subset of sequences, apply random masking (like during training),
       and compute MLM loss. Lower loss (and lower perplexity) is better.

       This helps us compare:
         - our model vs another model
         - model versions across runs
       """
       self._require_model()
       random.seed(seed)

       data_collator = DataCollatorForLanguageModeling(
           tokenizer=self.tokenizer,
           mlm=True,
           mlm_probability=mlm_probability
       )

       # pick random sequences
       idxs = list(range(len(self.dataset)))
       random.shuffle(idxs)
       idxs = idxs[: min(n_samples, len(idxs))]

       losses = []

       self.model.eval()
       with torch.no_grad():
           for i in idxs:
               item = self.dataset[i]
               # collator expects a list of dicts
               batch = data_collator([item])
               batch = {k: v.to(self.device) for k, v in batch.items()}

               out = self.model(**batch)
               loss = float(out.loss.item())
               losses.append(loss)

       mean_loss = float(np.mean(losses)) if losses else float("nan")
       ppl = float(math.exp(mean_loss)) if (losses and mean_loss < 50) else float("nan")

       return {"mlm_loss": mean_loss, "perplexity": ppl, "n_samples": len(losses)}

   # ---------------------------------------------------------------------
   # Reconstruction helper (mask one position)
   # ---------------------------------------------------------------------
   def predict_position(self, sequence: str, position: int) -> np.ndarray:
       """
       Mask ONE position in the sequence and return the full probability distribution
       over the vocabulary at that masked position.
       """
       self._require_model()

       s = list(sequence)
       s[position] = "[MASK]"
       masked = "".join(s)

       inputs = self.tokenizer(masked, return_tensors="pt").to(self.device)

       # token positions: [CLS] + seq + [SEP]
       # our earlier logic used position+1, keep consistent
       mask_token_position = min(position + 1, inputs.input_ids.shape[1] - 2)

       with torch.no_grad():
           logits = self.model(**inputs).logits[0, mask_token_position]

       probs = torch.softmax(logits, dim=-1)
       return probs.detach().cpu().numpy()

   def get_full_reconstruction_probs(self, sequence_to_evaluate: str) -> np.ndarray:
       """
       For every position i:
         - mask i
         - ask the model for p(A), p(C), p(G), p(T), p('-')
       Returns an (L x 5) matrix.
       """
       self._require_model()

       seq_len = len(sequence_to_evaluate)
       char_to_id = {c: self.tokenizer.vocab[c] for c in self.relevant_chars}
       prob_matrix = np.zeros((seq_len, len(self.relevant_chars)), dtype=float)

       for pos in range(seq_len):
           probs = self.predict_position(sequence_to_evaluate, pos)
           for j, c in enumerate(self.relevant_chars):
               prob_matrix[pos, j] = probs[char_to_id[c]]

       return prob_matrix

   # ---------------------------------------------------------------------
   # NEW METHOD 1: FAST global delta-likelihood-ish score (no masking)
   # ---------------------------------------------------------------------
   def delta_likelihood_fast(
       self,
       reference_sequence: str,
       perturbed_sequence: str,
       region: Optional[Tuple[int, int]] = None
   ) -> Dict[str, Any]:
       """
       We want one quick number:
         "Does the model like the perturbed (deleted) sequence less than the reference?"

       We do:
         - run model once on ref
         - run model once on alt
         - at each position, take log prob of the observed token
         - sum over positions

       Then:
         delta = sum_logp(alt) - sum_logp(ref)

       Interpretation:
         delta << 0  : deletion made the sequence look much less plausible to the model
         delta ~ 0   : deletion barely changed global plausibility
       """
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
           ref_logits = self.model(**ref_inputs).logits[0]  # (token_len, vocab)
           alt_logits = self.model(**alt_inputs).logits[0]

       ref_logp = F.log_softmax(ref_logits, dim=-1)
       alt_logp = F.log_softmax(alt_logits, dim=-1)

       # string pos -> token pos (shift by 1 because of [CLS])
       tok_start = start + 1
       tok_end = end + 1

       # avoid [SEP]
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

   # ---------------------------------------------------------------------
   # NEW METHOD 2: Influence / probability shift score (mask targets)
   # ---------------------------------------------------------------------
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
       """

       This is the "Pedro-inspired" idea:

       If we introduce a deletion (ref -> alt),
       the model's predicted distribution at many positions can shift.

       We measure:
         For each target position j in some window:
           - mask j in ref, get p_ref(A,C,G,T,'-')
           - mask j in alt, get p_alt(A,C,G,T,'-')
           - compute how much the distribution moved

       metric options:
         - max_abs_logodds : max |log p_alt(v) - log p_ref(v)|
         - kl_ref_alt      : KL(p_ref || p_alt)
         - tv              : 0.5 * sum |p_alt - p_ref|

       reduce:
         - "mean" or "sum" across targets per query

       final influence_score:
         sum of query scores
       """
       self._require_model()

       if len(reference_sequence) != len(perturbed_sequence):
           raise ValueError("ref and alt must have the same length (use '-' for deletions).")

       # default queries: all positions that changed (where alt has '-' but ref doesn't)
       if query_positions is None:
           query_positions = [i for i, (a, b) in enumerate(zip(reference_sequence, perturbed_sequence)) if a != b]

       # targets: either full sequence or a window
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
       per_query = []

       for q in query_positions:
           per_target_scores = []
           for j in targets:
               if j == q:
                   continue
               p_ref = masked_probs(reference_sequence, j)
               p_alt = masked_probs(perturbed_sequence, j)
               per_target_scores.append(float(shift_score(p_ref, p_alt).item()))

           if len(per_target_scores) == 0:
               q_score = 0.0
           else:
               q_score = float(np.mean(per_target_scores)) if reduce == "mean" else float(np.sum(per_target_scores))

           total += q_score
           per_query.append({"query_pos": int(q), "score": q_score})

       return {
           "influence_score": float(total),
           "query_positions": query_positions,
           "target_window": (t0, t1),
           "metric": metric,
           "reduce": reduce,
           "per_query": per_query
       }

   # ---------------------------------------------------------------------
   # Tokenizer
   # ---------------------------------------------------------------------
   @staticmethod
   def create_tokenizer(save_dir: Optional[str] = None) -> PreTrainedTokenizerFast:
       """
       Minimal tokenizer for single-character DNA tokens plus '-'.

       If save_dir is provided, we store tokenizer.json there,
       so later GLMModel(...) can load it again.
       """
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

       # save a local tokenizer.json
       tokenizer_path = "tokenizer.json"
       if save_dir is not None:
           os.makedirs(save_dir, exist_ok=True)
           tokenizer_path = os.path.join(save_dir, "tokenizer.json")

       tok.save(tokenizer_path)

       return PreTrainedTokenizerFast(
           tokenizer_file=tokenizer_path,
           unk_token="[UNK]",
           sep_token="[SEP]",
           pad_token="[PAD]",
           cls_token="[CLS]",
           mask_token="[MASK]"
       )
