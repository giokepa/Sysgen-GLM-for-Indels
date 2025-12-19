import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from transformers import *
from tokenizers import Tokenizer, models, pre_tokenizers
from fundemental_classes.dna_dataset import DNADataset
import os
import torch.nn.functional as F


class GLMModel:
    def __init__(self, model_path, fasta_file, max_seq_length=122):
        self.model_path = model_path

        self.model = None if not os.path.exists(model_path) else BertForMaskedLM.from_pretrained(model_path)
        self.tokenizer = self.create_tokenizer() if not os.path.exists(model_path) else PreTrainedTokenizerFast.from_pretrained(model_path)
        self.max_length = max_seq_length
        self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)
        self.relevant_chars = ['A', 'C', 'G', 'T', '-']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs=30, batch_size=16, lr=2e-4):
        os.makedirs(self.model_path, exist_ok=True)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        config = BertConfig(
            vocab_size=10, hidden_size=256, num_hidden_layers=4,
            num_attention_heads=4, intermediate_size=512,
            max_position_embeddings=512, type_vocab_size=1
        )
        model = BertForMaskedLM(config)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            logging_steps=10,
            log_level="info",
            logging_first_step=True,
            report_to="all",
            learning_rate=lr,
            warmup_steps=100,
            dataloader_pin_memory=False,
            disable_tqdm=False, # could also be set to True here for fancy bars?
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
        )

        print("Starting Training")

        trainer.train()

        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print("Training complete!")

        self.plot_training_curves(trainer.state.log_history)

        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def predict_position(self, sequence, position):
        seq_len = len(sequence)

        input_seq = list(sequence)
        input_seq[position] = '[MASK]'
        input_str = ''.join(input_seq)

        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.device)

        mask_token_position = min(position + 1, inputs.input_ids.shape[1] - 2)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, mask_token_position]

        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def get_full_reconstruction_probs(self, sequence_to_evaluate):
        seq_len = len(sequence_to_evaluate)
        char_to_id = {c: self.tokenizer.vocab[c] for c in self.relevant_chars}

        prob_matrix = np.zeros((seq_len, len(self.relevant_chars)))

        for pos in range(seq_len):
            probs = self.predict_position(sequence_to_evaluate, pos)

            for i, char in enumerate(self.relevant_chars):
                prob_matrix[pos, i] = probs[char_to_id[char]]

        return prob_matrix

    @staticmethod
    def plot_training_curves(log_history):
        import matplotlib.pyplot as plt

        if not log_history or len(log_history) < 2:
            print("No training logs available for plotting. Something bad happened")
            print(f"Log history length: {len(log_history)}")
            return

        losses = []
        lrs = []
        steps = []

        for i, log in enumerate(log_history):
            if 'loss' in log:
                losses.append(log['loss'])
                steps.append(i)
            if 'learning_rate' in log:
                lrs.append(log['learning_rate'])

        if not losses:
            print("No loss data found in logs.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(steps[:len(losses)], losses, 'b-', linewidth=2, marker='o')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Logging Steps')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.grid(True, alpha=0.3)

        if lrs:
            ax2.plot(steps[:len(lrs)], lrs, 'r-', linewidth=2, marker='s')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Logging Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No LR data\nlogged', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Learning Rate (No Data)')

        plt.tight_layout()
        plt.show()

        print(f"Final loss: {losses[-1]:.4f}")

    @staticmethod
    def create_tokenizer():
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
            "-": 9
        }
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
        tokenizer.save("temp_tokenizer.json")

        return PreTrainedTokenizerFast(
            tokenizer_file="temp_tokenizer.json",
            unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
            cls_token="[CLS]", mask_token="[MASK]"
        )
############################################################################################################################
    # NEW EVALUATION METHOD #1 (FAST):
    # Probability-shift / Δ-likelihood-ish score WITHOUT masking
############################################################################################################################
    def delta_likelihood_fast(self, reference_sequence, perturbed_sequence, region=None):
        """
        We want a quick number that says:
            "Did the model like the perturbed sequence less than the reference sequence?"

        So we do this:

        1) Run the model once on the reference sequence
        2) Run the model once on the perturbed sequence
        3) For each position i, take log(probability of the token that is actually there at i)
        4) Sum these log-probabilities over a region (or the whole sequence)
        5) Return:
                delta = sum_logp(perturbed) - sum_logp(reference)

        - For a MASKED LM (BERT), this is NOT a “true likelihood” because we are not masking.
        - But it is *fast* and often useful as a rough “screening” score.

        Requirements:
        - reference_sequence and perturbed_sequence should be the same length.

        """
        import torch.nn.functional as F

        if len(reference_sequence) != len(perturbed_sequence):
            raise ValueError(
                "reference_sequence and perturbed_sequence must have the same length. "
                "If we have deletions, encode them as '-' so lengths stay aligned."
            )

        # Decide which part of the sequence we score
        if region is None:
            start = 0
            end = len(reference_sequence)
        else:
            start, end = int(region[0]), int(region[1])
            start = max(0, start)
            end = min(len(reference_sequence), end)
            if end <= start:
                raise ValueError("region must satisfy end > start (or end >= start)")

        # Tokenize both sequences
        ref_inputs = self.tokenizer(reference_sequence, return_tensors="pt").to(self.device)
        alt_inputs = self.tokenizer(perturbed_sequence, return_tensors="pt").to(self.device)

        # Forward pass once each
        with torch.no_grad():
            ref_logits = self.model(**ref_inputs).logits[0]  # (token_len, vocab)
            alt_logits = self.model(**alt_inputs).logits[0]

        # Convert to log-probabilities
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        alt_logp = F.log_softmax(alt_logits, dim=-1)

        # Map "string indices" -> "token indices"
        # because token 0 is [CLS], and last token is [SEP]
        # So position i in the string usually corresponds to token i+1
        tok_start = start + 1
        tok_end = end + 1

        # Also: make sure we do not touch [SEP]
        # (the last usable token is token_len-2)
        max_token_index = ref_inputs.input_ids.shape[1] - 2
        tok_start = max(1, min(tok_start, max_token_index))
        tok_end = max(tok_start, min(tok_end, max_token_index + 1))

        # Gather the log-prob for the actually observed token at each position
        ref_ids = ref_inputs.input_ids[0]
        alt_ids = alt_inputs.input_ids[0]

        idx = torch.arange(tok_start, tok_end, device=self.device)

        ref_per_pos = ref_logp[idx, ref_ids[idx]]
        alt_per_pos = alt_logp[idx, alt_ids[idx]]

        ref_sum = float(ref_per_pos.sum().item())
        alt_sum = float(alt_per_pos.sum().item())
        delta = alt_sum - ref_sum

        return {
            "delta": delta,
            "reference_sum": ref_sum,
            "perturbed_sum": alt_sum,
            "region": (start, end),
            "per_pos_reference": ref_per_pos.detach().cpu().numpy(),
            "per_pos_perturbed": alt_per_pos.detach().cpu().numpy()
        }

    # =====================================================================
    # NEW EVALUATION METHOD #2:
    # Influence / dependency-like score (mask TARGET positions only)
    # =====================================================================
    def influence_probability_shift(self,
                                   reference_sequence,
                                   perturbed_sequence,
                                   query_positions=None,
                                   target_window=None,
                                   metric="max_abs_logodds",
                                   reduce="mean",
                                   eps=1e-9):
        """

        Pedro's key idea is basically:
            "If I change nucleotide i (query), the model's beliefs about other positions j (targets)
             can shift. That shift is a signal of functional coupling."

        Here we implement a simple, practical version for our setup:

        - We compare REFERENCE vs PERTURBED (with indel encoded by '-').
        - We decide which positions are "queries" (the changed positions).
        - We pick a window of "targets" to look at (to keep it fast).
        - For each target j:
              1) mask j in reference -> get probs over A/C/G/T/'-'
              2) mask j in perturbed  -> get probs over A/C/G/T/'-'
              3) quantify "how much the distribution moved"

        How we score the movement (metric):
          - "max_abs_logodds": max_v |log p_alt(v) - log p_ref(v)|
          - "kl_ref_alt": KL(p_ref || p_alt)
          - "tv": total variation distance (0.5*sum|p_alt - p_ref|)

        Then we aggregate:
          - reduce="mean" or "sum" over targets for each query
          - then sum over all queries
        """

        if len(reference_sequence) != len(perturbed_sequence):
            raise ValueError(
                "reference_sequence and perturbed_sequence must have the same length. "
                "Use '-' for deletions so the sequences stay aligned."
            )

        # If user did not provide query positions, we just take all mismatches
        if query_positions is None:
            query_positions = [i for i, (a, b) in enumerate(zip(reference_sequence, perturbed_sequence)) if a != b]

        # Decide which target positions we look at
        if target_window is None:
            t0, t1 = 0, len(reference_sequence)
        else:
            t0, t1 = int(target_window[0]), int(target_window[1])
            t0 = max(0, t0)
            t1 = min(len(reference_sequence), t1)
            if t1 <= t0:
                raise ValueError("target_window must satisfy end > start")

        targets = list(range(t0, t1))

        # We only care about probs for these tokens
        relevant_ids = [self.tokenizer.vocab[c] for c in self.relevant_chars]
        relevant_ids = torch.tensor(relevant_ids, device=self.device)

        def masked_probs_over_acgt_gap(seq, j):
            """
            Mask position j, run model, return normalized probs over [A,C,G,T,'-'].
            """
            s = list(seq)
            s[j] = "[MASK]"
            masked = "".join(s)

            inputs = self.tokenizer(masked, return_tensors="pt").to(self.device)

            mask_token_position = min(j + 1, inputs.input_ids.shape[1] - 2)

            with torch.no_grad():
                logits = self.model(**inputs).logits[0, mask_token_position]

            p = F.softmax(logits, dim=-1)[relevant_ids]
            p = p / (p.sum() + eps)
            return p

        def score_shift(p_ref, p_alt):
            if metric == "max_abs_logodds":
                return torch.max(torch.abs(torch.log(p_alt + eps) - torch.log(p_ref + eps)))
            if metric == "kl_ref_alt":
                return torch.sum(p_ref * (torch.log(p_ref + eps) - torch.log(p_alt + eps)))
            if metric == "tv":
                return 0.5 * torch.sum(torch.abs(p_alt - p_ref))
            raise ValueError(f"Unknown metric: {metric}")

        total_score = 0.0
        per_query_breakdown = []

        for q in query_positions:
            q = int(q)

            # For a given query position q, we measure shifts across targets.
            # We usually skip j == q (not necessary, but it avoids trivial self-effects).
            per_target_scores = []

            for j in targets:
                if j == q:
                    continue

                p_ref = masked_probs_over_acgt_gap(reference_sequence, j)
                p_alt = masked_probs_over_acgt_gap(perturbed_sequence, j)

                per_target_scores.append(float(score_shift(p_ref, p_alt).item()))

            per_target_scores = np.array(per_target_scores, dtype=float)

            if per_target_scores.size == 0:
                q_score = 0.0
            else:
                q_score = float(per_target_scores.mean()) if reduce == "mean" else float(per_target_scores.sum())

            total_score += q_score
            per_query_breakdown.append({
                "query_pos": q,
                "ref_char": reference_sequence[q],
                "alt_char": perturbed_sequence[q],
                "score": q_score,
                "per_target_scores": per_target_scores
            })

        return {
            "influence_score": float(total_score),
            "metric": metric,
            "reduce": reduce,
            "query_positions": query_positions,
            "target_window": (t0, t1),
            "per_query": per_query_breakdown
        }

