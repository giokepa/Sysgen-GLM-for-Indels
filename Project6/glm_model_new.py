import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling, BertConfig, TrainingArguments, Trainer
from tokenizers import Tokenizer, models, pre_tokenizers
import os
import torch.nn.functional as F

from dna_dataset import DNADataset


class GLMModel:
    def __init__(self, model_path, fasta_file, max_seq_length=122):
        self.model_path = model_path

        self.model = None if not os.path.exists(model_path) else BertForMaskedLM.from_pretrained(model_path)
        self.tokenizer = self.create_tokenizer() if not os.path.exists(model_path) else PreTrainedTokenizerFast.from_pretrained(model_path)
        self.max_length = max_seq_length
        self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)
        self.relevant_chars = ['A', 'C', 'G', 'T', '-']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def train(self, epochs=3, batch_size=16, lr=2e-4):
        """
        epochs default lowered to 3 for local testing.
        Raise later.
        """
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
            report_to="none",
            learning_rate=lr,
            warmup_steps=100,
            dataloader_pin_memory=False,
            disable_tqdm=False,
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

        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def predict_position(self, sequence, position):
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

    # =====================================================================
    # NEW METHOD 1: FAST Δ-likelihood-ish score WITHOUT masking
    # =====================================================================
    def delta_likelihood_fast(self, reference_sequence, perturbed_sequence, region=None):
        if len(reference_sequence) != len(perturbed_sequence):
            raise ValueError(
                "reference_sequence and perturbed_sequence must have the same length. "
                "For deletions, encode them as '-' so lengths stay aligned."
            )

        if region is None:
            start = 0
            end = len(reference_sequence)
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
        }

    # =====================================================================
    # NEW METHOD 2: Influence / dependency-like score (mask TARGET positions)
    # =====================================================================
    def influence_probability_shift(self,
                                   reference_sequence,
                                   perturbed_sequence,
                                   query_positions=None,
                                   target_window=None,
                                   metric="max_abs_logodds",
                                   reduce="mean",
                                   eps=1e-9):

        if len(reference_sequence) != len(perturbed_sequence):
            raise ValueError(
                "reference_sequence and perturbed_sequence must have the same length."
            )

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

        relevant_ids = [self.tokenizer.vocab[c] for c in self.relevant_chars]
        relevant_ids = torch.tensor(relevant_ids, device=self.device)

        def masked_probs_over_acgt_gap(seq, j):
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

        per_query_scores = []
        for q in query_positions:
            q = int(q)
            vals = []
            for j in targets:
                if j == q:
                    continue
                p_ref = masked_probs_over_acgt_gap(reference_sequence, j)
                p_alt = masked_probs_over_acgt_gap(perturbed_sequence, j)
                vals.append(float(score_shift(p_ref, p_alt).item()))

            if len(vals) == 0:
                per_query_scores.append(0.0)
            else:
                per_query_scores.append(float(np.mean(vals)) if reduce == "mean" else float(np.sum(vals)))

        total = float(np.sum(per_query_scores))
        return {
            "influence_score": total,
            "n_queries": len(query_positions),
            "target_window": (t0, t1),
            "metric": metric,
            "reduce": reduce
        }
