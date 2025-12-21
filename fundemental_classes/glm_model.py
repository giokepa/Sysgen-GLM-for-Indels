from dataclasses import dataclass

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from fundemental_classes.dna_dataset import DNADataset
import os
from transformers import (
    BertForMaskedLM,
    BertConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


@dataclass
class MotifAwareCollator:
    tokenizer: PreTrainedTokenizerFast
    motif_prob: float = 0.8  # High probability to mask motifs
    bg_prob: float = 0.1  # Low probability to mask background

    def __call__(self, examples):
        input_ids = torch.stack([e['input_ids'] for e in examples])
        attention_mask = torch.stack([e['attention_mask'] for e in examples])

        # Create a probability matrix for masking
        prob_matrix = torch.full(input_ids.shape, self.bg_prob)

        for i, ex in enumerate(examples):
            pos_a = ex.get("pos_a", -1)
            pos_b = ex.get("pos_b", -1)

            # Apply higher probability to motif regions (offset by 1 for CLS token)
            if pos_a != -1:
                prob_matrix[i, pos_a + 1: pos_a + 8] = self.motif_prob
            if pos_b != -1:
                prob_matrix[i, pos_b + 1: pos_b + 8] = self.motif_prob

        # Decide which tokens to mask based on the probability matrix
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # Never mask special tokens
        special_tokens_mask = torch.tensor([
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ], dtype=torch.bool)
        masked_indices[special_tokens_mask] = False

        # Prepare labels: -100 for non-masked tokens
        labels = input_ids.clone()
        labels[~masked_indices] = -100

        # Apply 80/10/10 strategy to masked tokens
        # 80% become [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% become random
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
class GLMModel:
    def __init__(self, model_path, fasta_file, max_seq_length=122):
        self.model_path = model_path

        self.model = None if not os.path.exists(model_path) else BertForMaskedLM.from_pretrained(model_path)
        self.tokenizer = self.create_tokenizer() if not os.path.exists(
            model_path) else PreTrainedTokenizerFast.from_pretrained(model_path)
        self.max_length = max_seq_length
        self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)
        self.relevant_chars = ['A', 'C', 'G', 'T', '-']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs=30, batch_size=16, lr=2e-4):
        os.makedirs(self.model_path, exist_ok=True)

        config = BertConfig(
            vocab_size=10,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512,
            type_vocab_size=1,
        )

        model = BertForMaskedLM(config)
        model.to(self.device)

        data_collator = MotifAwareCollator(
            tokenizer=self.tokenizer, motif_prob=0.8, bg_prob=0.1
        )

        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            gradient_accumulation_steps=2,
            save_steps=5000,
            logging_steps=500,
            log_level="error",
            logging_first_step=True,
            report_to="all",
            warmup_steps=1000,
            disable_tqdm=False,  # TURN THIS TRUE IF YOU WANNA NOT HAVE FANCY LOADING BAR
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
        inputs = self.tokenizer(sequence_to_evaluate, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        seq_len = input_ids.shape[1]

        num_valid_tokens = seq_len - 2
        prob_matrix = np.zeros((num_valid_tokens, len(self.relevant_chars)))

        char_to_id = {c: self.tokenizer.vocab[c] for c in self.relevant_chars}
        mask_token_id = self.tokenizer.mask_token_id  # This is ID 4

        for i in range(num_valid_tokens):
            token_idx = i + 1

            masked_input_ids = input_ids.clone()
            masked_input_ids[0, token_idx] = mask_token_id

            with torch.no_grad():
                outputs = self.model(input_ids=masked_input_ids)
                logits = outputs.logits[0, token_idx]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for char_idx, char in enumerate(self.relevant_chars):
                if char in char_to_id:
                    prob_matrix[i, char_idx] = probs[char_to_id[char]]

        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        prob_matrix = np.divide(prob_matrix, row_sums, out=np.zeros_like(prob_matrix), where=row_sums != 0)

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
