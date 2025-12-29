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

from torch.utils.data import random_split

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

        # split the data into train and validation sets (80% train, 20% val)
        g = torch.Generator().manual_seed(727)

        n_total = len(self.dataset)
        n_val = int(n_total * 0.2)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(self.dataset, [n_train, n_val], generator=g)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15 # default: mask_replace_prob=0.8, random_replace_prob=0.1
        )

        config = BertConfig(
        vocab_size=10,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1536,
        max_position_embeddings=512,
        type_vocab_size=1,
        )
        model = BertForMaskedLM(config)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_steps=500,
            logging_steps=200,

            eval_steps=500,
            log_level="info",
            logging_first_step=True,
            eval_strategy="steps",
            save_strategy="steps",
            report_to="all",
            load_best_model_at_end=True, # take best model instead of last model to avoid overfitting
            metric_for_best_model="eval_loss",
            learning_rate=lr,
            warmup_steps=100,
            dataloader_pin_memory=False,
            disable_tqdm=False,  # TURN THIS TRUE IF YOU WANNA NOT HAVE FANCY LOADING BAR
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
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
