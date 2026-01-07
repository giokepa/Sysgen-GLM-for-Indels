import numpy as np
import torch
import json
import shutil
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
    def __init__(self, model_path, fasta_file, max_seq_length=122, force_retrain=False):
        self.model_path = model_path
        self.meta_path = os.path.join(model_path, "training_metadata.json")
        self.max_length = max_seq_length
        self.relevant_chars = ['A', 'C', 'G', 'T', '-']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if force_retrain:
            print(f"force_retrain=True: Clearing all model files")
            self._cleanup_all()

        load_success = False
        if os.path.exists(model_path) and not force_retrain:
            load_success = self._try_load_existing_model(fasta_file, max_seq_length)

        if not load_success:
            print("Initializing fresh model")
            self.tokenizer = self.create_tokenizer()
            self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)
            self.model = None
            print("No trained model loaded. Call train() to train the model.")

    def _cleanup_all(self):
        if os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
                print(f"Removed directory: {self.model_path}")
            except Exception as e:
                print(f"Could not remove {self.model_path}: {e}")

        if os.path.exists("temp_tokenizer.json"):
            try:
                os.remove("temp_tokenizer.json")
                print(f"Removed temp_tokenizer.json")
            except Exception as e:
                print(f"Could not remove temp_tokenizer.json: {e}")

    def _try_load_existing_model(self, fasta_file, max_seq_length):
        try:
            print(f"Checking for existing trained model in {self.model_path}")

            metadata = self._load_metadata()
            if not metadata.get("trained", False):
                print("No trained model found (metadata missing or trained=False)")
                return False

            required_files = {
                "config.json": os.path.join(self.model_path, "config.json"),
                "model weights": os.path.join(self.model_path, "pytorch_model.bin"),
                "tokenizer": os.path.join(self.model_path, "tokenizer.json")
            }

            if not os.path.exists(required_files["model weights"]):
                required_files["model weights"] = os.path.join(self.model_path, "model.safetensors")

            missing = []
            for name, path in required_files.items():
                if not os.path.exists(path):
                    missing.append(name)

            if missing:
                print(f"Model directory incomplete. Missing: {', '.join(missing)}")
                return False

            print("Loading tokenizer...")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_path)
            print("Tokenizer loaded")

            print("Loading dataset...")
            self.dataset = DNADataset(fasta_file, self.tokenizer, max_seq_length)
            print("Dataset loaded")

            print("Loading model...")
            self.model = BertForMaskedLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully!")
            print(f"Trained for {metadata.get('epochs_completed', 'N/A')} epochs")
            print(f"Final validation loss: {metadata.get('final_val_loss', 'N/A'):.4f}")

            return True

        except Exception as e:
            print(f"\nError loading model: {type(e).__name__}")
            print(f"Details: {str(e)[:200]}")
            print(f"\n🗑Auto-cleanup: Removing corrupted model files...")
            self._cleanup_all()
            print(f"Cleanup complete. Will initialize fresh model.\n")
            return False

    def _load_metadata(self):
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_metadata(self, epochs, final_train_loss, final_val_loss):
        import datetime
        metadata = {
            "trained": True,
            "epochs_completed": epochs,
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "training_date": datetime.datetime.now().isoformat(),
            "dataset_size": len(self.dataset)
        }
        os.makedirs(self.model_path, exist_ok=True)
        with open(self.meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {self.meta_path}")

    def train(self, epochs=30, batch_size=16, lr=2e-4, validation_split=0.2):
        os.makedirs(self.model_path, exist_ok=True)

        g = torch.Generator().manual_seed(727)
        n_total = len(self.dataset)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(self.dataset, [n_train, n_val], generator=g)

        print(f"Dataset split: {n_train} training, {n_val} validation")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
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
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            learning_rate=lr,
            warmup_steps=100,
            dataloader_pin_memory=False,
            disable_tqdm=False,
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

        print("Saving model and tokenizer...")
        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print("Training complete")

        log_history = trainer.state.log_history
        train_losses = [log['loss'] for log in log_history if 'loss' in log]
        eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]

        final_train_loss = train_losses[-1] if train_losses else float('nan')
        final_val_loss = eval_losses[-1] if eval_losses else float('nan')

        self._save_metadata(epochs, final_train_loss, final_val_loss)

        self.plot_training_and_validation_curves(log_history,
                                                 save_path=os.path.join(self.model_path, "training_curves.png"))

        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def predict_position(self, sequence, position, debug=False):
        if self.model is None:
            raise RuntimeError("Model not loaded or trained. Call train() first or load a trained model.")

        if position < 0 or position >= len(sequence):
            raise ValueError(f"Position {position} out of range for sequence of length {len(sequence)}")

        input_seq = list(sequence)
        original_char = input_seq[position]
        input_seq[position] = '[MASK]'
        input_str = ''.join(input_seq)

        inputs = self.tokenizer(
            input_str,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True
        ).to(self.device)

        # Find mask position
        mask_token_id = self.tokenizer.mask_token_id
        input_ids = inputs.input_ids[0]
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            raise RuntimeError(f"[MASK] token not found in tokenized sequence.")
        if len(mask_positions) > 1:
            raise RuntimeError(f"Multiple [MASK] tokens found.")

        mask_token_position = mask_positions[0].item()

        if debug:
            print(f"\n--- Debug Info for position {position} ---")
            print(f"Original sequence: {sequence}")
            print(f"Masked sequence:   {input_str}")
            print(f"Original char at pos {position}: '{original_char}'")
            print(f"Token IDs: {input_ids.tolist()}")
            print(f"Attention mask: {inputs.attention_mask[0].tolist()}")
            print(f"[MASK] token ID: {mask_token_id}")
            print(f"[MASK] found at token position: {mask_token_position}")

            tokens = [self.tokenizer.decode([tid]) for tid in input_ids]
            print(f"Decoded tokens: {tokens}")

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            logits = outputs.logits[0, mask_token_position]
            probs = torch.softmax(logits, dim=-1)

        if debug:
            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)
            print(f"\nTop {top_k} predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = self.tokenizer.decode([idx.item()])
                print(f"  {i + 1}. '{token}' (ID: {idx.item()}): {prob.item():.4f}")

            print(f"\nDNA character probabilities:")
            for char in self.relevant_chars:
                char_id = self.tokenizer.vocab[char]
                print(f"  {char}: {probs[char_id].item():.4f}")
            print(f"---\n")

        return probs.cpu().numpy()

    def get_full_reconstruction_probs(self, sequence_to_evaluate, debug=False):
        if self.model is None:
            raise RuntimeError("Model not loaded or trained. Call train() first or load a trained model.")

        seq_len = len(sequence_to_evaluate)

        valid_chars = set(self.relevant_chars)
        invalid_chars = set(sequence_to_evaluate) - valid_chars
        if invalid_chars:
            raise ValueError(f"Sequence contains invalid characters: {invalid_chars}. "
                             f"Valid characters: {valid_chars}")

        char_to_id = {c: self.tokenizer.vocab[c] for c in self.relevant_chars}

        prob_matrix = np.zeros((seq_len, len(self.relevant_chars)))

        for pos in range(seq_len):
            try:
                probs = self.predict_position(sequence_to_evaluate, pos, debug)

                for i, char in enumerate(self.relevant_chars):
                    prob_matrix[pos, i] = probs[char_to_id[char]]

            except Exception as e:
                print(f"Warning: Error predicting position {pos}: {e}")
                prob_matrix[pos, :] = 1.0 / len(self.relevant_chars)

        return prob_matrix

    @staticmethod
    def plot_training_and_validation_curves(log_history, save_path=None):
        import matplotlib.pyplot as plt

        if not log_history or len(log_history) < 2:
            print("No training logs available for plotting")
            print(f"Log history length: {len(log_history)}")
            return

        train_losses = []
        train_steps = []
        eval_losses = []
        eval_steps = []
        lrs = []
        lr_steps = []

        for i, log in enumerate(log_history):
            if 'loss' in log:
                train_losses.append(log['loss'])
                train_steps.append(log.get('step', i))
            if 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])
                eval_steps.append(log.get('step', i))
            if 'learning_rate' in log:
                lrs.append(log['learning_rate'])
                lr_steps.append(log.get('step', i))

        if not train_losses:
            print("No loss data found in logs")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.plot(train_steps, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss',
                 alpha=0.7)
        if eval_losses:
            ax1.plot(eval_steps, eval_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss',
                     alpha=0.7)
        ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Training Steps', fontsize=10)
        ax1.set_ylabel('Cross-Entropy Loss', fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2 = axes[1]
        if lrs:
            ax2.plot(lr_steps, lrs, 'g-', linewidth=2, marker='d', markersize=4)
            ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Training Steps', fontsize=10)
            ax2.set_ylabel('Learning Rate', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No LR data logged', ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            ax2.set_title('Learning Rate (No Data)', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")

        plt.show()

        print(f"\nTraining Summary:")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        if eval_losses:
            print(f"Final validation loss: {eval_losses[-1]:.4f}")
            print(f"Best validation loss: {min(eval_losses):.4f}")

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
