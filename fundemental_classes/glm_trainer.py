from transformers import *
from tokenizers import Tokenizer, models, pre_tokenizers
from fundemental_classes.dna_dataset import DNADataset
import os


class GLMTrainer:
    def __init__(self, model_path="./dna_bert_final", max_length=122):

        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = self.create_tokenizer()

    def train(self, epochs=30, batch_size=16, lr=2e-4):
        os.makedirs(self.model_path, exist_ok=True)

        dataset = DNADataset.create_dataset(self.tokenizer, self.max_length)
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
            disable_tqdm=False, # TURN THIS TRUE IF YOU WANNA NOT HAVE FANCY LOADING BAR
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print("Starting Training")

        trainer.train()

        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print("Training complete!")

        self.plot_training_curves(trainer.state.log_history)
        return model

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
