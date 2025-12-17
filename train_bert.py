import torch
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors


def create_dna_tokenizer():
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
    tokenizer.post_processor = processors.BertProcessing(
        sep=("|SEP|", 3),
        cls=("|CLS|", 2),
    )

    from transformers import PreTrainedTokenizerFast

    tokenizer.save("dna_tokenizer.json")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="dna_tokenizer.json",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    return hf_tokenizer


class DNADataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        with open(fasta_file, "r") as f:
            lines = f.readlines()
            current_seq = ""
            for line in lines:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        self.sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                self.sequences.append(current_seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


def train_model():
    fasta_path = "simulated_sequences/augumented_sequence_size100_length120_deletions20_nodeletionseq0.25.fasta"
    tokenizer = create_dna_tokenizer()

    dataset = DNADataset(fasta_path, tokenizer, max_length=122)

    config = BertConfig(
        vocab_size=10,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        type_vocab_size=1,
    )

    model = BertForMaskedLM(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Mask 15% of tokens
    )

    training_args = TrainingArguments(
        output_dir="./dna_bert_output",
        overwrite_output_dir=True,
        num_train_epochs=50,  # Train longer since dataset is small?
        per_device_train_batch_size=16,
        save_steps=100,
        logging_steps=10,
        prediction_loss_only=True,
        learning_rate=1e-4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    trainer.save_model("./dna_bert_final")
    tokenizer.save_pretrained("./dna_bert_final")


if __name__ == "__main__":
    train_model()
