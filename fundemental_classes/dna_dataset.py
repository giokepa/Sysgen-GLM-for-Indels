import torch
import re
from torch.utils.data import Dataset


class DNADataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_len=152):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fasta_file = fasta_file
        # This now stores (header, sequence) tuples
        self.records = self._read_fasta()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        header, seq = self.records[idx]

        # Tokenize the DNA sequence
        encoded = self.tokenizer(
            seq,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoded.items()}

        # Parse motif positions from the header
        pos_a, pos_b = self.parse_header(header)

        # Add positions to the item dictionary for the collator to use
        item['pos_a'] = pos_a if pos_a is not None else -1
        item['pos_b'] = pos_b if pos_b is not None else -1

        return item

    def _read_fasta(self):
        records = []
        with open(self.fasta_file, "r") as f:
            current_header = ""
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        records.append((current_header, current_seq))
                    current_header = line
                    current_seq = ""
                else:
                    current_seq += line
            if current_seq:  # Add the last record
                records.append((current_header, current_seq))
        return records

    @staticmethod
    def parse_header(header_str: str):
        pos_a_match = re.search(r"posAmotif=([\d]+|None)", header_str)
        pos_b_match = re.search(r"posBmotif=([\d]+|None)", header_str)

        start_a = int(pos_a_match.group(1)) if pos_a_match and pos_a_match.group(1) != "None" else None
        start_b = int(pos_b_match.group(1)) if pos_b_match and pos_b_match.group(1) != "None" else None

        return start_a, start_b
