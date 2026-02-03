import torch
import re
import numpy as np


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file, tokenizer, max_len, add_special_tokens=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fasta_file = fasta_file
        self.add_special_tokens = add_special_tokens
        self.seqs = self.create_sequences()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        encoded = self.tokenizer(
            seq,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def create_sequences(self):
        sequences = []
        with open(self.fasta_file, 'r') as f:
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)

        return sequences
    def get_raw_sequence(self, idx):
        with open(self.fasta_file, 'r') as f:
            current_idx = -1
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        current_idx += 1
                        if current_idx == idx:
                            return header, current_seq
                    header = line
                    current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                current_idx += 1
                if current_idx == idx:
                    return header, current_seq
        raise IndexError("Index out of range")
    @staticmethod
    def parse_motif_positions(header):
        motif_positions = {}
        parts = header.split('|')
        for part in parts:
            if 'posAmotif=' in part:
                pos_str = part.split('=')[1]
                motif_positions['A'] = [int(x) for x in pos_str.split(',')]
            elif 'posBmotif=' in part:
                pos_str = part.split('=')[1]
                motif_positions['B'] = [int(x) for x in pos_str.split(',')]

        return motif_positions
