import torch
import re


class DNADataset(torch.utils.data.Dataset):
    """
    Reads a FASTA file.
    Keeps: 
      - self.headers: list[str]
      - self.seqs:    list[str]
    """
    def __init__(self, fasta_file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fasta_file = fasta_file
        self.headers, self.seqs = self._read_fasta()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        encoded = self.tokenizer(
            seq,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def _read_fasta(self):
        headers = []
        seqs = []
        current_seq = []
        current_header = None

        with open(self.fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    # flush previous
                    if current_header is not None:
                        headers.append(current_header)
                        seqs.append("".join(current_seq))
                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line)

        if current_header is not None:
            headers.append(current_header)
            seqs.append("".join(current_seq))

        return headers, seqs

    @staticmethod
    def parse_header(header_str):
        pos_a = re.search(r"posAmotif=(\d+|None)", header_str)
        pos_b = re.search(r"posBmotif=(\d+|None)", header_str)

        start_a = int(pos_a.group(1)) if pos_a and pos_a.group(1) != "None" else None
        start_b = int(pos_b.group(1)) if pos_b and pos_b.group(1) != "None" else None

        return start_a, start_b
