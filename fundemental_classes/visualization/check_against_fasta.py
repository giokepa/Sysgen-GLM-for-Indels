import os

def load_fasta(fasta_path):
    headers, seqs = [], []
    with open(fasta_path) as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    seqs.append("".join(seq))
                header = line
                seq = []
            else:
                seq.append(line)
        if header is not None:
            headers.append(header)
            seqs.append("".join(seq))
    return headers, seqs

def analyze_sequences(fasta_path, out_plots_dir):
    headers, seqs = load_fasta(fasta_path)

    plot_files = os.listdir(out_plots_dir)
    indices = sorted(
        {int(f.split("_")[1].split(".")[0]) for f in plot_files if f.startswith(("ref_", "alt_"))}
    )

    print(f"Found {len(indices)} ref/alt plot pairs\n")

    for i in indices:
        seq = seqs[i]
        header = headers[i]
        has_deletion = "-" in seq

        print(f"Index {i}")
        print(f"  Header: {header}")
        print(f"  Length: {len(seq)}")
        print(f"  Contains '-' in FASTA: {has_deletion}")
        print()
