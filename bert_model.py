import torch
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

model_path = "./dna_bert_final"
model = BertForMaskedLM.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)


def parse_header(header_str):
    pos_a = re.search(r"posAmotif=(\d+|None)", header_str)
    pos_b = re.search(r"posBmotif=(\d+|None)", header_str)

    # Defaults
    start_a, start_b = None, None

    if pos_a and pos_a.group(1) != 'None':
        start_a = int(pos_a.group(1))
    if pos_b and pos_b.group(1) != 'None':
        start_b = int(pos_b.group(1))

    return start_a, start_b


def plot_compressed_logo(header, sequence, motif_length=10):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1).numpy()

    relevant_chars = ['A', 'C', 'G', 'T']
    relevant_ids = [tokenizer.vocab[c] for c in relevant_chars]
    prob_matrix = probs[:, relevant_ids]

    df = pd.DataFrame(prob_matrix, columns=relevant_chars)
    ic_df = logomaker.transform_matrix(df, from_type='probability', to_type='information')

    fig, ax = plt.subplots(figsize=(20, 2.5))

    logo = logomaker.Logo(ic_df,
                          ax=ax,
                          color_scheme='classic',
                          vpad=0.05,
                          width=1.0)
    logo.style_spines(spines=['top', 'right', 'left'], visible=False)
    ax.spines['left'].set_visible(True)
    ax.set_ylabel("Bits", fontsize=10)
    ax.set_xticks([])
    start_a, start_b = parse_header(header)
    seq_len = len(sequence)

    y_line = -0.2
    ax.plot([0, seq_len], [y_line, y_line], color='black', lw=1, clip_on=False)

    def draw_motif_box(start_idx, label):
        if start_idx is None: return

        width = motif_length

        rect = patches.Rectangle((start_idx, y_line - 0.5), width, 0.4,
                                 linewidth=1, edgecolor='black', facecolor='white', clip_on=False)
        ax.add_patch(rect)

        ax.text(start_idx + width / 2, y_line - 0.3, label,
                ha='center', va='center', fontsize=9, fontweight='bold', clip_on=False)

    draw_motif_box(start_a, "Motif A")
    draw_motif_box(start_b, "Motif B")

    plt.title(header.split('|')[0], fontsize=10)  # Add seq ID as title
    plt.tight_layout()
    plt.show()


full_header = ">seq0005|label=both|posAmotif=41|posBmotif=98|gaplength=50"
full_seq = "ATCAACTGTTAGGTAACCATTTGCCCGCCAACTACAAGTACATATTCACATCACGAATCGGGCGGAAAACCTGAGACCGACTGATGCGGATGTGGTAGGTACTGCGCGCGACGTGCCGGC"

plot_compressed_logo(full_header, full_seq, motif_length=7)  # Adjust motif_length as needed
