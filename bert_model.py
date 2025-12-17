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

    relevant_chars = ['A', 'C', 'G', 'T', '-']
    relevant_ids = [tokenizer.vocab[c] for c in relevant_chars]
    prob_matrix = probs[:, relevant_ids]

    df = pd.DataFrame(prob_matrix, columns=relevant_chars)
    ic_df = logomaker.transform_matrix(df, from_type='probability', to_type='information')

    fig, ax = plt.subplots(figsize=(15, 2.5))

    dna_colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', '-': 'black'}

    logo = logomaker.Logo(ic_df,
                          ax=ax,
                          color_scheme=dna_colors,
                          vpad=0.0,
                          width=1.0)
    logo.style_spines(spines=['top', 'right', 'left'], visible=False)
    ax.spines['left'].set_visible(True)

    ax.set_ylabel("Reconstruction\n(scaled by IC)", fontsize=9)

    seq_len = len(sequence)
    ax.set_xlim(0, seq_len)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=True, length=3)

    start_a, start_b = parse_header(header)
    y_line = -0.05

    ax.plot([0, seq_len], [y_line, y_line], color='black', lw=1, clip_on=False)

    def draw_motif_box(start_idx, label):
        if start_idx is None: return
        rect = patches.Rectangle((start_idx, y_line - 0.5), motif_length, 0.4,
                                 linewidth=1, edgecolor='black', facecolor='white', clip_on=False)
        ax.add_patch(rect)
        ax.text(start_idx + motif_length / 2, y_line - 0.3, label,
                ha='center', va='center', fontsize=8, fontweight='bold', clip_on=False)

    draw_motif_box(start_a, "Motif A")
    draw_motif_box(start_b, "Motif B")

    clean_title = header.split('|')[0].replace(">", "")
    plt.title(f"Sequence: {clean_title}", fontsize=10)

    plt.tight_layout()
    plt.show()


full_header = ">seq0019|label=B_only|posAmotif=None|posBmotif=106|gaplength=None"
full_seq = "T-TAA-------A--GG-CG-A-ACGG-GG--CCCTCA-AT-ATTTGC-TATACCAGAA-TA-GTTT-AGAT-GTGCATT--------GACTTCGGCA-A-GTCCAGT-C-GCAC-TGTG"

plot_compressed_logo(full_header, full_seq, motif_length=7)
