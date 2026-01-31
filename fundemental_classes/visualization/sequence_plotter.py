import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from fundemental_classes.dna_dataset import DNADataset


def plot(header, sequence, prob_matrix, motif_length=10, with_deletions=True):
    df = pd.DataFrame(prob_matrix, columns=['A', 'C', 'G', 'T', '-']) if (
        with_deletions) else pd.DataFrame(prob_matrix, columns=['A', 'C', 'G', 'T'])
    ic_df = logomaker.transform_matrix(df, from_type='probability', to_type='information')

    # Plot
    fig, ax = plt.subplots(figsize=(15, 2.5))

    dna_colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', '-': 'black'}

    logo = logomaker.Logo(ic_df, ax=ax, color_scheme=dna_colors, vpad=0.0, width=1.0)
    logo.style_spines(spines=['top', 'right', 'left'], visible=False)
    ax.spines['left'].set_visible(True)
    ax.set_ylabel("Reconstruction\n(scaled by IC)", fontsize=9)

    seq_len = len(sequence)
    ax.set_xlim(0, seq_len)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=True, length=3)

    motif_positions = parse_motif_positions(header)
    y_line = -0.05
    ax.plot([0, seq_len], [y_line, y_line], color='black', lw=1, clip_on=False)

    def draw_motif_box(start_idx, label):
        rect = patches.Rectangle((start_idx, y_line - 0.5), motif_length, 0.4,
                                 linewidth=1, edgecolor='black', facecolor='white', clip_on=False)
        ax.add_patch(rect)
        ax.text(start_idx + motif_length / 2, y_line - 0.3, label,
                ha='center', va='center', fontsize=8, fontweight='bold', clip_on=False)

    if 'A' in motif_positions:
        for pos in motif_positions['A']:
            draw_motif_box(pos, "Motif A")

    if 'B' in motif_positions:
        for pos in motif_positions['B']:
            draw_motif_box(pos, "Motif B")

    clean_title = header.split('|')[0].replace(">", "")
    plt.title(f"Sequence: {clean_title}", fontsize=10)
    plt.tight_layout()
    plt.show()


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
