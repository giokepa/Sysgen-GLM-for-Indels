import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

from fundemental_classes.dna_dataset import DNADataset


def plot(header, sequence, prob_matrix, motif_length=10, small_ic_threshold=0.05, with_deletions=True):
    df = pd.DataFrame(prob_matrix, columns=['A', 'C', 'G', 'T', '-']) if (
        with_deletions) else pd.DataFrame(prob_matrix, columns=['A', 'C', 'G', 'T'])
    ic_df = logomaker.transform_matrix(df, from_type='probability', to_type='information')

    fig, ax = plt.subplots(figsize=(15, 2.5))

    dna_colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', '-': 'black'}

    logo = logomaker.Logo(ic_df, ax=ax, color_scheme=dna_colors, vpad=0.0, width=1.0)
    logo.style_spines(spines=['top', 'right', 'left'], visible=False)
    ax.spines['left'].set_visible(True)
    ax.set_ylabel("Reconstruction\n(scaled by IC)", fontsize=9)

    y_max = ax.get_ylim()[1]
    for i, nucleotide in enumerate(sequence):
        color = dna_colors.get(nucleotide, 'black')
        ax.text(i, y_max * 1.05, nucleotide,
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color=color, family='monospace')

    seq_len = len(sequence)
    ax.set_xlim(0, seq_len)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=True, length=3)

    motif_positions = DNADataset.parse_motif_positions(header)

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

    def inv_h(v, thr, min_len, max_len, gamma=2.0):
        s = (thr - v) / max(thr, 1e-12)
        s = float(np.clip(s, 0.0, 1.0))
        s = s ** gamma
        return min_len + s * (max_len - min_len)

    letters = ['A', 'C', 'G', 'T', '-'] if with_deletions else ['A', 'C', 'G', 'T']
    L = min(len(ic_df), seq_len)

    for pos in range(L):
        small = [(letter, float(ic_df.loc[pos, letter]))
                 for letter in letters
                 if 0 < float(ic_df.loc[pos, letter]) <= small_ic_threshold]

        if not small:
            continue

        small.sort(key=lambda x: x[1])

        y_cursor = y_line
        for letter, v in small:
            h = inv_h(
                v,
                small_ic_threshold,
                min_len=0.01,
                max_len=0.8,
                gamma=4.0
            )

            rect = patches.Rectangle(
                (pos - 0.5, y_cursor - h),
                0.7,
                h,
                linewidth=0,
                facecolor=dna_colors[letter],
                clip_on=False
            )
            ax.add_patch(rect)
            y_cursor -= h

    clean_title = header.split('|')[0].replace(">", "")
    #plt.title(f"Sequence: {clean_title}", fontsize=10)
    plt.tight_layout()
    plt.show()
