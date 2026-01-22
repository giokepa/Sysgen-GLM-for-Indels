import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

from fundemental_classes.dna_dataset import DNADataset


def plot(header, sequence, prob_matrix, motif_length=10,
         small_ic_threshold=0.05,     # letters with IC <= this are moved "below"
         min_below_total=0.04,        # minimum total stack height below baseline
         max_below_total=0.35): 

    df = pd.DataFrame(prob_matrix, columns=['A', 'C', 'G', 'T', '-'])
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

    start_a, start_b = DNADataset.parse_header(header)
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

    def inv_total_height(values, threshold):
        """
        values: array of IC values that are <= threshold
        smaller values => larger total height
        """
        if len(values) == 0:
            return 0.0
        # Use mean as a summary of "how small" they are
        m = float(np.mean(values))
        s = (threshold - m) / max(threshold, 1e-12)  # m->0 => s->1
        s = float(np.clip(s, 0.0, 1.0))
        return min_below_total + s * (max_below_total - min_below_total)

    letters = ['A', 'C', 'G', 'T', '-']
    L = min(len(ic_df), seq_len)

    for pos in range(L):
        # Pick "small" letters at this position
        small = [(letter, float(ic_df.loc[pos, letter]))
                 for letter in letters
                 if 0 < float(ic_df.loc[pos, letter]) <= small_ic_threshold]

        if not small:
            continue

        vals = np.array([v for _, v in small], dtype=float)
        total_h = inv_total_height(vals, small_ic_threshold)

        # Split total height among letters proportional to their IC
        denom = float(vals.sum()) if vals.sum() > 0 else 1.0

        y_cursor = y_line  # start at baseline and stack downward
        for letter, v in sorted(small, key=lambda x: x[1]):  # small-to-larger (order is aesthetic)
            h = total_h * (v / denom)

            # Draw a bar segment of width 1 stacked downward at x=pos
            rect = patches.Rectangle(
                (pos - 0.5, y_cursor - h),
                1.0,
                h,
                linewidth=0,
                facecolor=dna_colors[letter],
                clip_on=False
            )
            ax.add_patch(rect)
            y_cursor -= h


    clean_title = header.split('|')[0].replace(">", "")
    plt.title(f"Sequence: {clean_title}", fontsize=10)
    plt.tight_layout()
    plt.show()
