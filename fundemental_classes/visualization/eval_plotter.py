import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import linregress


class EvaluationVisualizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        sns.set_style("whitegrid")

    def _clean_data(self, data):
        data = np.array(data)
        return data[np.isfinite(data)]

    def _format_pvalue(self, p):
        if pd.isna(p):
            return "nan"
        if p == 0 or p < 1e-16:
            return "< 1e-16"
        return f"{p:.4e}"

    def plot_boxplots(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        base_4nt_ce = self._clean_data(self.evaluator.results['baseline_4nt']['cross_entropies'])
        del_4nt_ce = self._clean_data(self.evaluator.results['deletion_4nt']['cross_entropies'])

        base_4nt_perp = self._clean_data(self.evaluator.results['baseline_4nt']['perplexities'])
        del_4nt_perp = self._clean_data(self.evaluator.results['deletion_4nt']['perplexities'])

        def plot_single_boxplot(ax, data1, data2, title, ylabel, color1='lightgreen', color2='lightsalmon'):
            bp = ax.boxplot([data1, data2], labels=['Baseline', 'With Deletions'],
                            patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor(color1)
            bp['boxes'][1].set_facecolor(color2)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            means = [np.mean(data1), np.mean(data2)]
            ax.plot([1, 2], means, 'D', color='red', markersize=8, label='Mean', zorder=3)
            ax.legend()

        plot_single_boxplot(axes[0], base_4nt_ce, del_4nt_ce,
                            'Cross-Entropy (4-Nucleotide Motif)', 'Cross-Entropy')

        plot_single_boxplot(axes[1], base_4nt_perp, del_4nt_perp,
                            'Perplexity (4-Nucleotide Motif)', 'Perplexity')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boxplots saved to: {save_path}")
        plt.show()

    def plot_distributions(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        baseline_ce = self._clean_data(self.evaluator.results['baseline_4nt']['cross_entropies'])
        deletion_ce = self._clean_data(self.evaluator.results['deletion_4nt']['cross_entropies'])

        if len(baseline_ce) == 0 or len(deletion_ce) == 0:
            print("Warning: No finite cross-entropy values found!")
            return

        ax1 = axes[0]
        ax1.hist(baseline_ce, bins=30, alpha=0.6, label='Baseline', color='green', edgecolor='black')
        ax1.hist(deletion_ce, bins=30, alpha=0.6, label='With Deletions', color='orange', edgecolor='black')
        ax1.set_xlabel('Cross-Entropy (4-Nucleotide Motif)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Cross-Entropy Distribution (Histogram)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        df = self.evaluator.get_results_dataframe()
        df_4nt = df[df['comparison_type'] == '4nt']
        df_4nt = df_4nt[np.isfinite(df_4nt['cross_entropy'])]

        if len(df_4nt) > 0:
            sns.violinplot(data=df_4nt, x='model', y='cross_entropy', ax=ax2,
                           palette=['lightgreen', 'lightsalmon'])
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Cross-Entropy (4-Nucleotide Motif)', fontsize=12)
            ax2.set_title('Cross-Entropy Distribution (Violin)', fontsize=14, fontweight='bold')
            ax2.set_xticklabels(['Baseline', 'With Deletions'])
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plots saved to: {save_path}")
        plt.show()

    def plot_scatter_comparison(self, save_path: str = None):
        fig, ax = plt.subplots(figsize=(8, 7))

        base_4nt = np.array(self.evaluator.results['baseline_4nt']['cross_entropies'])
        del_4nt = np.array(self.evaluator.results['deletion_4nt']['cross_entropies'])

        mask = np.isfinite(base_4nt) & np.isfinite(del_4nt)
        x_clean, y_clean = base_4nt[mask], del_4nt[mask]

        if len(x_clean) > 0:
            ax.scatter(x_clean, y_clean, alpha=0.5, s=50,
                       edgecolors='black', linewidth=0.5, color='green')

            min_val = min(x_clean.min(), y_clean.min())
            max_val = max(x_clean.max(), y_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                    label='y=x (Perfect Agreement)')

            slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
            x_line = np.array([min_val, max_val])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'b-', linewidth=2,
                    label=f'Regression (R²={r_value ** 2:.3f})')

            ax.legend(fontsize=10)

        ax.set_xlabel('Baseline CE', fontsize=12)
        ax.set_ylabel('Deletion CE', fontsize=12)
        ax.set_title('Comparison (4-Nucleotide Motif)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")
        plt.show()

    def plot_summary_figure(self, comparison_results: dict, save_path: str = None):
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        base_4nt = self._clean_data(self.evaluator.results['baseline_4nt']['cross_entropies'])
        del_4nt = self._clean_data(self.evaluator.results['deletion_4nt']['cross_entropies'])

        ax1 = fig.add_subplot(gs[0, :2])
        bp2 = ax1.boxplot([base_4nt, del_4nt], labels=['Baseline', 'With Deletions'],
                          patch_artist=True, widths=0.5)
        bp2['boxes'][0].set_facecolor('lightgreen')
        bp2['boxes'][1].set_facecolor('lightsalmon')
        ax1.set_ylabel('Cross-Entropy', fontsize=11)
        ax1.set_title('Cross-Entropy Comparison (4-Nucleotide Motif)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        min_len_4nt = min(len(base_4nt), len(del_4nt))
        if min_len_4nt > 0:
            b_trunc_4nt = base_4nt[:min_len_4nt]
            d_trunc_4nt = del_4nt[:min_len_4nt]
            t_stat_4nt, t_p_4nt = sp_stats.ttest_rel(b_trunc_4nt, d_trunc_4nt)
            w_stat_4nt, w_p_4nt = sp_stats.wilcoxon(b_trunc_4nt, d_trunc_4nt)
        else:
            t_p_4nt, w_p_4nt = np.nan, np.nan

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')

        table_data_4nt = [
            ['Statistical Test', 'p-value', 'Sig.'],
            ['Paired t-test',
             self._format_pvalue(t_p_4nt),
             '✓' if (not pd.isna(t_p_4nt) and t_p_4nt < 0.05) else '✗'],
            ['Wilcoxon',
             self._format_pvalue(w_p_4nt),
             '✓' if (not pd.isna(w_p_4nt) and w_p_4nt < 0.05) else '✗']
        ]

        table2 = ax2.table(cellText=table_data_4nt, cellLoc='center', loc='center',
                           colWidths=[0.5, 0.3, 0.15])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 2)
        for i in range(3):
            table2[(0, i)].set_facecolor('#FF9800')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        ax2.set_title('Statistical Tests (α=0.05)', fontsize=10, fontweight='bold', pad=20)

        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(base_4nt, bins=30, alpha=0.6, label='Baseline', color='green', edgecolor='black')
        ax3.hist(del_4nt, bins=30, alpha=0.6, label='With Deletions', color='orange', edgecolor='black')
        ax3.set_xlabel('Cross-Entropy (4-Nucleotide Motif)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[2, :2])
        df = self.evaluator.get_results_dataframe()
        df_4nt = df[df['comparison_type'] == '4nt']
        df_4nt = df_4nt[np.isfinite(df_4nt['cross_entropy'])]

        if len(df_4nt) > 0:
            sns.violinplot(data=df_4nt, x='model', y='cross_entropy', hue='model', ax=ax4,
                           palette=['lightgreen', 'lightsalmon'], legend=False)

            ax4.set_xlabel('')
            ax4.set_ylabel('CE (4-nt)', fontsize=10)
            ax4.set_title('Distribution Shape', fontsize=11, fontweight='bold')
            ax4.set_xticks(range(2))
            ax4.set_xticklabels(['Baseline', 'Deletion'])
        ax5 = fig.add_subplot(gs[2, 2])

        def calc_d(x, y):
            if len(x) == 0 or len(y) == 0: return 0
            diff = x.mean() - y.mean()
            pooled_std = np.sqrt((x.std() ** 2 + y.std() ** 2) / 2)
            return diff / pooled_std

        d_4nt = calc_d(base_4nt, del_4nt)

        bars = ax5.barh(['4-Nucleotide'],
                        [abs(d_4nt)],
                        color=['green'])

        limit = max(1.5, abs(d_4nt) * 1.2)
        ax5.set_xlim([0, limit])
        ax5.set_xlabel("Cohen's d (absolute)", fontsize=10)
        ax5.set_title("Effect Sizes", fontsize=11, fontweight='bold')
        ax5.axvline(0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax5.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax5.axvline(0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax5.grid(True, alpha=0.3, axis='x')

        fig.suptitle('Model Evaluation Summary (Motif Regions Only)', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary figure saved to: {save_path}")
        plt.show()
