import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy import stats as sp_stats
from scipy.stats import linregress


class EvaluationVisualizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        sns.set_style("whitegrid")

    def plot_boxplots(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        baseline_motif_ce = np.array(self.evaluator.results['baseline']['motif_cross_entropies'])
        deletion_motif_ce = np.array(self.evaluator.results['deletion']['motif_cross_entropies'])

        baseline_motif_ce = baseline_motif_ce[np.isfinite(baseline_motif_ce)]
        deletion_motif_ce = deletion_motif_ce[np.isfinite(deletion_motif_ce)]

        ax1 = axes[0, 0]
        data_motif_ce = [baseline_motif_ce, deletion_motif_ce]
        bp1 = ax1.boxplot(data_motif_ce, labels=['Baseline', 'With Deletions'],
                          patch_artist=True, widths=0.6)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp1['boxes'][1].set_facecolor('lightcoral')
        ax1.set_ylabel('Cross-Entropy', fontsize=12)
        ax1.set_title('Cross-Entropy Comparison (Motif-Only)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        means_motif_ce = [np.mean(baseline_motif_ce), np.mean(deletion_motif_ce)]
        ax1.plot([1, 2], means_motif_ce, 'D', color='red', markersize=8, label='Mean', zorder=3)
        ax1.legend()

        baseline_perp = np.array(self.evaluator.results['baseline']['perplexities'])
        deletion_perp = np.array(self.evaluator.results['deletion']['perplexities'])

        baseline_perp = baseline_perp[np.isfinite(baseline_perp)]
        deletion_perp = deletion_perp[np.isfinite(deletion_perp)]

        ax2 = axes[0, 1]
        data_perp = [baseline_perp, deletion_perp]
        bp2 = ax2.boxplot(data_perp, labels=['Baseline', 'With Deletions'],
                          patch_artist=True, widths=0.6)
        bp2['boxes'][0].set_facecolor('lightblue')
        bp2['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Perplexity Comparison (Motif-Only)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        means_perp = [np.mean(baseline_perp), np.mean(deletion_perp)]
        ax2.plot([1, 2], means_perp, 'D', color='red', markersize=8, label='Mean', zorder=3)
        ax2.legend()

        baseline_4nt_ce = np.array(self.evaluator.results['baseline_4nt']['cross_entropies'])
        deletion_4nt_ce = np.array(self.evaluator.results['deletion_4nt']['cross_entropies'])

        baseline_4nt_ce = baseline_4nt_ce[np.isfinite(baseline_4nt_ce)]
        deletion_4nt_ce = deletion_4nt_ce[np.isfinite(deletion_4nt_ce)]

        ax3 = axes[1, 0]
        data_4nt_ce = [baseline_4nt_ce, deletion_4nt_ce]
        bp3 = ax3.boxplot(data_4nt_ce, labels=['Baseline', 'With Deletions'],
                          patch_artist=True, widths=0.6)
        bp3['boxes'][0].set_facecolor('lightgreen')
        bp3['boxes'][1].set_facecolor('lightsalmon')
        ax3.set_ylabel('Cross-Entropy', fontsize=12)
        ax3.set_title('Cross-Entropy Comparison (4-Nucleotide)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        means_4nt_ce = [np.mean(baseline_4nt_ce), np.mean(deletion_4nt_ce)]
        ax3.plot([1, 2], means_4nt_ce, 'D', color='red', markersize=8, label='Mean', zorder=3)
        ax3.legend()

        baseline_4nt_perp = np.array(self.evaluator.results['baseline_4nt']['perplexities'])
        deletion_4nt_perp = np.array(self.evaluator.results['deletion_4nt']['perplexities'])

        baseline_4nt_perp = baseline_4nt_perp[np.isfinite(baseline_4nt_perp)]
        deletion_4nt_perp = deletion_4nt_perp[np.isfinite(deletion_4nt_perp)]

        ax4 = axes[1, 1]
        data_4nt_perp = [baseline_4nt_perp, deletion_4nt_perp]
        bp4 = ax4.boxplot(data_4nt_perp, labels=['Baseline', 'With Deletions'],
                          patch_artist=True, widths=0.6)
        bp4['boxes'][0].set_facecolor('lightgreen')
        bp4['boxes'][1].set_facecolor('lightsalmon')
        ax4.set_ylabel('Perplexity', fontsize=12)
        ax4.set_title('Perplexity Comparison (4-Nucleotide)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        means_4nt_perp = [np.mean(baseline_4nt_perp), np.mean(deletion_4nt_perp)]
        ax4.plot([1, 2], means_4nt_perp, 'D', color='red', markersize=8, label='Mean', zorder=3)
        ax4.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boxplots saved to: {save_path}")
        plt.show()

    def plot_distributions(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        baseline_ce = np.array(self.evaluator.results['baseline']['motif_cross_entropies'])
        deletion_ce = np.array(self.evaluator.results['deletion']['motif_cross_entropies'])

        baseline_ce = baseline_ce[np.isfinite(baseline_ce)]
        deletion_ce = deletion_ce[np.isfinite(deletion_ce)]

        if len(baseline_ce) == 0 or len(deletion_ce) == 0:
            print("Warning: No finite cross-entropy values found!")
            return

        ax1 = axes[0, 0]
        ax1.hist(baseline_ce, bins=30, alpha=0.6, label='Baseline', color='blue', edgecolor='black')
        ax1.hist(deletion_ce, bins=30, alpha=0.6, label='With Deletions', color='red', edgecolor='black')
        ax1.set_xlabel('Cross-Entropy (Motif-Only)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Cross-Entropy Distribution (Histogram)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        kde_baseline = gaussian_kde(baseline_ce)
        kde_deletion = gaussian_kde(deletion_ce)
        x_range = np.linspace(min(baseline_ce.min(), deletion_ce.min()),
                              max(baseline_ce.max(), deletion_ce.max()), 200)
        ax2.plot(x_range, kde_baseline(x_range), label='Baseline', linewidth=2, color='blue')
        ax2.plot(x_range, kde_deletion(x_range), label='With Deletions', linewidth=2, color='red')
        ax2.fill_between(x_range, kde_baseline(x_range), alpha=0.3, color='blue')
        ax2.fill_between(x_range, kde_deletion(x_range), alpha=0.3, color='red')
        ax2.set_xlabel('Cross-Entropy (Motif-Only)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Cross-Entropy Distribution (KDE)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        df = self.evaluator.get_results_dataframe()
        df_motif = df[df['comparison_type'] == 'motif_only']
        df_motif = df_motif[np.isfinite(df_motif['cross_entropy'])]

        if len(df_motif) > 0:
            sns.violinplot(data=df_motif, x='model', y='cross_entropy', ax=ax3,
                           palette=['lightblue', 'lightcoral'])
            ax3.set_xlabel('Model', fontsize=11)
            ax3.set_ylabel('Cross-Entropy (Motif-Only)', fontsize=11)
            ax3.set_title('Cross-Entropy Distribution (Violin)', fontsize=12, fontweight='bold')
            ax3.set_xticklabels(['Baseline', 'With Deletions'])

        ax4 = axes[1, 1]
        min_len = min(len(baseline_ce), len(deletion_ce))
        if min_len > 0:
            baseline_ce_sorted = np.sort(baseline_ce)[:min_len]
            deletion_ce_sorted = np.sort(deletion_ce)[:min_len]
            sp_stats.probplot(baseline_ce_sorted - deletion_ce_sorted, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Baseline - Deletion Differences)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plots saved to: {save_path}")
        plt.show()

    def plot_scatter_comparison(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        baseline_motif_ce = np.array(self.evaluator.results['baseline']['motif_cross_entropies'])
        deletion_motif_ce = np.array(self.evaluator.results['deletion']['motif_cross_entropies'])

        finite_mask = np.isfinite(baseline_motif_ce) & np.isfinite(deletion_motif_ce)
        baseline_motif_ce = baseline_motif_ce[finite_mask]
        deletion_motif_ce = deletion_motif_ce[finite_mask]

        if len(baseline_motif_ce) > 0 and len(deletion_motif_ce) > 0:
            ax1 = axes[0]
            ax1.scatter(baseline_motif_ce, deletion_motif_ce, alpha=0.5, s=50,
                        edgecolors='black', linewidth=0.5)

            min_val = min(baseline_motif_ce.min(), deletion_motif_ce.min())
            max_val = max(baseline_motif_ce.max(), deletion_motif_ce.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                     label='y=x (Perfect Agreement)')

            slope, intercept, r_value, p_value, std_err = linregress(baseline_motif_ce,
                                                                     deletion_motif_ce)
            x_line = np.array([min_val, max_val])
            y_line = slope * x_line + intercept
            ax1.plot(x_line, y_line, 'b-', linewidth=2,
                     label=f'Regression (R²={r_value ** 2:.3f})')

            ax1.set_xlabel('Baseline Model CE (Motif-Only)', fontsize=12)
            ax1.set_ylabel('Deletion Model CE (Motif-Only)', fontsize=12)
            ax1.set_title('Per-Sequence Comparison (Motif-Only)', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')

        baseline_4nt_ce = np.array(self.evaluator.results['baseline_4nt']['cross_entropies'])
        deletion_4nt_ce = np.array(self.evaluator.results['deletion_4nt']['cross_entropies'])

        finite_mask_4nt = np.isfinite(baseline_4nt_ce) & np.isfinite(deletion_4nt_ce)
        baseline_4nt_ce = baseline_4nt_ce[finite_mask_4nt]
        deletion_4nt_ce = deletion_4nt_ce[finite_mask_4nt]

        if len(baseline_4nt_ce) > 0 and len(deletion_4nt_ce) > 0:
            ax2 = axes[1]
            ax2.scatter(baseline_4nt_ce, deletion_4nt_ce, alpha=0.5, s=50,
                        edgecolors='black', linewidth=0.5, color='green')

            min_val = min(baseline_4nt_ce.min(), deletion_4nt_ce.min())
            max_val = max(baseline_4nt_ce.max(), deletion_4nt_ce.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                     label='y=x (Perfect Agreement)')

            slope, intercept, r_value, p_value, std_err = linregress(baseline_4nt_ce,
                                                                     deletion_4nt_ce)
            x_line = np.array([min_val, max_val])
            y_line = slope * x_line + intercept
            ax2.plot(x_line, y_line, 'b-', linewidth=2,
                     label=f'Regression (R²={r_value ** 2:.3f})')

            ax2.set_xlabel('Baseline Model CE (4-Nucleotide)', fontsize=12)
            ax2.set_ylabel('Deletion Model CE (4-Nucleotide)', fontsize=12)
            ax2.set_title('Per-Sequence Comparison (4-Nucleotide)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")
        plt.show()

    def plot_summary_figure(self, comparison_results: dict, save_path: str = None):
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        baseline_motif_ce = np.array(self.evaluator.results['baseline']['motif_cross_entropies'])
        deletion_motif_ce = np.array(self.evaluator.results['deletion']['motif_cross_entropies'])

        baseline_motif_ce = baseline_motif_ce[np.isfinite(baseline_motif_ce)]
        deletion_motif_ce = deletion_motif_ce[np.isfinite(deletion_motif_ce)]

        ax1 = fig.add_subplot(gs[0, :2])
        data_motif = [baseline_motif_ce, deletion_motif_ce]
        bp1 = ax1.boxplot(data_motif, labels=['Baseline', 'With Deletions'],
                          patch_artist=True, widths=0.5)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp1['boxes'][1].set_facecolor('lightcoral')
        ax1.set_ylabel('Cross-Entropy', fontsize=11)
        ax1.set_title('Cross-Entropy Comparison (Motif-Only)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        motif_stats = comparison_results.get('motif_only', {})
        table_data_motif = [
            ['Statistical Test', 'p-value', 'Sig.'],
            ['Paired t-test',
             f"{motif_stats.get('paired_ttest', {}).get('pvalue', 0):.3e}",
             '✓' if motif_stats.get('paired_ttest', {}).get('significant', False) else '✗'],
            ['Wilcoxon',
             f"{motif_stats.get('wilcoxon', {}).get('pvalue', 0):.3e}",
             '✓' if motif_stats.get('wilcoxon', {}).get('significant', False) else '✗']
        ]
        table1 = ax2.table(cellText=table_data_motif, cellLoc='center', loc='center',
                           colWidths=[0.5, 0.3, 0.15])
        table1.auto_set_font_size(False)
        table1.set_fontsize(8)
        table1.scale(1, 2)
        for i in range(3):
            table1[(0, i)].set_facecolor('#4CAF50')
            table1[(0, i)].set_text_props(weight='bold', color='white')
        ax2.set_title('Motif-Only Tests (α=0.05)', fontsize=10, fontweight='bold', pad=20)

        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(baseline_motif_ce, bins=30, alpha=0.6, label='Baseline', color='blue',
                 edgecolor='black')
        ax3.hist(deletion_motif_ce, bins=30, alpha=0.6, label='With Deletions', color='red',
                 edgecolor='black')
        ax3.axvline(baseline_motif_ce.mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Baseline Mean: {baseline_motif_ce.mean():.3f}')
        ax3.axvline(deletion_motif_ce.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Deletion Mean: {deletion_motif_ce.mean():.3f}')
        ax3.set_xlabel('Cross-Entropy (Motif-Only)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distribution Comparison (Motif-Only)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        baseline_4nt_ce = np.array(self.evaluator.results['baseline_4nt']['cross_entropies'])
        deletion_4nt_ce = np.array(self.evaluator.results['deletion_4nt']['cross_entropies'])

        baseline_4nt_ce = baseline_4nt_ce[np.isfinite(baseline_4nt_ce)]
        deletion_4nt_ce = deletion_4nt_ce[np.isfinite(deletion_4nt_ce)]

        ax4 = fig.add_subplot(gs[2, :2])
        data_4nt = [baseline_4nt_ce, deletion_4nt_ce]
        bp2 = ax4.boxplot(data_4nt, labels=['Baseline', 'With Deletions'],
                          patch_artist=True, widths=0.5)
        bp2['boxes'][0].set_facecolor('lightgreen')
        bp2['boxes'][1].set_facecolor('lightsalmon')
        ax4.set_ylabel('Cross-Entropy', fontsize=11)
        ax4.set_title('Cross-Entropy Comparison (4-Nucleotide)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        nt4_stats = comparison_results.get('4nt_comparison', {})
        table_data_4nt = [
            ['Statistical Test', 'p-value', 'Sig.'],
            ['Paired t-test',
             f"{nt4_stats.get('paired_ttest', {}).get('pvalue', 0):.3e}",
             '✓' if nt4_stats.get('paired_ttest', {}).get('significant', False) else '✗'],
            ['Wilcoxon',
             f"{nt4_stats.get('wilcoxon', {}).get('pvalue', 0):.3e}",
             '✓' if nt4_stats.get('wilcoxon', {}).get('significant', False) else '✗']
        ]
        table2 = ax5.table(cellText=table_data_4nt, cellLoc='center', loc='center',
                           colWidths=[0.5, 0.3, 0.15])
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(1, 2)
        for i in range(3):
            table2[(0, i)].set_facecolor('#FF9800')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        ax5.set_title('4-Nucleotide Tests (α=0.05)', fontsize=10, fontweight='bold', pad=20)

        ax6 = fig.add_subplot(gs[3, 0])
        df = self.evaluator.get_results_dataframe()
        df_motif = df[df['comparison_type'] == 'motif_only']
        df_motif = df_motif[np.isfinite(df_motif['cross_entropy'])]

        if len(df_motif) > 0:
            sns.violinplot(data=df_motif, x='model', y='cross_entropy', ax=ax6,
                           palette=['lightblue', 'lightcoral'])
            ax6.set_xlabel('')
            ax6.set_ylabel('CE (Motif)', fontsize=10)
            ax6.set_title('Distribution Shape', fontsize=11, fontweight='bold')
            ax6.set_xticklabels(['Baseline', 'Deletion'])

        ax7 = fig.add_subplot(gs[3, 1])
        sorted_baseline_motif = np.sort(baseline_motif_ce)
        sorted_deletion_motif = np.sort(deletion_motif_ce)
        y_baseline = np.arange(1, len(sorted_baseline_motif) + 1) / len(sorted_baseline_motif)
        y_deletion = np.arange(1, len(sorted_deletion_motif) + 1) / len(sorted_deletion_motif)
        ax7.plot(sorted_baseline_motif, y_baseline, label='Baseline', linewidth=2, color='blue')
        ax7.plot(sorted_deletion_motif, y_deletion, label='Deletion', linewidth=2, color='red')
        ax7.set_xlabel('CE (Motif)', fontsize=10)
        ax7.set_ylabel('Cumulative Probability', fontsize=10)
        ax7.set_title('CDF Comparison', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[3, 2])
        cohens_d_motif = motif_stats.get('effect_size', {}).get('cohens_d', 0)
        cohens_d_4nt = nt4_stats.get('effect_size', {}).get('cohens_d', 0)

        bars = ax8.barh(['Motif-Only', '4-Nucleotide'],
                        [abs(cohens_d_motif), abs(cohens_d_4nt)],
                        color=['blue', 'green'])
        ax8.set_xlim([0, max(1.5, abs(cohens_d_motif) * 1.2, abs(cohens_d_4nt) * 1.2)])
        ax8.set_xlabel("Cohen's d", fontsize=10)
        ax8.set_title("Effect Sizes", fontsize=11, fontweight='bold')
        ax8.axvline(0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax8.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax8.axvline(0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax8.text(0.2, -0.5, 'Small', fontsize=7, ha='center')
        ax8.text(0.5, -0.5, 'Medium', fontsize=7, ha='center')
        ax8.text(0.8, -0.5, 'Large', fontsize=7, ha='center')
        ax8.grid(True, alpha=0.3, axis='x')

        fig.suptitle('Model Evaluation Summary', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary figure saved to: {save_path}")
        plt.show()
