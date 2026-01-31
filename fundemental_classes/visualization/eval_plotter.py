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
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        baseline_ce = self.evaluator.results['baseline']['cross_entropies']
        deletion_ce = self.evaluator.results['deletion']['cross_entropies']
        baseline_perp = self.evaluator.results['baseline']['perplexities']
        deletion_perp = self.evaluator.results['deletion']['perplexities']

        ax1 = axes[0]
        data_ce = [baseline_ce, deletion_ce]
        bp1 = ax1.boxplot(data_ce, labels=['Baseline', 'With Deletions'],patch_artist=True, widths=0.6)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp1['boxes'][1].set_facecolor('lightcoral')
        ax1.set_ylabel('Cross-Entropy', fontsize=12)
        ax1.set_title('Cross-Entropy Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        means_ce = [np.mean(baseline_ce), np.mean(deletion_ce)]
        ax1.plot([1, 2], means_ce, 'D', color='red', markersize=8, label='Mean', zorder=3)
        ax1.legend()

        ax2 = axes[1]
        data_perp = [baseline_perp, deletion_perp]
        bp2 = ax2.boxplot(data_perp, labels=['Baseline', 'With Deletions'],
        patch_artist = True, widths = 0.6)
        bp2['boxes'][0].set_facecolor('lightblue')
        bp2['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Perplexity Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        means_perp = [np.mean(baseline_perp), np.mean(deletion_perp)]
        ax2.plot([1, 2], means_perp, 'D', color='red', markersize=8, label='Mean', zorder=3)
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Boxplots saved to: {save_path}")
        plt.show()


    def plot_distributions(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        baseline_ce = np.array(self.evaluator.results['baseline']['cross_entropies'])
        deletion_ce = np.array(self.evaluator.results['deletion']['cross_entropies'])

        ax1 = axes[0, 0]
        ax1.hist(baseline_ce, bins=30, alpha=0.6, label='Baseline', color='blue', edgecolor='black')
        ax1.hist(deletion_ce, bins=30, alpha=0.6, label='With Deletions', color='red', edgecolor='black')
        ax1.set_xlabel('Cross-Entropy', fontsize=11)
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
        ax2.set_xlabel('Cross-Entropy', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Cross-Entropy Distribution (KDE)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        df = self.evaluator.get_results_dataframe()
        sns.violinplot(data=df, x='model', y='cross_entropy', ax=ax3, palette=['lightblue', 'lightcoral'])
        ax3.set_xlabel('Model', fontsize=11)
        ax3.set_ylabel('Cross-Entropy', fontsize=11)
        ax3.set_title('Cross-Entropy Distribution (Violin)', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(['Baseline', 'With Deletions'])

        ax4 = axes[1, 1]
        sp_stats.probplot(baseline_ce - deletion_ce, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Baseline - Deletion Differences)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plots saved to: {save_path}")
        plt.show()


    def plot_scatter_comparison(self, save_path: str = None):
        baseline_ce = np.array(self.evaluator.results['baseline']['cross_entropies'])
        deletion_ce = np.array(self.evaluator.results['deletion']['cross_entropies'])

        if len(baseline_ce) != len(deletion_ce):
            print("Cannot create scatter plot: different number of sequences evaluated")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(baseline_ce, deletion_ce, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)

        min_val = min(baseline_ce.min(), deletion_ce.min())
        max_val = max(baseline_ce.max(), deletion_ce.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (Perfect Agreement)')

        slope, intercept, r_value, p_value, std_err = linregress(baseline_ce, deletion_ce)
        x_line = np.array([min_val, max_val])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'b-', linewidth=2, label=f'Regression (R²={r_value ** 2:.3f})')

        ax.set_xlabel('Baseline Model Cross-Entropy', fontsize=12)
        ax.set_ylabel('Deletion Model Cross-Entropy', fontsize=12)
        ax.set_title('Per-Sequence Cross-Entropy Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")
        plt.show()


    def plot_summary_figure(self, comparison_results: dict, save_path: str = None):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        baseline_ce = np.array(self.evaluator.results['baseline']['cross_entropies'])
        deletion_ce = np.array(self.evaluator.results['deletion']['cross_entropies'])

        ax1 = fig.add_subplot(gs[0, :2])
        data = [baseline_ce, deletion_ce]
        bp = ax1.boxplot(data, labels=['Baseline', 'With Deletions'], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax1.set_ylabel('Cross-Entropy', fontsize=11)
        ax1.set_title('Cross-Entropy Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        table_data = [
            ['Statistical Test', 'p-value', 'Sig.'],
            ['Paired t-test', f"{comparison_results.get('paired_ttest', {}).get('pvalue', 0):.4f}",
             '✓' if comparison_results.get('paired_ttest', {}).get('significant', False) else '✗'],
            ['Wilcoxon', f"{comparison_results.get('wilcoxon', {}).get('pvalue', 0):.4f}",
             '✓' if comparison_results.get('wilcoxon', {}).get('significant', False) else '✗'],
            ['Mann-Whitney', f"{comparison_results.get('mann_whitney', {}).get('pvalue', 0):.4f}",
             '✓' if comparison_results.get('mann_whitney', {}).get('significant', False) else '✗']
        ]
        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.5, 0.25, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        ax2.set_title('Statistical Tests (α=0.05)', fontsize=10, fontweight='bold', pad=20)

        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(baseline_ce, bins=30, alpha=0.6, label='Baseline', color='blue', edgecolor='black')
        ax3.hist(deletion_ce, bins=30, alpha=0.6, label='With Deletions', color='red', edgecolor='black')
        ax3.axvline(baseline_ce.mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Baseline Mean: {baseline_ce.mean():.3f}')
        ax3.axvline(deletion_ce.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Deletion Mean: {deletion_ce.mean():.3f}')
        ax3.set_xlabel('Cross-Entropy', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[2, 0])
        df = self.evaluator.get_results_dataframe()
        sns.violinplot(data=df, x='model', y='cross_entropy', ax=ax4, palette=['lightblue', 'lightcoral'])
        ax4.set_xlabel('')
        ax4.set_ylabel('Cross-Entropy', fontsize=10)
        ax4.set_title('Distribution Shape', fontsize=11, fontweight='bold')
        ax4.set_xticklabels(['Baseline', 'Deletion'])

        ax5 = fig.add_subplot(gs[2, 1])
        sorted_baseline = np.sort(baseline_ce)
        sorted_deletion = np.sort(deletion_ce)
        y_baseline = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline)
        y_deletion = np.arange(1, len(sorted_deletion) + 1) / len(sorted_deletion)
        ax5.plot(sorted_baseline, y_baseline, label='Baseline', linewidth=2, color='blue')
        ax5.plot(sorted_deletion, y_deletion, label='With Deletions', linewidth=2, color='red')
        ax5.set_xlabel('Cross-Entropy', fontsize=10)
        ax5.set_ylabel('Cumulative Probability', fontsize=10)
        ax5.set_title('CDF Comparison', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 2])
        cohens_d = comparison_results.get('effect_size', {}).get('cohens_d', 0)
        colors = ['green' if abs(cohens_d) < 0.2 else 'yellow' if abs(cohens_d) < 0.5 else 'orange' if abs(
            cohens_d) < 0.8 else 'red']
        ax6.barh(['Effect Size'], [abs(cohens_d)], color=colors)
        ax6.set_xlim([0, 1.5])
        ax6.set_xlabel("Cohen's d", fontsize=10)
        ax6.set_title(
            f"Effect Size: {cohens_d:.3f}\n({comparison_results.get('effect_size', {}).get('interpretation', 'N/A')})",
            fontsize=11, fontweight='bold')
        ax6.axvline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
        ax6.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
        ax6.axvline(0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
        ax6.legend(fontsize=8, loc='upper right')
        ax6.grid(True, alpha=0.3, axis='x')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary figure saved to: {save_path}")
        plt.show()
