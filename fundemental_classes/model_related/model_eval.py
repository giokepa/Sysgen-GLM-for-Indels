import numpy as np
import pandas as pd
import torch
from scipy import stats
from pathlib import Path
import json
from typing import Dict, List, Tuple
from fundemental_classes.dna_dataset import DNADataset
from fundemental_classes.model_related.glm_model import GLMModel


class ModelEvaluator:
    def __init__(self, baseline_model: GLMModel, deletion_model: GLMModel):
        self.baseline_model = baseline_model
        self.deletion_model = deletion_model
        self.results = {
            'baseline': {'cross_entropies': [], 'perplexities': [], 'sequence_ids': []},
            'deletion': {'cross_entropies': [], 'perplexities': [], 'sequence_ids': []}
        }

    def compute_cross_entropy(self, model: GLMModel, sequence: str) -> float:
        total_ce = 0.0
        valid_positions = 0

        for pos in range(len(sequence)):
            true_char = sequence[pos]

            if true_char not in model.relevant_chars:
                continue

            probs = model.predict_position(sequence, pos, debug=False, dna_only=True)
            char_id = model.tokenizer.vocab[true_char]
            pred_prob = probs[char_id]

            ce = -np.log(pred_prob + 1e-10)
            total_ce += ce
            valid_positions += 1

        return total_ce / valid_positions if valid_positions > 0 else float('inf')

    def compute_perplexity(self, cross_entropy: float) -> float:
        return np.exp(cross_entropy)

    def evaluate_dataset(self, test_fasta_path: str, model_type: str, max_sequences: int = None):
        model = self.baseline_model if model_type == 'baseline' else self.deletion_model

        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_type.upper()} model on: {test_fasta_path}")
        print(f"{'=' * 60}")

        dataset = DNADataset(test_fasta_path, model.tokenizer, model.max_length)

        n_sequences = len(dataset) if max_sequences is None else min(max_sequences, len(dataset))

        for idx in range(n_sequences):
            header, sequence = dataset.get_raw_sequence(idx)

            ce = self.compute_cross_entropy(model, sequence)
            perplexity = self.compute_perplexity(ce)

            self.results[model_type]['cross_entropies'].append(ce)
            self.results[model_type]['perplexities'].append(perplexity)
            self.results[model_type]['sequence_ids'].append(header.split('|')[0].replace('>', ''))

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{n_sequences} sequences...")

        ce_array = np.array(self.results[model_type]['cross_entropies'])
        perp_array = np.array(self.results[model_type]['perplexities'])

        print(f"\n{model_type.upper()} Model Results:")
        print(f"  Cross-Entropy: {ce_array.mean():.4f} ± {ce_array.std():.4f}")
        print(f"  Perplexity: {perp_array.mean():.4f} ± {perp_array.std():.4f}")
        print(f"  Min CE: {ce_array.min():.4f}, Max CE: {ce_array.max():.4f}")

    def compare_models(self) -> Dict:
        baseline_ce = np.array(self.results['baseline']['cross_entropies'])
        deletion_ce = np.array(self.results['deletion']['cross_entropies'])

        if len(baseline_ce) == 0 or len(deletion_ce) == 0:
            raise ValueError("Both models must be evaluated before comparison")

        print(f"\n{'=' * 60}")
        print("STATISTICAL COMPARISON")
        print(f"{'=' * 60}")

        comparison_results = {}

        if len(baseline_ce) == len(deletion_ce):
            t_stat, t_pvalue = stats.ttest_rel(baseline_ce, deletion_ce)
            comparison_results['paired_ttest'] = {
                'statistic': float(t_stat),
                'pvalue': float(t_pvalue),
                'significant': t_pvalue < 0.05
            }
            print(f"\nPaired t-test:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {t_pvalue:.6f}")
            print(f"  Significant (α=0.05): {t_pvalue < 0.05}")

        if len(baseline_ce) == len(deletion_ce):
            w_stat, w_pvalue = stats.wilcoxon(baseline_ce, deletion_ce)
            comparison_results['wilcoxon'] = {
                'statistic': float(w_stat),
                'pvalue': float(w_pvalue),
                'significant': w_pvalue < 0.05
            }
            print(f"\nWilcoxon signed-rank test:")
            print(f"  W-statistic: {w_stat:.4f}")
            print(f"  p-value: {w_pvalue:.6f}")
            print(f"  Significant (α=0.05): {w_pvalue < 0.05}")

        u_stat, u_pvalue = stats.mannwhitneyu(baseline_ce, deletion_ce, alternative='two-sided')
        comparison_results['mann_whitney'] = {
            'statistic': float(u_stat),
            'pvalue': float(u_pvalue),
            'significant': u_pvalue < 0.05
        }
        print(f"\nMann-Whitney U test:")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {u_pvalue:.6f}")
        print(f"  Significant (α=0.05): {u_pvalue < 0.05}")

        mean_diff = baseline_ce.mean() - deletion_ce.mean()
        pooled_std = np.sqrt((baseline_ce.std() ** 2 + deletion_ce.std() ** 2) / 2)
        cohens_d = mean_diff / pooled_std
        comparison_results['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
        print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
        print(f"  Interpretation: {self._interpret_cohens_d(cohens_d)}")

        mean_diff = baseline_ce.mean() - deletion_ce.mean()
        se_diff = np.sqrt(baseline_ce.var() / len(baseline_ce) + deletion_ce.var() / len(deletion_ce))
        ci_95 = 1.96 * se_diff
        comparison_results['mean_difference'] = {
            'difference': float(mean_diff),
            'ci_lower': float(mean_diff - ci_95),
            'ci_upper': float(mean_diff + ci_95)
        }
        print(f"\nMean Difference (Baseline - Deletion): {mean_diff:.4f}")
        print(f"  95% CI: [{mean_diff - ci_95:.4f}, {mean_diff + ci_95:.4f}]")

        return comparison_results

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def export_results(self, output_path: str):
        output_data = {
            'baseline': {
                'cross_entropies': [float(x) for x in self.results['baseline']['cross_entropies']],
                'perplexities': [float(x) for x in self.results['baseline']['perplexities']],
                'sequence_ids': self.results['baseline']['sequence_ids'],
                'summary': {
                    'mean_ce': float(np.mean(self.results['baseline']['cross_entropies'])),
                    'std_ce': float(np.std(self.results['baseline']['cross_entropies'])),
                    'mean_perplexity': float(np.mean(self.results['baseline']['perplexities']))
                }
            },
            'deletion': {
                'cross_entropies': [float(x) for x in self.results['deletion']['cross_entropies']],
                'perplexities': [float(x) for x in self.results['deletion']['perplexities']],
                'sequence_ids': self.results['deletion']['sequence_ids'],
                'summary': {
                    'mean_ce': float(np.mean(self.results['deletion']['cross_entropies'])),
                    'std_ce': float(np.std(self.results['deletion']['cross_entropies'])),
                    'mean_perplexity': float(np.mean(self.results['deletion']['perplexities']))
                }
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults exported to: {output_path}")

    def get_results_dataframe(self) -> pd.DataFrame:
        baseline_df = pd.DataFrame({
            'sequence_id': self.results['baseline']['sequence_ids'],
            'model': 'baseline',
            'cross_entropy': self.results['baseline']['cross_entropies'],
            'perplexity': self.results['baseline']['perplexities']
        })

        deletion_df = pd.DataFrame({
            'sequence_id': self.results['deletion']['sequence_ids'],
            'model': 'deletion',
            'cross_entropy': self.results['deletion']['cross_entropies'],
            'perplexity': self.results['deletion']['perplexities']
        })

        return pd.concat([baseline_df, deletion_df], ignore_index=True)
