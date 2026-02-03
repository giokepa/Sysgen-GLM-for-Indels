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
            'baseline': {
                'cross_entropies': [],
                'perplexities': [],
                'sequence_ids': [],
                'motif_cross_entropies': []
            },
            'deletion': {
                'cross_entropies': [],
                'perplexities': [],
                'sequence_ids': [],
                'motif_cross_entropies': []
            },
            'baseline_4nt': {
                'cross_entropies': [],
                'perplexities': [],
                'sequence_ids': []
            },
            'deletion_4nt': {
                'cross_entropies': [],
                'perplexities': [],
                'sequence_ids': []
            }
        }

    def extract_motif_positions(self, header: str) -> List[Tuple[int, int]]:
        motif_positions = []
        parts = header.split('|')

        motif_length = 6

        for part in parts:
            if 'posAmotif=' in part:
                pos_str = part.split('=')[1]
                if pos_str.strip().lower() == 'none' or not pos_str.strip():
                    continue
                for pos in pos_str.split(','):
                    pos = pos.strip()
                    if pos and pos.lower() != 'none':
                        try:
                            start = int(pos)
                            motif_positions.append((start, start + motif_length))
                        except ValueError:
                            continue
            elif 'posBmotif=' in part:
                pos_str = part.split('=')[1]
                if pos_str.strip().lower() == 'none' or not pos_str.strip():
                    continue
                for pos in pos_str.split(','):
                    pos = pos.strip()
                    if pos and pos.lower() != 'none':
                        try:
                            start = int(pos)
                            motif_positions.append((start, start + motif_length))
                        except ValueError:
                            continue

        return motif_positions

    def compute_cross_entropy(self, model: GLMModel, sequence: str,
                              motif_positions: List[Tuple[int, int]] = None,
                              four_nt_only: bool = False) -> float:

        total_ce = 0.0
        valid_positions = 0

        if motif_positions and len(motif_positions) > 0:
            positions_to_eval = []
            for start, end in motif_positions:
                positions_to_eval.extend(range(start, min(end, len(sequence))))
        else:
            positions_to_eval = range(len(sequence))

        if four_nt_only:
            valid_chars = ['A', 'C', 'G', 'T']
        else:
            valid_chars = model.relevant_chars

        for pos in positions_to_eval:
            if pos >= len(sequence):
                continue

            true_char = sequence[pos]

            if true_char not in valid_chars:
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
        results_key_4nt = 'baseline_4nt' if model_type == 'baseline' else 'deletion_4nt'

        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_type.upper()} model on: {test_fasta_path}")
        print(f"{'=' * 60}")

        dataset = DNADataset(test_fasta_path, model.tokenizer, model.max_length)
        n_sequences = len(dataset) if max_sequences is None else min(max_sequences, len(dataset))

        for idx in range(n_sequences):
            header, sequence = dataset.get_raw_sequence(idx)

            motif_positions = self.extract_motif_positions(header)

            motif_ce = self.compute_cross_entropy(model, sequence, motif_positions=motif_positions, four_nt_only=False)
            motif_perplexity = self.compute_perplexity(motif_ce)

            ce_4nt = self.compute_cross_entropy(model, sequence, motif_positions=None, four_nt_only=True)
            perplexity_4nt = self.compute_perplexity(ce_4nt)

            self.results[model_type]['motif_cross_entropies'].append(motif_ce)
            self.results[model_type]['cross_entropies'].append(motif_ce)
            self.results[model_type]['perplexities'].append(motif_perplexity)
            self.results[model_type]['sequence_ids'].append(header.split('|')[0].replace('>', ''))

            self.results[results_key_4nt]['cross_entropies'].append(ce_4nt)
            self.results[results_key_4nt]['perplexities'].append(perplexity_4nt)
            self.results[results_key_4nt]['sequence_ids'].append(header.split('|')[0].replace('>', ''))

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{n_sequences} sequences...")

        motif_ce_array = np.array(self.results[model_type]['motif_cross_entropies'])
        perp_array = np.array(self.results[model_type]['perplexities'])

        print(f"\n{model_type.upper()} Model Results (MOTIF-ONLY):")
        print(f"  Cross-Entropy: {motif_ce_array.mean():.4f} ± {motif_ce_array.std():.4f}")
        print(f"  Perplexity: {perp_array.mean():.4f} ± {perp_array.std():.4f}")
        print(f"  Min CE: {motif_ce_array.min():.4f}, Max CE: {motif_ce_array.max():.4f}")

        ce_4nt_array = np.array(self.results[results_key_4nt]['cross_entropies'])
        print(f"\n{model_type.upper()} Model Results (4-NUCLEOTIDE COMPARISON):")
        print(f"  Cross-Entropy: {ce_4nt_array.mean():.4f} ± {ce_4nt_array.std():.4f}")

    def compare_models(self) -> Dict:
        print(f"\n{'=' * 60}")
        print("STATISTICAL COMPARISON (MOTIF-ONLY)")
        print(f"{'=' * 60}")

        baseline_ce = np.array(self.results['baseline']['motif_cross_entropies'])
        deletion_ce = np.array(self.results['deletion']['motif_cross_entropies'])

        if len(baseline_ce) == 0 or len(deletion_ce) == 0:
            raise ValueError("Both models must be evaluated before comparison")

        comparison_results = {'motif_only': {}, '4nt_comparison': {}}

        if len(baseline_ce) == len(deletion_ce):
            t_stat, t_pvalue = stats.ttest_rel(baseline_ce, deletion_ce)
            comparison_results['motif_only']['paired_ttest'] = {
                'statistic': float(t_stat),
                'pvalue': float(t_pvalue),
                'significant': t_pvalue < 0.05
            }
            print(f"\nPaired t-test (Motif-only):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {t_pvalue:.6e}")
            print(f"  Significant (α=0.05): {t_pvalue < 0.05}")

        if len(baseline_ce) == len(deletion_ce):
            w_stat, w_pvalue = stats.wilcoxon(baseline_ce, deletion_ce)
            comparison_results['motif_only']['wilcoxon'] = {
                'statistic': float(w_stat),
                'pvalue': float(w_pvalue),
                'significant': w_pvalue < 0.05
            }
            print(f"\nWilcoxon signed-rank test (Motif-only):")
            print(f"  W-statistic: {w_stat:.4f}")
            print(f"  p-value: {w_pvalue:.6e}")
            print(f"  Significant (α=0.05): {w_pvalue < 0.05}")

        mean_diff = baseline_ce.mean() - deletion_ce.mean()
        pooled_std = np.sqrt((baseline_ce.std() ** 2 + deletion_ce.std() ** 2) / 2)
        cohens_d = mean_diff / pooled_std
        comparison_results['motif_only']['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
        print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
        print(f"  Interpretation: {self._interpret_cohens_d(cohens_d)}")

        print(f"\n{'=' * 60}")
        print("4-NUCLEOTIDE COMPARISON (Fair Comparison)")
        print(f"{'=' * 60}")

        baseline_4nt = np.array(self.results['baseline_4nt']['cross_entropies'])
        deletion_4nt = np.array(self.results['deletion_4nt']['cross_entropies'])

        if len(baseline_4nt) == len(deletion_4nt):
            t_stat_4nt, t_pvalue_4nt = stats.ttest_rel(baseline_4nt, deletion_4nt)
            comparison_results['4nt_comparison']['paired_ttest'] = {
                'statistic': float(t_stat_4nt),
                'pvalue': float(t_pvalue_4nt),
                'significant': t_pvalue_4nt < 0.05
            }
            print(f"\nPaired t-test (4-nucleotide):")
            print(f"  t-statistic: {t_stat_4nt:.4f}")
            print(f"  p-value: {t_pvalue_4nt:.6e}")
            print(f"  Significant (α=0.05): {t_pvalue_4nt < 0.05}")

            w_stat_4nt, w_pvalue_4nt = stats.wilcoxon(baseline_4nt, deletion_4nt)
            comparison_results['4nt_comparison']['wilcoxon'] = {
                'statistic': float(w_stat_4nt),
                'pvalue': float(w_pvalue_4nt),
                'significant': w_pvalue_4nt < 0.05
            }
            print(f"\nWilcoxon signed-rank test (4-nucleotide):")
            print(f"  W-statistic: {w_stat_4nt:.4f}")
            print(f"  p-value: {w_pvalue_4nt:.6e}")
            print(f"  Significant (α=0.05): {w_pvalue_4nt < 0.05}")

        mean_diff_4nt = baseline_4nt.mean() - deletion_4nt.mean()
        pooled_std_4nt = np.sqrt((baseline_4nt.std() ** 2 + deletion_4nt.std() ** 2) / 2)
        cohens_d_4nt = mean_diff_4nt / pooled_std_4nt
        comparison_results['4nt_comparison']['effect_size'] = {
            'cohens_d': float(cohens_d_4nt),
            'interpretation': self._interpret_cohens_d(cohens_d_4nt)
        }
        print(f"\nEffect Size (Cohen's d): {cohens_d_4nt:.4f}")
        print(f"  Interpretation: {self._interpret_cohens_d(cohens_d_4nt)}")

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
                'motif_cross_entropies': [float(x) for x in self.results['baseline']['motif_cross_entropies']],
                'perplexities': [float(x) for x in self.results['baseline']['perplexities']],
                'sequence_ids': self.results['baseline']['sequence_ids'],
                'summary': {
                    'mean_motif_ce': float(np.mean(self.results['baseline']['motif_cross_entropies'])),
                    'std_motif_ce': float(np.std(self.results['baseline']['motif_cross_entropies'])),
                    'mean_perplexity': float(np.mean(self.results['baseline']['perplexities']))
                }
            },
            'deletion': {
                'motif_cross_entropies': [float(x) for x in self.results['deletion']['motif_cross_entropies']],
                'perplexities': [float(x) for x in self.results['deletion']['perplexities']],
                'sequence_ids': self.results['deletion']['sequence_ids'],
                'summary': {
                    'mean_motif_ce': float(np.mean(self.results['deletion']['motif_cross_entropies'])),
                    'std_motif_ce': float(np.std(self.results['deletion']['motif_cross_entropies'])),
                    'mean_perplexity': float(np.mean(self.results['deletion']['perplexities']))
                }
            },
            'baseline_4nt': {
                'cross_entropies': [float(x) for x in self.results['baseline_4nt']['cross_entropies']],
                'sequence_ids': self.results['baseline_4nt']['sequence_ids'],
                'summary': {
                    'mean_ce': float(np.mean(self.results['baseline_4nt']['cross_entropies'])),
                    'std_ce': float(np.std(self.results['baseline_4nt']['cross_entropies']))
                }
            },
            'deletion_4nt': {
                'cross_entropies': [float(x) for x in self.results['deletion_4nt']['cross_entropies']],
                'sequence_ids': self.results['deletion_4nt']['sequence_ids'],
                'summary': {
                    'mean_ce': float(np.mean(self.results['deletion_4nt']['cross_entropies'])),
                    'std_ce': float(np.std(self.results['deletion_4nt']['cross_entropies']))
                }
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults exported to: {output_path}")

    @classmethod
    def load_results(cls, json_path: str, baseline_model: GLMModel = None, deletion_model: GLMModel = None):
        with open(json_path, 'r') as f:
            data = json.load(f)

        evaluator = cls(baseline_model, deletion_model)

        if 'motif_cross_entropies' in data['baseline']:
            evaluator.results['baseline']['motif_cross_entropies'] = data['baseline']['motif_cross_entropies']
        elif 'cross_entropies' in data['baseline']:
            evaluator.results['baseline']['motif_cross_entropies'] = data['baseline']['cross_entropies']

        evaluator.results['baseline']['perplexities'] = data['baseline']['perplexities']
        evaluator.results['baseline']['sequence_ids'] = data['baseline']['sequence_ids']
        evaluator.results['baseline']['cross_entropies'] = evaluator.results['baseline']['motif_cross_entropies']

        if 'motif_cross_entropies' in data['deletion']:
            evaluator.results['deletion']['motif_cross_entropies'] = data['deletion']['motif_cross_entropies']
        elif 'cross_entropies' in data['deletion']:
            evaluator.results['deletion']['motif_cross_entropies'] = data['deletion']['cross_entropies']

        evaluator.results['deletion']['perplexities'] = data['deletion']['perplexities']
        evaluator.results['deletion']['sequence_ids'] = data['deletion']['sequence_ids']
        evaluator.results['deletion']['cross_entropies'] = evaluator.results['deletion']['motif_cross_entropies']

        if 'baseline_4nt' in data:
            evaluator.results['baseline_4nt']['cross_entropies'] = data['baseline_4nt']['cross_entropies']
            evaluator.results['baseline_4nt']['sequence_ids'] = data['baseline_4nt']['sequence_ids']
            evaluator.results['baseline_4nt']['perplexities'] = [np.exp(ce) if ce != float('inf') else float('inf')
                                                                 for ce in data['baseline_4nt']['cross_entropies']]
        else:
            print("Warning: No 4-nucleotide comparison data found in JSON. Will use motif data.")
            evaluator.results['baseline_4nt']['cross_entropies'] = evaluator.results['baseline'][
                'motif_cross_entropies']
            evaluator.results['baseline_4nt']['sequence_ids'] = evaluator.results['baseline']['sequence_ids']
            evaluator.results['baseline_4nt']['perplexities'] = evaluator.results['baseline']['perplexities']

        if 'deletion_4nt' in data:
            evaluator.results['deletion_4nt']['cross_entropies'] = data['deletion_4nt']['cross_entropies']
            evaluator.results['deletion_4nt']['sequence_ids'] = data['deletion_4nt']['sequence_ids']
            evaluator.results['deletion_4nt']['perplexities'] = [np.exp(ce) if ce != float('inf') else float('inf')
                                                                 for ce in data['deletion_4nt']['cross_entropies']]
        else:
            evaluator.results['deletion_4nt']['cross_entropies'] = evaluator.results['deletion'][
                'motif_cross_entropies']
            evaluator.results['deletion_4nt']['sequence_ids'] = evaluator.results['deletion']['sequence_ids']
            evaluator.results['deletion_4nt']['perplexities'] = evaluator.results['deletion']['perplexities']

        print(f"✓ Results loaded from {json_path}")
        return evaluator

    def get_results_dataframe(self) -> pd.DataFrame:
        baseline_motif_df = pd.DataFrame({
            'sequence_id': self.results['baseline']['sequence_ids'],
            'model': 'baseline',
            'comparison_type': 'motif_only',
            'cross_entropy': self.results['baseline']['motif_cross_entropies'],
            'perplexity': self.results['baseline']['perplexities']
        })

        deletion_motif_df = pd.DataFrame({
            'sequence_id': self.results['deletion']['sequence_ids'],
            'model': 'deletion',
            'comparison_type': 'motif_only',
            'cross_entropy': self.results['deletion']['motif_cross_entropies'],
            'perplexity': self.results['deletion']['perplexities']
        })

        baseline_4nt_df = pd.DataFrame({
            'sequence_id': self.results['baseline_4nt']['sequence_ids'],
            'model': 'baseline',
            'comparison_type': '4nt',
            'cross_entropy': self.results['baseline_4nt']['cross_entropies'],
            'perplexity': self.results['baseline_4nt']['perplexities']
        })

        deletion_4nt_df = pd.DataFrame({
            'sequence_id': self.results['deletion_4nt']['sequence_ids'],
            'model': 'deletion',
            'comparison_type': '4nt',
            'cross_entropy': self.results['deletion_4nt']['cross_entropies'],
            'perplexity': self.results['deletion_4nt']['perplexities']
        })

        return pd.concat([baseline_motif_df, deletion_motif_df,
                          baseline_4nt_df, deletion_4nt_df], ignore_index=True)

