import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import DefaultDataCollator



class DependencyMapGenerator:
    def __init__(self, glm_model_wrapper, type, use_deletions=True):
        # type can be 'snp' or 'indel' or removal
        self.type = type
        self.use_deletions = use_deletions
        self.tokenizer = glm_model_wrapper.tokenizer
        self.device = glm_model_wrapper.device
        self.model = glm_model_wrapper.model.to(self.device)
        self.model.eval()

        self.add_special_tokens = getattr(glm_model_wrapper, 'add_special_tokens', True)

        self.nuc_table = {"-": 0, "A": 1, "C": 2, "G": 3, "T": 4} if use_deletions else {"A": 0, "C": 1, "G": 2, "T": 3}
        self.acgt_idxs = [self.tokenizer.get_vocab()[nuc] for nuc in ['-', 'A', 'C', 'G', 'T']] if use_deletions else \
            [self.tokenizer.get_vocab()[nuc] for nuc in ['A', 'C', 'G', 'T']]
       
    def _mutate_sequence(self, seq):
        seq = seq.upper()
        mutated_sequences = {'seq': [], 'mutation_pos': [], 'nuc': [], 'var_nt_idx': []}

        mutated_sequences['seq'].append(seq)
        mutated_sequences['mutation_pos'].append(-1)
        mutated_sequences['nuc'].append('real sequence')
        mutated_sequences['var_nt_idx'].append(-1)

        mutate_until_position = len(seq)

        if self.type == 'snp': # substitution of ACGT to other ACGT
            for i in range(mutate_until_position):
                for nuc in ['A', 'C', 'G', 'T']:
                        if nuc != seq[i] and seq[i] in ['A', 'C', 'G', 'T']:
                            mutated_sequences['seq'].append(seq[:i] + nuc + seq[i + 1:])
                            mutated_sequences['mutation_pos'].append(i)
                            mutated_sequences['nuc'].append(nuc)
                            mutated_sequences['var_nt_idx'].append(self.nuc_table[nuc])
        elif self.type == 'indel': # mutate sequence by replacing ACGT with '-'
            if self.use_deletions is False:
                raise ValueError("Indel mutation type requires use_deletions=True")
            for i in range(mutate_until_position):
                if seq[i] in ['A', 'C', 'G', 'T']:
                    mutated_sequences['seq'].append(seq[:i] + '-' + seq[i + 1:])
                    mutated_sequences['mutation_pos'].append(i)
                    mutated_sequences['nuc'].append('-')
                    mutated_sequences['var_nt_idx'].append(self.nuc_table['-'])
        elif self.type == 'removal':  # remove nucleotide, pad '-' at end
            if self.use_deletions == True:
                for i in range(mutate_until_position):
                    mutated_sequences['seq'].append(seq[:i] + seq[i + 1:] + "-")
                    mutated_sequences['mutation_pos'].append(i)
                    mutated_sequences['nuc'].append('deletion')
                    mutated_sequences['var_nt_idx'].append(self.nuc_table['-'])
            else:
                for i in range(mutate_until_position):
                    mutated_sequences['seq'].append(seq[:i] + seq[i + 1:] + "A")
                    mutated_sequences['mutation_pos'].append(i)
                    mutated_sequences['nuc'].append('deletion')
                    mutated_sequences['var_nt_idx'].append(self.nuc_table['A'])

            
        return pd.DataFrame(mutated_sequences)

    def _tok_func(self, x):
        ids = self.tokenizer.encode(x["seq"], add_special_tokens=self.add_special_tokens)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    def _create_dataloader(self, dataset, batch_size=64):
        ds = Dataset.from_pandas(dataset[['seq']])

        tok_ds = ds.map(lambda x: self._tok_func(x), batched=False, num_proc=None)

        rem_tok_ds = tok_ds.remove_columns('seq')

        data_collator = DefaultDataCollator()
        return torch.utils.data.DataLoader(
            rem_tok_ds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=data_collator
        )

    def _model_inference(self, data_loader):
        output_arrays = []
        for batch in data_loader:
            tokens = batch['input_ids']
            with torch.autocast(device_type=self.device.type):
                with torch.no_grad():
                    outputs = self.model(tokens.to(self.device)).logits.cpu().to(torch.float32)

            output_probs = torch.nn.functional.softmax(outputs, dim=-1)[:, :, self.acgt_idxs]
            output_arrays.append(output_probs)

        snp_reconstruct = torch.concat(output_arrays, axis=0)
        return snp_reconstruct.to(torch.float32).numpy()

    @staticmethod
    def reinsert_zero_row_probs(mut_probs_L: np.ndarray, i: int, epsilon=1e-10) -> np.ndarray:
        L, V = mut_probs_L.shape
        out = np.zeros((L + 1, V), dtype=mut_probs_L.dtype)

        out[:i, :] = mut_probs_L[:i, :]
        out[i+1:, :] = mut_probs_L[i:, :]  # shift down by 1 starting at i

        out[i, :] = epsilon
        out[i, :] = out[i, :] / out[i, :].sum()

        out = out[:-1, :]  # drop last row (pad)
        return out
    
    def compute_map(self, seq, epsilon=1e-10):
        dataset = self._mutate_sequence(seq)
        data_loader = self._create_dataloader(dataset, batch_size=16)
        snp_reconstruct = self._model_inference(data_loader)

        model_seq_len = snp_reconstruct.shape[1]
        input_len = len(seq)

        if model_seq_len == input_len:
            pass
        elif model_seq_len == input_len + 2:
            snp_reconstruct = snp_reconstruct[:, 1:-1, :]  # Remove [CLS] and [SEP]
        else:
            diff = model_seq_len - input_len
            if diff > 0 and diff % 2 == 0:
                trim = diff // 2
                snp_reconstruct = snp_reconstruct[:, trim:-trim, :]
            else:
                raise ValueError(f"Cannot automatically determine slicing. Input: {input_len}, Output: {model_seq_len}")

        if snp_reconstruct.shape[1] != len(seq):
            raise ValueError(f"Length mismatch after slicing: len(seq)={len(seq)} vs model={snp_reconstruct.shape[1]}")

        snp_reconstruct = snp_reconstruct + epsilon
        snp_reconstruct = snp_reconstruct / snp_reconstruct.sum(axis=-1)[:, :, np.newaxis]

        seq_len = snp_reconstruct.shape[1]
        snp_effect = np.zeros((seq_len, seq_len, 5 if self.use_deletions else 4, 5 if self.use_deletions else 4))

        reference_probs = snp_reconstruct[dataset[dataset['nuc'] == 'real sequence'].index[0]]

        # reference row (real sequence)
        ref_row = dataset.index[dataset['nuc'] == 'real sequence'][0]
        reference_probs = snp_reconstruct[ref_row]

        # --- aligned removal: realign every mutant row 1..N ---
        if self.type == "removal":
            for k, pos in enumerate(dataset.iloc[1:]['mutation_pos'].values, start=1):
                snp_reconstruct[k] = self.reinsert_zero_row_probs(
                    snp_reconstruct[k], int(pos), epsilon=epsilon
                )

        snp_effect[dataset.iloc[1:]['mutation_pos'].values, :, dataset.iloc[1:]['var_nt_idx'].values, :] = \
            np.log2(snp_reconstruct[1:]) - np.log2(1 - snp_reconstruct[1:]) \
            - np.log2(reference_probs) + np.log2(1 - reference_probs)

        if self.type == "removal":
            # pick deletion at pos i=0 (row k=1 if your dataset ordering is simple)
            i = int(dataset.iloc[1]['mutation_pos'])
            before = self._model_inference(self._create_dataloader(dataset.iloc[[1]], batch_size=1))[0]  # optional
            after  = snp_reconstruct[1]
            print("aligned deletion pos:", i, "shape:", after.shape)
               
        dep_map = np.max(np.abs(snp_effect), axis=(2, 3))
        np.fill_diagonal(dep_map, 0)

        return dep_map

    def plot(self, matrix, sequence=None, plot_size=10, vmax=None, annot=False):
        plt.figure(figsize=(plot_size, plot_size))

        ax = sns.heatmap(matrix, cmap="coolwarm", vmax=vmax, annot=annot,
                         fmt=".2f", annot_kws={"size": 8},
                         xticklabels=False, yticklabels=False)

        if sequence:
            tick_positions = np.arange(len(sequence)) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(list(sequence), rotation=0, fontsize=8)
            ax.set_yticklabels(list(sequence), rotation=0, fontsize=8)

        ax.set_aspect('equal')
        plt.show()

    def analyze(self, sequence, vmax=None, annot=False, show_plot=True):
        print("Computing dependency map...")
        dep_map = self.compute_map(sequence)
        if show_plot:
            self.plot(dep_map, sequence=sequence, vmax=vmax, annot=annot)
        return dep_map
