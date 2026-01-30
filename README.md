# Project6 – DNA GLM (Indels) 

generate_data.ipynb can be used to generate data for training and testing etc (evaluation as well).... 

visualize_data.ipynb can be used to train the model and visualize reconstruction of the model.

# EVALUATION PART ONLY

- Uses TWO FASTA files (REF + ALT) that share the same ordering / seq_id:
  - **REFERENCE: no-deletions reference sequences**
    **ALTERNATIVE: corresponding alternative sequences (with '-' etc.), SAME INDEX as REFERENCE**
- Samples EXACTLY:
    * 100 sequences with label=A_only
    * 100 sequences with label=B_only
    * 100 sequences with label=both
  Sampling is random (seeded) but indices are then sorted ascending (smallest -> biggest).
- Evaluates ONLY WITHIN MOTIF POSITIONS for:
    * **Cross-entropy (CE) (masked pseudo-CE)**
    * **PLL "fitness" (mean log-prob)**
    * **elta log-likelihood (delta), and ref_sum/alt_sum (ALL motif-only)**
    * **Influence score: perturbations (query positions where ref!=alt) -> target positions ONLY within motif**


---

## Quick start (what you need)

- A trained model folder (example: `dna_bert_final/`) containing:
 - `config.json`
 - `tokenizer.json`
 - `training_metadata.json`
 - model weights (`pytorch_model.bin` or `model.safetensors`)
 - etc...
- A FASTA dataset file used for loading / sampling sequences  
 (same format as the training FASTA, with metadata stored in the header line)

---

### `stats.py`
This script is used to summarize dataset characteristics and/or output distributions.
I use it after `main_all_in_one_evaluation.py` to quickly sanity-check:
- how many sequences per label exist
- deletion statistics
- motif position distributions
- any basic CSV summaries that help interpret results

(So: **main runs the evaluation part**, **stats summarizes the data/results**.)

---

## How to run (recommended)

### 1) Run the full pipeline
From the project root:

```bash
python3 main_all_in_one_evaluation.py and then clean.py and plots_evaluation.py
