# Project6 – DNA GLM (Indels) 

generate_data.ipynb can be used to generate data for training and testing. 
visualize_data.ipynb can be used to train the model and visualize reconstruction of the model.

This repository contains my end-to-end workflow for testing a trained DNA BERT/GLM model on:
1) **Reconstruction plots** (per-position reconstruction probabilities)  
2) **Model-level evaluation on a validation split** (MLM loss + perplexity)  
3) **Ref vs. Alt scoring** (delta likelihood + influence / probability shift)  
4) **Dependency maps** (how much a deletion at position *i* changes predictions at position *j*)

Everything is built to run locally in PyCharm and also headless (cluster-safe), because all plots use a non-interactive Matplotlib backend (`Agg`) and are saved directly to disk.

---

## Quick start (what you need)

- A trained model folder (example: `dna_bert_final/`) containing:
 - `config.json`
 - `tokenizer.json`
 - `training_metadata.json`
 - model weights (`pytorch_model.bin` or `model.safetensors`)

- A FASTA dataset file used for loading / sampling sequences  
 (same format as the training FASTA, with metadata stored in the header line)

---

## What each script does

### `main_all_in_one.py`  “run everything”
This is the main integration script. It:
- loads the trained `GLMModel`
- creates a **reconstruction plot** for a chosen example sequence
- builds a reproducible **train/val split**
- runs **MLM validation quality** (loss + perplexity)
- runs **ref vs alt scoring** (delta-likelihood + influence score)
- generates **dependency maps** for many sequences (e.g. 50 `A_only` + 50 `B_only`)
- writes all outputs into one result folder (plots + CSVs + manifest)

Outputs typically include:
- `reconstruction_*.png`
- `model_quality.csv`
- `eval_ref_alt.csv`
- `dependency_maps/` folder:
 - `.png` heatmaps
 - `.npy` matrices
 - `manifest.csv` pointing to all files
 - `*_input.txt` storing the exact sequence used (for traceability)

---

### `stats.py`
This script is used to summarize dataset characteristics and/or output distributions.
I use it after `main_all_in_one.py` to quickly sanity-check:
- how many sequences per label exist
- deletion statistics
- motif position distributions
- any basic CSV summaries that help interpret results

(So: **main runs the experiments**, **stats summarizes the data/results**.)

---

## How to run (recommended)

### 1) Run the full pipeline
From the project root:

```bash
python3 main_all_in_one.py
