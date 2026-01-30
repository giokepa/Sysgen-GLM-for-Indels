#!/usr/bin/env python3
import os
import csv
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F

from glm_model import GLMModel


# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "/Users/amelielaura/Documents/dna_bert_final"
FASTA_FILE = "/Users/amelielaura/Documents/Project6/data/augumented_sequence_size10000_length150_deletions0.2_nodeletionseq0.1.fasta"
OUT_DIR = "/Users/amelielaura/Documents/Project6/outputs/dependency_maps_only"

GLOBAL_SEED = 727

DEP_MAPS_PER_CLASS = 10
DEP_METRIC = "max_abs_logodds"
DEP_MODES = ["deletion", "mutation", "removal"]

HEATMAP_DPI = 220
LOGO_HEIGHT = 1.1

DNA = ["A", "C", "G", "T"]
RELEVANT_CHARS = ["A", "C", "G", "T", "-"]


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_fasta(path: str):
    headers, seqs = [], []
    with open(path) as f:
        h, buf = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if h is not None:
                    headers.append(h)
                    seqs.append("".join(buf))
                h, buf = line[1:], []
            else:
                buf.append(line)
        if h is not None:
            headers.append(h)
            seqs.append("".join(buf))
    return headers, seqs


def get_field(header: str, key: str) -> Optional[str]:
    for p in header.split("|"):
        if p.startswith(key + "="):
            return p.split("=", 1)[1]
    return None


def get_label(header: str) -> str:
    return get_field(header, "label") or "unknown"


def get_int_field(header: str, key: str) -> Optional[int]:
    v = get_field(header, key)
    return None if v in (None, "None") else int(v)


@dataclass
class SeqMeta:
    seq_id: str
    label: str
    posA: Optional[int]
    posB: Optional[int]

    @staticmethod
    def from_header(header: str) -> "SeqMeta":
        return SeqMeta(
            seq_id=header.split("|")[0],
            label=get_label(header),
            posA=get_int_field(header, "posAmotif"),
            posB=get_int_field(header, "posBmotif"),
        )


# ============================================================
# GLM helper
# ============================================================
class GLMHelper:
    def __init__(self, glm: GLMModel):
        self.glm = glm
        self.device = glm.device
        self.rel_ids = torch.tensor(
            [glm.tokenizer.vocab[c] for c in RELEVANT_CHARS],
            device=self.device,
        )
        self.glm.model.eval()

    def tokenize(self, seq: str):
        return self.glm.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

    def masked_probs(self, seq: str, j: int):
        s = list(seq)
        s[j] = "[MASK]"
        inputs = self.tokenize("".join(s))
        input_ids = inputs["input_ids"][0]
        mask_pos = (input_ids == self.glm.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

        with torch.no_grad():
            logits = self.glm.model(**inputs).logits[0, mask_pos]

        p = F.softmax(logits, dim=-1)[self.rel_ids]
        return p / p.sum()

    @staticmethod
    def max_abs_logodds(p_ref, p_alt, eps=1e-9):
        return torch.max(torch.abs(torch.log(p_alt + eps) - torch.log(p_ref + eps)))


# ============================================================
# Dependency maps
# ============================================================
class DependencyMapComputer:
    def __init__(self, H: GLMHelper):
        self.H = H

    def compute(self, seq: str, mode: str) -> np.ndarray:
        L = len(seq)
        M = np.zeros((L, L), dtype=np.float32)
        ref_probs = [self.H.masked_probs(seq, j) for j in range(L)]

        for i in range(L):
            if mode == "deletion":
                alt_seqs = [seq[:i] + "-" + seq[i+1:]]
            elif mode == "removal":
                alt_seqs = [seq[:i] + "N" + seq[i+1:]]
            elif mode == "mutation":
                alt_seqs = [seq[:i] + b + seq[i+1:] for b in DNA if b != seq[i]]
            else:
                raise ValueError(mode)

            for j in range(L):
                if i == j:
                    M[i, j] = np.nan
                    continue
                scores = []
                for alt in alt_seqs:
                    p_alt = self.H.masked_probs(alt, j)
                    scores.append(self.H.max_abs_logodds(ref_probs[j], p_alt).item())
                M[i, j] = float(np.mean(scores))

        return M


# ============================================================
# Plotting
# ============================================================
class HeatmapPlotter:
    def plot(self, M, out_png, title, posA=None, posB=None):
        fig, ax = plt.subplots(figsize=(12, 9))
        im = ax.imshow(M, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("target position j")
        ax.set_ylabel("disruption position i")
        plt.colorbar(im, ax=ax)

        if posA is not None:
            ax.axvline(posA)
            ax.axhline(posA)
        if posB is not None:
            ax.axvline(posB)
            ax.axhline(posB)

        fig.tight_layout()
        fig.savefig(out_png, dpi=HEATMAP_DPI)
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    random.seed(GLOBAL_SEED)
    ensure_dir(OUT_DIR)

    glm = GLMModel(model_path=MODEL_DIR, fasta_file=FASTA_FILE)
    H = GLMHelper(glm)
    dep = DependencyMapComputer(H)
    plotter = HeatmapPlotter()

    headers, seqs = load_fasta(FASTA_FILE)

    classes = ["A_only", "B_only", "both"]
    by_label = {c: [] for c in classes}
    for i, h in enumerate(headers):
        lab = get_label(h)
        if lab in by_label:
            by_label[lab].append(i)

    manifest = []

    for cls in classes:
        for idx in by_label[cls][:DEP_MAPS_PER_CLASS]:
            meta = SeqMeta.from_header(headers[idx])
            seq = seqs[idx]

            for mode in DEP_MODES:
                out_dir = os.path.join(OUT_DIR, cls, mode)
                ensure_dir(out_dir)

                M = dep.compute(seq, mode)
                npy = os.path.join(out_dir, f"{meta.seq_id}__dep_{DEP_METRIC}.npy")
                png = os.path.join(out_dir, f"{meta.seq_id}__dep_{DEP_METRIC}.png")

                np.save(npy, M)
                plotter.plot(M, png, f"{meta.seq_id} | {mode}", meta.posA, meta.posB)

                manifest.append({
                    "id": meta.seq_id,
                    "class": cls,
                    "mode": mode,
                    "npy": npy,
                    "png": png,
                })

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader()
        w.writerows(manifest)

    print("DONE")


if __name__ == "__main__":
    main()
