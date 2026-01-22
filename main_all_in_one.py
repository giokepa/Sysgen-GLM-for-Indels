#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_all_in_one.py
"""

import os
import csv
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from glm_model import GLMModel
from sequence_plotter import plot as plot_reconstruction


# ============================================================
# CONFIG (EDIT PATHS)
# ============================================================
MODEL_DIR = "/Users/amelielaura/Documents/dna_bert_final"
FASTA_FILE = "/Users/amelielaura/Documents/Project6/data/augumented_sequence_size10000_length150_deletions0.2_nodeletionseq0.1.fasta"
OUT_DIR   = "/Users/amelielaura/Documents/Project6/outputs/all_results_final"

GLOBAL_SEED = 727

# Reconstruction example (optional)
RECON_HEADER = "seq0082|label=both|posAmotif=17|posBmotif=58|gaplength=30|deletions=24"
RECON_SEQ    = "GTAT---TTAGTGTGGCATATTCACTACTC-TTCGGACCATTG-TACG-AAAAC-ACCGTACTGCG-TGA-TCCCCTCATAG-CGCA-A-A-TGTGTGGTAGT-C-GC-C-G-GCC--GCTAAAAGG---GAATTGTGTGC-TCACTAGG"

# Evaluation limits
MAX_EVAL_SEQS     = 300
CE_STRIDE         = 1
CE_MAX_POS        = 150
PLL_STRIDE        = 1
PLL_MAX_POS       = 150

# Dependency maps
DEP_MAPS_PER_CLASS = 10
DEP_METRIC         = "max_abs_logodds"
DEP_MODES          = ["deletion", "mutation", "removal"]

# Plot appearance
HEATMAP_DPI  = 220

DNA = ["A", "C", "G", "T"]


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_fasta(path: str) -> Tuple[List[str], List[str]]:
    headers, seqs = [], []
    with open(path) as f:
        h = None
        buf: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if h is not None:
                    headers.append(h)
                    seqs.append("".join(buf))
                h = line[1:]
                buf = []
            else:
                buf.append(line)
        if h is not None:
            headers.append(h)
            seqs.append("".join(buf))
    return headers, seqs


def write_fasta(headers: List[str], seqs: List[str], out_path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    with open(out_path, "w") as f:
        for h, s in zip(headers, seqs):
            f.write(f">{h}\n{s}\n")


def get_field(header: str, key: str) -> Optional[str]:
    for p in header.split("|"):
        if p.startswith(key + "="):
            return p.split("=", 1)[1]
    return None


def get_label(header: str) -> str:
    return get_field(header, "label") or "unknown"


def get_int_field(header: str, key: str) -> Optional[int]:
    v = get_field(header, key)
    if v is None or v == "None":
        return None
    return int(v)


@dataclass
class SeqMeta:
    seq_id: str
    label: str
    posA: Optional[int]
    posB: Optional[int]
    gap: Optional[int]
    deletions_hdr: Optional[int]

    @staticmethod
    def from_header(header: str) -> "SeqMeta":
        return SeqMeta(
            seq_id=header.split("|")[0],
            label=get_label(header),
            posA=get_int_field(header, "posAmotif"),
            posB=get_int_field(header, "posBmotif"),
            gap=get_int_field(header, "gaplength"),
            deletions_hdr=get_int_field(header, "deletions"),
        )


def restore_reference_from_alt(alt_seq: str, rng: random.Random) -> str:
    """Replace '-' with random A/C/G/T so ref has no deletions but SAME LENGTH."""
    s = list(alt_seq)
    for i, c in enumerate(s):
        if c == "-":
            s[i] = rng.choice(DNA)
    return "".join(s)


# ============================================================
# GLM helper wrapper
# ============================================================
class GLMHelper:
    def __init__(self, glm: GLMModel):
        if glm.model is None:
            raise RuntimeError("GLMModel not loaded.")
        self.glm = glm
        self.device = glm.device

        # choose an UNK-like replacement token for "removal"
        vocab = getattr(self.glm.tokenizer, "vocab", {})
        unk_token = getattr(self.glm.tokenizer, "unk_token", None)
        if unk_token is None:
            unk_token = "N" if "N" in vocab else "-"
        if unk_token not in vocab and unk_token != "-":
            unk_token = "-"
        self.unk_token = unk_token

        self.glm.model.eval()

    def tokenize(self, seq: str) -> Dict[str, torch.Tensor]:
        return self.glm.tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=getattr(self.glm, "add_special_tokens", False),
        ).to(self.device)

    def get_offsets(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        ids = input_ids[0].tolist()
        left_offset = 0
        if self.glm.tokenizer.cls_token_id is not None and len(ids) > 0 and ids[0] == self.glm.tokenizer.cls_token_id:
            left_offset = 1
        right_excl = len(ids)
        if self.glm.tokenizer.sep_token_id is not None and len(ids) > 0 and ids[-1] == self.glm.tokenizer.sep_token_id:
            right_excl = len(ids) - 1
        return left_offset, right_excl

    def masked_probs_full_vocab(self, seq: str, j: int) -> torch.Tensor:
        """
        Returns full-vocab probabilities at masked position j.
        Used only to compute max_abs_logodds shift via log-softmax over full vocab.
        """
        s = list(seq)
        s[j] = "[MASK]"
        inputs = self.tokenize("".join(s))

        mask_id = self.glm.tokenizer.mask_token_id
        input_ids = inputs["input_ids"][0]
        mpos = (input_ids == mask_id).nonzero(as_tuple=True)[0]
        if len(mpos) != 1:
            raise RuntimeError(f"Expected 1 mask, found {len(mpos)}")
        mask_pos = int(mpos[0].item())

        with torch.no_grad():
            logits = self.glm.model(**inputs).logits[0, mask_pos]
        return F.softmax(logits, dim=-1)

    @staticmethod
    def shift_score_max_abs_logodds(p_ref: torch.Tensor, p_alt: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        return torch.max(torch.abs(torch.log(p_alt + eps) - torch.log(p_ref + eps)))

    def pll_mean_loglik(self, seq: str, max_positions: int = 150, stride: int = 1) -> Dict[str, Any]:
        orig = self.tokenize(seq)
        orig_ids = orig["input_ids"]
        left_off, right_excl = self.get_offsets(orig_ids)

        L = len(seq)
        positions = list(range(0, min(L, max_positions), stride))

        total_logp = 0.0
        count = 0

        with torch.no_grad():
            for p in positions:
                masked = list(seq)
                masked[p] = "[MASK]"
                inputs = self.tokenize("".join(masked))

                input_ids = inputs["input_ids"][0]
                mask_id = self.glm.tokenizer.mask_token_id
                mpos = (input_ids == mask_id).nonzero(as_tuple=True)[0]
                if len(mpos) != 1:
                    continue
                mask_pos = int(mpos[0].item())

                logits = self.glm.model(**inputs).logits[0, mask_pos]
                logp = F.log_softmax(logits, dim=-1)

                tok_pos = left_off + p
                if tok_pos < left_off or tok_pos >= right_excl:
                    continue

                true_id = int(orig_ids[0, tok_pos].item())
                lp = float(logp[true_id].item())
                if not math.isfinite(lp):
                    continue

                total_logp += lp
                count += 1

        mean_logp = float("nan") if count == 0 else float(total_logp / count)
        return {"fitness": mean_logp, "n_positions": count}


# ============================================================
# Eval metrics 
# ============================================================
def mlm_pseudo_cross_entropy(H: GLMHelper, seq: str, max_positions: int = 150, stride: int = 1) -> Dict[str, Any]:
    orig = H.tokenize(seq)
    orig_ids = orig["input_ids"]
    left_off, right_excl = H.get_offsets(orig_ids)

    L = len(seq)
    positions = list(range(0, min(L, max_positions), stride))

    nll_sum = 0.0
    count = 0

    with torch.no_grad():
        for p in positions:
            masked_list = list(seq)
            masked_list[p] = "[MASK]"
            inputs = H.tokenize("".join(masked_list))
            input_ids = inputs["input_ids"][0]

            mask_id = H.glm.tokenizer.mask_token_id
            mpos = (input_ids == mask_id).nonzero(as_tuple=True)[0]
            if len(mpos) != 1:
                continue
            mask_pos = int(mpos[0].item())

            logits = H.glm.model(**inputs).logits[0, mask_pos]
            logp = F.log_softmax(logits, dim=-1)

            tok_pos = left_off + p
            if tok_pos < left_off or tok_pos >= right_excl:
                continue

            true_id = int(orig_ids[0, tok_pos].item())
            nll = -float(logp[true_id].item())
            if not math.isfinite(nll):
                continue

            nll_sum += nll
            count += 1

    ce = float("nan") if count == 0 else float(nll_sum / count)
    return {"cross_entropy": ce, "n_positions": count}


def delta_likelihood_fast(H: GLMHelper, reference_sequence: str, perturbed_sequence: str) -> Dict[str, Any]:
    if len(reference_sequence) != len(perturbed_sequence):
        raise ValueError("ref and alt must have same length.")

    ref_inputs = H.tokenize(reference_sequence)
    alt_inputs = H.tokenize(perturbed_sequence)

    with torch.no_grad():
        ref_logits = H.glm.model(**ref_inputs).logits[0]
        alt_logits = H.glm.model(**alt_inputs).logits[0]

    ref_logp = F.log_softmax(ref_logits, dim=-1)
    alt_logp = F.log_softmax(alt_logits, dim=-1)

    ref_ids = ref_inputs["input_ids"][0]
    alt_ids = alt_inputs["input_ids"][0]
    left_off, right_excl = H.get_offsets(ref_inputs["input_ids"])

    tok_start = left_off
    tok_end = right_excl
    idx = torch.arange(tok_start, tok_end, device=H.device)

    ref_sum = float(ref_logp[idx, ref_ids[idx]].sum().item())
    alt_sum = float(alt_logp[idx, alt_ids[idx]].sum().item())
    delta = alt_sum - ref_sum

    return {"delta": delta, "reference_sum": ref_sum, "perturbed_sum": alt_sum}


def influence_probability_shift(H: GLMHelper, reference_sequence: str, perturbed_sequence: str) -> Dict[str, Any]:
    if len(reference_sequence) != len(perturbed_sequence):
        raise ValueError("ref and alt must have same length.")

    query_positions = [i for i, (a, b) in enumerate(zip(reference_sequence, perturbed_sequence)) if a != b]
    targets = list(range(0, len(reference_sequence)))

    total = 0.0
    for q in query_positions:
        per_target: List[float] = []
        for j in targets:
            if j == q:
                continue
            p_ref = H.masked_probs_full_vocab(reference_sequence, j)
            p_alt = H.masked_probs_full_vocab(perturbed_sequence, j)
            score = H.shift_score_max_abs_logodds(p_ref, p_alt)
            per_target.append(float(score.item()))
        total += float(np.mean(per_target)) if per_target else 0.0

    return {
        "influence_score": float(total),
        "query_positions": query_positions,
        "target_window": (0, len(reference_sequence)),
        "metric": "max_abs_logodds",
        "reduce": "mean",
    }


# ============================================================
# Dependency maps (3 modes)
# ============================================================
class DependencyMapComputer:
    """
    Computes M[i,j]: disrupt position i, measure shift at j using max_abs_logodds.
    modes:
      - deletion : i -> '-'
      - mutation : i -> average effect over A/C/G/T substitutions
      - removal  : i -> unk_token (fallback to '-' or 'N')
    """
    def __init__(self, helper: GLMHelper):
        self.H = helper

    def disrupt_seq_deletion(self, ref: str, i: int) -> str:
        s = list(ref)
        s[i] = "-"
        return "".join(s)

    def disrupt_seq_removal(self, ref: str, i: int) -> str:
        s = list(ref)
        s[i] = self.H.unk_token
        return "".join(s)

    def disrupt_seq_mutations(self, ref: str, i: int) -> List[str]:
        orig = ref[i]
        muts = []
        for b in DNA:
            if b == orig:
                continue
            s = list(ref)
            s[i] = b
            muts.append("".join(s))
        if not muts:
            for b in DNA:
                s = list(ref)
                s[i] = b
                muts.append("".join(s))
        return muts

    def compute_map(self, ref_seq: str, mode: str, set_diag_nan: bool = True) -> np.ndarray:
        if DEP_METRIC != "max_abs_logodds":
            raise ValueError("This pipeline supports only max_abs_logodds.")

        L = len(ref_seq)
        M = np.zeros((L, L), dtype=np.float32)

        # cache p_ref(j) for all j (FULL VOCAB, no A/C/G/T filtering)
        ref_probs = [self.H.masked_probs_full_vocab(ref_seq, j) for j in range(L)]

        for i in range(L):
            if mode == "deletion":
                alt_seq = self.disrupt_seq_deletion(ref_seq, i)
                for j in range(L):
                    if j == i:
                        M[i, j] = np.nan if set_diag_nan else 0.0
                        continue
                    p_alt = self.H.masked_probs_full_vocab(alt_seq, j)
                    M[i, j] = float(self.H.shift_score_max_abs_logodds(ref_probs[j], p_alt).item())

            elif mode == "removal":
                alt_seq = self.disrupt_seq_removal(ref_seq, i)
                for j in range(L):
                    if j == i:
                        M[i, j] = np.nan if set_diag_nan else 0.0
                        continue
                    p_alt = self.H.masked_probs_full_vocab(alt_seq, j)
                    M[i, j] = float(self.H.shift_score_max_abs_logodds(ref_probs[j], p_alt).item())

            elif mode == "mutation":
                mutated = self.disrupt_seq_mutations(ref_seq, i)
                for j in range(L):
                    if j == i:
                        M[i, j] = np.nan if set_diag_nan else 0.0
                        continue
                    scores = []
                    for alt_seq in mutated:
                        p_alt = self.H.masked_probs_full_vocab(alt_seq, j)
                        scores.append(float(self.H.shift_score_max_abs_logodds(ref_probs[j], p_alt).item()))
                    M[i, j] = float(np.mean(scores)) if scores else 0.0

            else:
                raise ValueError(f"Unknown mode: {mode}")

        return M


# ============================================================
# Heatmap plotting 
# ============================================================
class HeatmapPlotter:
    def plot(
        self,
        M: np.ndarray,
        out_png: str,
        title: str,
        posA: Optional[int],
        posB: Optional[int],
        dpi: int = HEATMAP_DPI,
    ) -> None:
        ensure_dir(os.path.dirname(os.path.abspath(out_png)))

        fig, ax = plt.subplots(figsize=(12, 9))
        im = ax.imshow(M, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("target position j")
        ax.set_ylabel("disruption position i")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if posA is not None:
            ax.axvline(posA, linewidth=1)
            ax.axhline(posA, linewidth=1)
            ax.text(posA + 1, 1, "Motif A", rotation=90, va="bottom", fontsize=10)
        if posB is not None:
            ax.axvline(posB, linewidth=1)
            ax.axhline(posB, linewidth=1)
            ax.text(posB + 1, 1, "Motif B", rotation=90, va="bottom", fontsize=10)

        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    ensure_dir(OUT_DIR)

    out_recon = os.path.join(OUT_DIR, "reconstruction")
    out_eval  = os.path.join(OUT_DIR, "evaluation")
    out_maps  = os.path.join(OUT_DIR, "dependency_maps")
    ensure_dir(out_recon)
    ensure_dir(out_eval)
    ensure_dir(out_maps)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    print("Loading GLMModel...")
    glm = GLMModel(model_path=MODEL_DIR, fasta_file=FASTA_FILE, force_retrain=False)
    if glm.model is None:
        raise RuntimeError("Model not loaded.")

    H = GLMHelper(glm)
    dep_comp = DependencyMapComputer(H)
    plotter = HeatmapPlotter()

    # ------------------------------------------------------------
    # Load FASTA
    # ------------------------------------------------------------
    headers, seqs = load_fasta(FASTA_FILE)
    if len(headers) != len(seqs):
        raise RuntimeError("FASTA parse mismatch: headers != seqs")
    print(f"Loaded {len(seqs)} sequences.")

    # ------------------------------------------------------------
    # (A) Reconstruction plot 
    # ------------------------------------------------------------
    print("Running reconstruction plot...")
    prob_matrix = glm.get_full_reconstruction_probs(RECON_SEQ)
    plot_reconstruction(RECON_HEADER, RECON_SEQ, prob_matrix, motif_length=7)
    recon_png = os.path.join(out_recon, "reconstruction_seq0082.png")
    fig = plt.gcf()
    fig.savefig(recon_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("[OK] Saved:", recon_png)

    # ------------------------------------------------------------
    # (B) Evaluation 
    # ------------------------------------------------------------
    print("\n[B] Running evaluation on deletion-containing sequences...")

    rng = random.Random(GLOBAL_SEED + 999)
    del_indices = [i for i, s in enumerate(seqs) if "-" in s]
    rng.shuffle(del_indices)
    del_indices = del_indices[: min(MAX_EVAL_SEQS, len(del_indices))]
    print(f"Evaluating {len(del_indices)} sequences (max {MAX_EVAL_SEQS}).")

    eval_rows: List[Dict[str, Any]] = []
    ref_headers_out: List[str] = []
    ref_seqs_out: List[str] = []

    for k, idx in enumerate(del_indices, start=1):
        h = headers[idx]
        alt = seqs[idx]
        meta = SeqMeta.from_header(h)

        ref = restore_reference_from_alt(alt, rng)

        ce_ref = mlm_pseudo_cross_entropy(H, ref, max_positions=CE_MAX_POS, stride=CE_STRIDE)
        ce_alt = mlm_pseudo_cross_entropy(H, alt, max_positions=CE_MAX_POS, stride=CE_STRIDE)

        dlt = delta_likelihood_fast(H, ref, alt)
        infl = influence_probability_shift(H, ref, alt)

        fit_ref = H.pll_mean_loglik(ref, max_positions=PLL_MAX_POS, stride=PLL_STRIDE)
        fit_alt = H.pll_mean_loglik(alt, max_positions=PLL_MAX_POS, stride=PLL_STRIDE)

        row = {
            "index": idx,
            "id": meta.seq_id,
            "label": meta.label,
            "posA": meta.posA,
            "posB": meta.posB,
            "gap": meta.gap,
            "deletions_hdr": meta.deletions_hdr,
            "deletions_actual": alt.count("-"),

            "ce_ref": ce_ref["cross_entropy"],
            "ce_alt": ce_alt["cross_entropy"],
            "ce_npos_ref": ce_ref["n_positions"],
            "ce_npos_alt": ce_alt["n_positions"],

            "fitness_pll_mean_logp_ref": fit_ref["fitness"],
            "fitness_pll_mean_logp_alt": fit_alt["fitness"],
            "fitness_npos_ref": fit_ref["n_positions"],
            "fitness_npos_alt": fit_alt["n_positions"],

            "delta_loglik": dlt["delta"],
            "ref_sum": dlt["reference_sum"],
            "alt_sum": dlt["perturbed_sum"],

            "influence_score": infl["influence_score"],
            "metric": infl["metric"],
        }
        eval_rows.append(row)

        ref_header = (
            f"{meta.seq_id}|label={meta.label}|posAmotif={meta.posA}|posBmotif={meta.posB}|"
            f"gaplength={meta.gap}|restored_from={meta.seq_id}|original_deletions={alt.count('-')}"
        )
        ref_headers_out.append(ref_header)
        ref_seqs_out.append(ref)

        if k % 25 == 0 or k == len(del_indices):
            print(f"  [{k}/{len(del_indices)}] done")

    eval_csv = os.path.join(out_eval, "eval_all_sequences.csv")
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        w.writeheader()
        w.writerows(eval_rows)
    print(f"[OK] Saved: {eval_csv}")

    ref_fasta = os.path.join(out_eval, "restored_references.fasta")
    write_fasta(ref_headers_out, ref_seqs_out, ref_fasta)
    print(f"[OK] Saved: {ref_fasta}")

    # ------------------------------------------------------------
    # (C) Dependency maps 
    # ------------------------------------------------------------
    print("\n[C] Running dependency maps...")

    classes = ["A_only", "B_only", "both"]

    by_label: Dict[str, List[int]] = {c: [] for c in classes}
    for i, h in enumerate(headers):
        lab = get_label(h)
        if lab in by_label:
            by_label[lab].append(i)

    manifest_rows: List[Dict[str, Any]] = []

    for cls in classes:
        idxs_cls = by_label.get(cls, [])
        if not idxs_cls:
            print(f"[WARN] no sequences for class {cls}")
            continue

        rng.shuffle(idxs_cls)
        chosen = idxs_cls[: min(DEP_MAPS_PER_CLASS, len(idxs_cls))]
        print(f"  {cls}: {len(chosen)} sequences")

        for idx in chosen:
            h = headers[idx]
            s = seqs[idx]
            meta = SeqMeta.from_header(h)

            base_title = (
                f"{meta.seq_id} | label={meta.label} | "
                f"Motif A pos={meta.posA} | Motif B pos={meta.posB} | "
                f"gap={meta.gap} | deletions={s.count('-')}"
            )

            for mode in DEP_MODES:
                mode_dir = os.path.join(out_maps, cls, mode)
                ensure_dir(mode_dir)

                M = dep_comp.compute_map(s, mode=mode, set_diag_nan=True)

                out_npy = os.path.join(mode_dir, f"{meta.seq_id}__dep_{DEP_METRIC}.npy")
                np.save(out_npy, M)

                out_png = os.path.join(mode_dir, f"{meta.seq_id}__dep_{DEP_METRIC}.png")
                title = f"Dependency map ({DEP_METRIC}) | {mode} | {base_title}"
                plotter.plot(M, out_png, title, posA=meta.posA, posB=meta.posB, dpi=HEATMAP_DPI)

                manifest_rows.append({
                    "class": cls,
                    "mode": mode,
                    "metric": DEP_METRIC,
                    "id": meta.seq_id,
                    "posA": meta.posA,
                    "posB": meta.posB,
                    "length": len(s),
                    "deletions_actual": s.count("-"),
                    "npy_path": out_npy,
                    "png_path": out_png,
                })

    manifest_csv = os.path.join(out_maps, "manifest.csv")
    if manifest_rows:
        with open(manifest_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
        print("[OK] Saved:", manifest_csv)

    print("DONE")
    print("OUT_DIR:", OUT_DIR)


if __name__ == "__main__":
    main()
