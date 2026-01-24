#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_all_in_one_evaluation.py

EVALUATION ONLY

- Uses TWO FASTA files (REF + ALT) that share the same ordering / seq_id:
    * REF: no-deletions reference sequences
    * ALT: corresponding alternative sequences (with '-' etc.), SAME INDEX as REF
- Samples EXACTLY:
    * 100 sequences with label=A_only
    * 100 sequences with label=B_only
    * 100 sequences with label=both
  Sampling is random (seeded) but indices are then sorted ascending (smallest -> biggest).
- Evaluates ONLY WITHIN MOTIF POSITIONS for:
    * Cross-entropy (CE) (masked pseudo-CE)
    * PLL "fitness" (mean log-prob)
    * Delta log-likelihood (delta), and ref_sum/alt_sum (ALL motif-only)
    * Influence score: perturbations (query positions where ref!=alt) -> target positions ONLY within motif

"""

import os
import csv
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np

import torch
import torch.nn.functional as F

from glm_model import GLMModel


# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "/Users/amelielaura/Documents/dna_bert_final"
TRAIN_FASTA_FOR_MODEL = "/Users/amelielaura/Documents/Project6/data/augumented_sequence_size10000_length150_deletions0.2_nodeletionseq0.1.fasta"
EVAL_REF_FASTA = "/Users/amelielaura/Documents/Project6/data/sequence_size10000_length150_deletions0_nodeletionseq1_seed538.fasta"
EVAL_ALT_FASTA = "/Users/amelielaura/Documents/Project6/data/augmented_fromfile_fixedlen_sequence_size10000_length150_deletions0.2_nodeletionseq0.1_seed538.fasta"

OUT_DIR = "/Users/amelielaura/Documents/Project6/outputs/eval_only_motif_based"

GLOBAL_SEED = 727

# Exactly this many per label
N_PER_CLASS = 100

# Motif lengths
MOTIF_A_LEN = 7
MOTIF_B_LEN = 7

CE_STRIDE = 1
PLL_STRIDE = 1
COMPUTE_INFLUENCE = True


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

    @staticmethod
    def from_header(header: str) -> "SeqMeta":
        return SeqMeta(
            seq_id=header.split("|")[0],
            label=get_label(header),
            posA=get_int_field(header, "posAmotif"),
            posB=get_int_field(header, "posBmotif"),
            gap=get_int_field(header, "gaplength"),
        )


def motif_positions(meta: SeqMeta, seq_len: int) -> List[int]:
    pos: Set[int] = set()

    if meta.label == "A_only":
        if meta.posA is None:
            raise ValueError("A_only but posA is None")
        for i in range(meta.posA, meta.posA + MOTIF_A_LEN):
            if 0 <= i < seq_len:
                pos.add(i)

    elif meta.label == "B_only":
        if meta.posB is None:
            raise ValueError("B_only but posB is None")
        for i in range(meta.posB, meta.posB + MOTIF_B_LEN):
            if 0 <= i < seq_len:
                pos.add(i)

    elif meta.label == "both":
        if meta.posA is None or meta.posB is None:
            raise ValueError("both but posA/posB is None")
        for i in range(meta.posA, meta.posA + MOTIF_A_LEN):
            if 0 <= i < seq_len:
                pos.add(i)
        for i in range(meta.posB, meta.posB + MOTIF_B_LEN):
            if 0 <= i < seq_len:
                pos.add(i)

    else:
        raise ValueError(f"Unsupported label for motif-based evaluation: {meta.label}")

    return sorted(pos)


def pick_indices_exact_per_class_with_alt_deletions(
    headers: List[str],
    alt_seqs: List[str],
    n_per_class: int,
    seed: int
) -> List[int]:
    """
    Randomly pick exactly n_per_class indices for each of A_only, B_only, both,
    BUT ONLY from indices where ALT contains at least one deletion: '-' in alt_seqs[i].
    Then return combined indices sorted ascending.
    """
    rng = random.Random(seed)
    wanted = ["A_only", "B_only", "both"]
    pools: Dict[str, List[int]] = {k: [] for k in wanted}

    for i, h in enumerate(headers):
        lab = get_label(h)
        if lab not in pools:
            continue
        if alt_seqs[i].count("-") <= 0:
            continue
        pools[lab].append(i)

    chosen_all: List[int] = []
    for lab in wanted:
        pool = pools[lab]
        if len(pool) < n_per_class:
            raise RuntimeError(
                f"Not enough ALT-with-deletions sequences for {lab}: have {len(pool)}, need {n_per_class}. "
                f"(ALT must contain '-')"
            )
        rng.shuffle(pool)
        chosen_all.extend(pool[:n_per_class])

    return sorted(chosen_all)


# ============================================================
# GLM helper wrapper
# ============================================================
class GLMHelper:
    def __init__(self, glm: GLMModel):
        if glm.model is None:
            raise RuntimeError("GLMModel not loaded.")
        self.glm = glm
        self.device = glm.device
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

    def pll_mean_loglik_on_positions(self, seq: str, positions: List[int], stride: int = 1) -> Dict[str, Any]:
        if not positions:
            return {"fitness": float("nan")}

        orig = self.tokenize(seq)
        orig_ids = orig["input_ids"]
        left_off, right_excl = self.get_offsets(orig_ids)

        pos_eval = positions[::stride] if stride > 1 else positions

        total_logp = 0.0
        count = 0

        with torch.no_grad():
            for p in pos_eval:
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
        return {"fitness": mean_logp}


# ============================================================
# Eval metrics (motif-only)
# ============================================================
def mlm_pseudo_cross_entropy_on_positions(H: GLMHelper, seq: str, positions: List[int], stride: int = 1) -> Dict[str, Any]:
    if not positions:
        return {"cross_entropy": float("nan")}

    orig = H.tokenize(seq)
    orig_ids = orig["input_ids"]
    left_off, right_excl = H.get_offsets(orig_ids)

    pos_eval = positions[::stride] if stride > 1 else positions

    nll_sum = 0.0
    count = 0

    with torch.no_grad():
        for p in pos_eval:
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
    return {"cross_entropy": ce}


def delta_likelihood_motif_only(H: GLMHelper, ref_seq: str, alt_seq: str, positions: List[int]) -> Dict[str, Any]:
    if len(ref_seq) != len(alt_seq):
        raise ValueError("ref and alt must have same length.")
    if not positions:
        return {"delta": float("nan"), "reference_sum": float("nan"), "perturbed_sum": float("nan")}

    ref_inputs = H.tokenize(ref_seq)
    alt_inputs = H.tokenize(alt_seq)

    with torch.no_grad():
        ref_logits = H.glm.model(**ref_inputs).logits[0]
        alt_logits = H.glm.model(**alt_inputs).logits[0]

    ref_logp = F.log_softmax(ref_logits, dim=-1)
    alt_logp = F.log_softmax(alt_logits, dim=-1)

    ref_ids = ref_inputs["input_ids"][0]
    alt_ids = alt_inputs["input_ids"][0]
    left_off, right_excl = H.get_offsets(ref_inputs["input_ids"])

    tok_positions: List[int] = []
    for p in positions:
        tp = left_off + p
        if left_off <= tp < right_excl:
            tok_positions.append(tp)

    if not tok_positions:
        return {"delta": float("nan"), "reference_sum": float("nan"), "perturbed_sum": float("nan")}

    idx = torch.tensor(tok_positions, device=H.device, dtype=torch.long)

    ref_sum = float(ref_logp[idx, ref_ids[idx]].sum().item())
    alt_sum = float(alt_logp[idx, alt_ids[idx]].sum().item())
    return {"delta": alt_sum - ref_sum, "reference_sum": ref_sum, "perturbed_sum": alt_sum}


def influence_to_motif_only(H: GLMHelper, ref_seq: str, alt_seq: str, motif_pos: List[int]) -> Dict[str, Any]:
    if len(ref_seq) != len(alt_seq):
        raise ValueError("ref and alt must have same length.")
    if not motif_pos:
        return {"influence_score": float("nan"), "query_positions": [], "targets_n": 0}

    query_positions = [i for i, (a, b) in enumerate(zip(ref_seq, alt_seq)) if a != b]
    if not query_positions:
        # If ALT must always have deletions, this should not happen, but keep it safe.
        return {"influence_score": 0.0, "query_positions": [], "targets_n": len(motif_pos)}

    total = 0.0
    for q in query_positions:
        scores: List[float] = []
        for j in motif_pos:
            if j == q:
                continue
            p_ref = H.masked_probs_full_vocab(ref_seq, j)
            p_alt = H.masked_probs_full_vocab(alt_seq, j)
            score = H.shift_score_max_abs_logodds(p_ref, p_alt)
            scores.append(float(score.item()))
        total += float(np.mean(scores)) if scores else 0.0

    return {
        "influence_score": float(total),
        "query_positions": query_positions,
        "targets_n": len(motif_pos),
        "metric": "max_abs_logodds",
        "reduce": "mean_over_targets_then_sum_over_queries",
    }


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    ensure_dir(OUT_DIR)
    out_eval = os.path.join(OUT_DIR, "evaluation")
    ensure_dir(out_eval)

    print("Loading GLMModel...")
    glm = GLMModel(model_path=MODEL_DIR, fasta_file=TRAIN_FASTA_FOR_MODEL, force_retrain=False)
    if glm.model is None:
        raise RuntimeError("Model not loaded.")
    H = GLMHelper(glm)

    ref_headers, ref_seqs = load_fasta(EVAL_REF_FASTA)
    alt_headers, alt_seqs = load_fasta(EVAL_ALT_FASTA)

    if len(ref_headers) != len(alt_headers) or len(ref_seqs) != len(alt_seqs):
        raise RuntimeError("Ref/Alt FASTA sizes mismatch.")

    for i, (hr, ha) in enumerate(zip(ref_headers, alt_headers)):
        id_r = hr.split("|")[0]
        id_a = ha.split("|")[0]
        if id_r != id_a:
            raise RuntimeError(f"Ref/Alt seq_id mismatch at i={i}: {id_r} != {id_a}")

    # NEW: sample only indices where ALT has deletions (>0), per class
    chosen_indices = pick_indices_exact_per_class_with_alt_deletions(
        headers=ref_headers,
        alt_seqs=alt_seqs,
        n_per_class=N_PER_CLASS,
        seed=GLOBAL_SEED + 999,
    )

    eval_rows: List[Dict[str, Any]] = []

    for k, idx in enumerate(chosen_indices, start=1):
        h_ref = ref_headers[idx]
        ref = ref_seqs[idx]
        alt = alt_seqs[idx]

        if len(ref) != len(alt):
            raise RuntimeError(f"Length mismatch at idx={idx}: len(ref)={len(ref)} len(alt)={len(alt)}")

        # ENFORCE: REF no deletions, ALT has deletions
        if ref.count("-") != 0:
            raise RuntimeError(f"REF must have 0 deletions, but idx={idx} has ref.count('-')={ref.count('-')}")
        if alt.count("-") <= 0:
            raise RuntimeError(f"ALT must have >0 deletions, but idx={idx} has alt.count('-')={alt.count('-')}")

        meta = SeqMeta.from_header(h_ref)
        if meta.label not in {"A_only", "B_only", "both"}:
            continue

        motif_pos = motif_positions(meta, seq_len=len(ref))

        ce_ref = mlm_pseudo_cross_entropy_on_positions(H, ref, positions=motif_pos, stride=CE_STRIDE)
        ce_alt = mlm_pseudo_cross_entropy_on_positions(H, alt, positions=motif_pos, stride=CE_STRIDE)

        fit_ref = H.pll_mean_loglik_on_positions(ref, positions=motif_pos, stride=PLL_STRIDE)
        fit_alt = H.pll_mean_loglik_on_positions(alt, positions=motif_pos, stride=PLL_STRIDE)

        dlt_motif = delta_likelihood_motif_only(H, ref, alt, positions=motif_pos)

        infl = {"influence_score": float("nan"), "query_positions": [], "targets_n": len(motif_pos), "metric": "max_abs_logodds"}
        if COMPUTE_INFLUENCE:
            infl = influence_to_motif_only(H, ref, alt, motif_pos)

        row = {
            "index": idx,
            "id": meta.seq_id,
            "label": meta.label,
            "posA": meta.posA,
            "posB": meta.posB,
            "gap": meta.gap,

            "deletions_ref": ref.count("-"),
            "deletions_alt": alt.count("-"),

            "motif_positions_n": len(motif_pos),

            "ce_ref_motif": ce_ref["cross_entropy"],
            "ce_alt_motif": ce_alt["cross_entropy"],

            "fitness_pll_mean_logp_ref_motif": fit_ref["fitness"],
            "fitness_pll_mean_logp_alt_motif": fit_alt["fitness"],

            "ref_sum_motif": dlt_motif["reference_sum"],
            "alt_sum_motif": dlt_motif["perturbed_sum"],
            "delta_loglik_motif": dlt_motif["delta"],

            "influence_score_motif": infl["influence_score"],
            "influence_metric": infl.get("metric", "max_abs_logodds"),
            "n_query_positions": len(infl.get("query_positions", [])),

            "ref_sequence_used": ref,
            "alt_sequence_used": alt,
        }
        eval_rows.append(row)

        if k % 25 == 0 or k == len(chosen_indices):
            print(f"  [{k}/{len(chosen_indices)}] done")

    if not eval_rows:
        raise RuntimeError("No evaluation rows produced. Check labels and FASTA headers.")

    # keep sequences as last two columns
    fieldnames = [k for k in eval_rows[0].keys() if k not in ("ref_sequence_used", "alt_sequence_used")]
    fieldnames += ["ref_sequence_used", "alt_sequence_used"]

    eval_csv = os.path.join(out_eval, f"eval_motif_only__ALTmustHaveDeletions__A{N_PER_CLASS}_B{N_PER_CLASS}_both{N_PER_CLASS}.csv")
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(eval_rows)

    print(f"[OK] Saved: {eval_csv}")
    print("DONE")
    print("OUT_DIR:", OUT_DIR)


if __name__ == "__main__":
    main()

