import os
import csv
import math
import random
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from glm_model import GLMModel

MODEL_DIR = "/Users/amelielaura/Documents/dna_bert_final_new"

TRAIN_FASTA_FOR_MODEL = (
    "/Users/amelielaura/Documents/Project6/data/"
    "augumented_sequence_size10000_length150_deletions0.2_nodeletionseq0.1.fasta"
)

EVAL_REF_FASTA = "/Users/amelielaura/Documents/Project6/data/sequence_size10000_length150_deletions0_nodeletionseq1_seed538.fasta"
EVAL_ALT_FASTA = "/Users/amelielaura/Documents/Project6/data/augmented_fixedlen_DUPBOTH_deletions0.2_seed538.fasta"

OUT_DIR = "/Users/amelielaura/Documents/Project6/outputs/eval_only_motif_based/evaluation_232_paired"

GLOBAL_SEED = 727

N_A_ONLY = 65
N_B_ONLY = 47
N_BOTH_FLANKS = 100
N_BOTH_BETWEEN = 20

MOTIF_A = "ATATTCA"
MOTIF_B = "GTACTGC"
MOTIF_A_LEN = len(MOTIF_A)
MOTIF_B_LEN = len(MOTIF_B)

CE_STRIDE = 1
PLL_STRIDE = 1
COMPUTE_INFLUENCE = True

EXISTING_EVAL_CSV_FOR_SELECTION: Optional[str] = None
SELECTION_MANIFEST_CSV = os.path.join(OUT_DIR, "chosen_232_manifest.csv")

def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
def load_fasta(path: str) -> Tuple[List[str], List[str]]:
    headers: List[str] = []
    sequences: List[str] = []

    with open(path) as handle:
        current_header: Optional[str] = None
        current_chunks: List[str] = []

        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith(">"):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(current_chunks))
                current_header = stripped[1:]
                current_chunks = []
            else:
                current_chunks.append(stripped)

        if current_header is not None:
            headers.append(current_header)
            sequences.append("".join(current_chunks))

    return headers, sequences


def get_field(header: str, key: str) -> Optional[str]:
    parts = header.split("|")
    for part in parts:
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return None


def get_label(header: str) -> str:
    label_value = get_field(header, "label")
    if label_value is None:
        return "unknown"
    return label_value


def get_int_field(header: str, key: str) -> Optional[int]:
    value = get_field(header, key)
    if value is None or value == "None":
        return None
    return int(value)

@dataclass
class SeqMeta:
    seq_id: str
    label: str
    posA: Optional[int]
    posB: Optional[int]
    gap: Optional[int]
    both_mode: Optional[str]

    @staticmethod
    def from_header(header: str) -> "SeqMeta":
        return SeqMeta(
            seq_id=header.split("|")[0],
            label=get_label(header),
            posA=get_int_field(header, "posAmotif"),
            posB=get_int_field(header, "posBmotif"),
            gap=get_int_field(header, "gaplength"),
            both_mode=get_field(header, "both_mode"),
        )


def motif_positions(meta: SeqMeta, seq_len: int) -> List[int]:
    positions: Set[int] = set()

    if meta.label == "A_only":
        if meta.posA is None:
            raise ValueError("A_only sequence but posA is None.")
        for index in range(meta.posA, meta.posA + MOTIF_A_LEN):
            if 0 <= index < seq_len:
                positions.add(index)

    elif meta.label == "B_only":
        if meta.posB is None:
            raise ValueError("B_only sequence but posB is None.")
        for index in range(meta.posB, meta.posB + MOTIF_B_LEN):
            if 0 <= index < seq_len:
                positions.add(index)

    elif meta.label == "both":
        if meta.posA is None or meta.posB is None:
            raise ValueError("both sequence but posA or posB is None.")
        for index in range(meta.posA, meta.posA + MOTIF_A_LEN):
            if 0 <= index < seq_len:
                positions.add(index)
        for index in range(meta.posB, meta.posB + MOTIF_B_LEN):
            if 0 <= index < seq_len:
                positions.add(index)

    else:
        raise ValueError(f"Unsupported label for motif_positions: {meta.label}")

    return sorted(positions)

def deletion_regions_for_both(meta_ref: SeqMeta, alt_seq: str) -> Dict[str, bool]:
    if meta_ref.label != "both":
        raise ValueError("deletion_regions_for_both called on non-both sequence.")

    if meta_ref.posA is None or meta_ref.posB is None:
        raise ValueError("both sequence but posA or posB is None.")

    a_start = meta_ref.posA
    a_end = a_start + MOTIF_A_LEN
    b_start = meta_ref.posB
    b_end = b_start + MOTIF_B_LEN

    left_region = alt_seq[0:a_start]
    gap_region = alt_seq[a_end:b_start]
    right_region = alt_seq[b_end:len(alt_seq)]

    has_left = "-" in left_region
    has_gap = "-" in gap_region
    has_right = "-" in right_region
    has_outside = has_left or has_right
    has_any = "-" in alt_seq

    return {
        "has_left": has_left,
        "has_gap": has_gap,
        "has_right": has_right,
        "has_outside": has_outside,
        "has_any": has_any,
    }


def classify_both_bucket(meta_ref: SeqMeta, alt_seq: str) -> Optional[str]:
    regions = deletion_regions_for_both(meta_ref, alt_seq)

    if not regions["has_any"]:
        return None

    if regions["has_outside"] and not regions["has_gap"]:
        return "both_flanks"
    if regions["has_gap"] and not regions["has_outside"]:
        return "both_between"

    return None

@dataclass
class PairRecord:
    seq_id: str
    label: str
    bucket: str
    meta_ref: SeqMeta
    ref_seq: str
    alt_seq: str
    alt_both_mode: Optional[str]


def build_pair_pools(
    ref_headers: List[str],
    ref_seqs: List[str],
    alt_headers: List[str],
    alt_seqs: List[str],
) -> Dict[str, List[PairRecord]]:
    ref_map: Dict[str, Tuple[SeqMeta, str]] = {}
    for header, seq in zip(ref_headers, ref_seqs):
        meta = SeqMeta.from_header(header)
        ref_map[meta.seq_id] = (meta, seq)

    alt_map: Dict[str, List[Tuple[SeqMeta, str]]] = {}
    for header, seq in zip(alt_headers, alt_seqs):
        meta = SeqMeta.from_header(header)
        if meta.seq_id not in alt_map:
            alt_map[meta.seq_id] = []
        alt_map[meta.seq_id].append((meta, seq))

    pools: Dict[str, List[PairRecord]] = {
        "A_only": [],
        "B_only": [],
        "both_flanks": [],
        "both_between": [],
    }

    for seq_id, (meta_ref, ref_seq) in ref_map.items():
        if ref_seq.count("-") != 0:
            continue

        alternatives = alt_map.get(seq_id, [])
        if not alternatives:
            continue

        if meta_ref.label in ("A_only", "B_only"):
            for meta_alt, alt_seq in alternatives:
                if meta_alt.label != meta_ref.label:
                    continue
                if alt_seq.count("-") <= 0:
                    continue

                record = PairRecord(
                    seq_id=seq_id,
                    label=meta_ref.label,
                    bucket=meta_ref.label,
                    meta_ref=meta_ref,
                    ref_seq=ref_seq,
                    alt_seq=alt_seq,
                    alt_both_mode=meta_alt.both_mode,
                )
                pools[meta_ref.label].append(record)
                break

        elif meta_ref.label == "both":
            for meta_alt, alt_seq in alternatives:
                if meta_alt.label != "both":
                    continue
                if alt_seq.count("-") <= 0:
                    continue

                bucket_name = classify_both_bucket(meta_ref, alt_seq)
                if bucket_name in ("both_flanks", "both_between"):
                    record = PairRecord(
                        seq_id=seq_id,
                        label="both",
                        bucket=bucket_name,
                        meta_ref=meta_ref,
                        ref_seq=ref_seq,
                        alt_seq=alt_seq,
                        alt_both_mode=meta_alt.both_mode,
                    )
                    pools[bucket_name].append(record)

    return pools

def deterministic_take(
    pairs: List[PairRecord],
    n: int,
    seed: int,
    bucket: str,
) -> List[PairRecord]:
    scored: List[Tuple[str, PairRecord]] = []

    for record in pairs:
        alt_mode = record.alt_both_mode or "none"
        unique_key = f"{record.seq_id}|{bucket}|{alt_mode}"
        hash_input = f"{seed}|{unique_key}".encode("utf-8")
        hash_value = hashlib.sha256(hash_input).hexdigest()
        scored.append((hash_value, record))

    scored.sort(key=lambda pair: pair[0])

    if len(scored) < n:
        raise RuntimeError(f"Not enough pairs for {bucket}: have {len(scored)}, need {n}.")

    selected: List[PairRecord] = []
    for hash_value, record in scored[:n]:
        selected.append(record)

    return selected


def save_manifest(manifest_csv: str, chosen: List[PairRecord]) -> None:
    directory = os.path.dirname(manifest_csv) or "."
    ensure_directory(directory)

    with open(manifest_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq_id", "bucket", "label", "alt_both_mode"])
        for record in chosen:
            writer.writerow([record.seq_id, record.bucket, record.label, record.alt_both_mode])


def load_manifest(manifest_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    with open(manifest_csv) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_csv}")

    return rows


def choose_pairs_auto(
    pools: Dict[str, List[PairRecord]],
    seed: int,
    existing_eval_csv: Optional[str],
    manifest_csv: str,
) -> Tuple[List[PairRecord], str]:
    expected_total = N_A_ONLY + N_B_ONLY + N_BOTH_FLANKS + N_BOTH_BETWEEN

    if existing_eval_csv and os.path.exists(existing_eval_csv):
        chosen: List[PairRecord] = []

        with open(existing_eval_csv) as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "id" not in reader.fieldnames or "bucket" not in reader.fieldnames:
                raise RuntimeError(
                    "Existing eval CSV must contain 'id' and 'bucket' columns "
                    "(and ideally 'alt_both_mode')."
                )

            for row in reader:
                seq_id = (row.get("id") or "").strip()
                bucket = (row.get("bucket") or "").strip()
                mode = (row.get("alt_both_mode") or "").strip() or None

                candidates = pools.get(bucket, [])
                match: Optional[PairRecord] = None
                for record in candidates:
                    same_id = record.seq_id == seq_id
                    same_mode = (record.alt_both_mode == mode) or (mode is None and record.alt_both_mode is None)
                    if same_id and same_mode:
                        match = record
                        break

                if match is None:
                    raise RuntimeError(
                        f"Could not match row from existing eval CSV: id={seq_id}, bucket={bucket}, mode={mode}"
                    )

                chosen.append(match)

        if len(chosen) != expected_total:
            raise RuntimeError(
                f"Existing eval CSV has {len(chosen)} rows, expected {expected_total}."
            )

        return chosen, f"existing_eval_csv:{existing_eval_csv}"

    if os.path.exists(manifest_csv):
        rows = load_manifest(manifest_csv)
        chosen: List[PairRecord] = []

        for row in rows:
            seq_id = (row.get("seq_id") or "").strip()
            bucket = (row.get("bucket") or "").strip()
            mode = (row.get("alt_both_mode") or "").strip() or None

            candidates = pools.get(bucket, [])
            match: Optional[PairRecord] = None
            for record in candidates:
                same_id = record.seq_id == seq_id
                same_mode = (record.alt_both_mode == mode) or (mode is None and record.alt_both_mode is None)
                if same_id and same_mode:
                    match = record
                    break

            if match is None:
                raise RuntimeError(
                    f"Could not match manifest row: seq_id={seq_id}, bucket={bucket}, mode={mode}"
                )

            chosen.append(match)

        if len(chosen) != expected_total:
            raise RuntimeError(f"Manifest has {len(chosen)} rows, expected {expected_total}.")

        return chosen, f"manifest:{manifest_csv}"

    chosen: List[PairRecord] = []
    chosen.extend(deterministic_take(pools["A_only"], N_A_ONLY, seed, "A_only"))
    chosen.extend(deterministic_take(pools["B_only"], N_B_ONLY, seed, "B_only"))
    chosen.extend(deterministic_take(pools["both_flanks"], N_BOTH_FLANKS, seed, "both_flanks"))
    chosen.extend(deterministic_take(pools["both_between"], N_BOTH_BETWEEN, seed, "both_between"))

    if len(chosen) != expected_total:
        raise RuntimeError(
            f"Created selection has {len(chosen)} rows, expected {expected_total}."
        )

    save_manifest(manifest_csv, chosen)
    return chosen, f"created:deterministic + saved_manifest:{manifest_csv}"

class GLMHelper:
    def __init__(self, glm: GLMModel):
        if glm.model is None:
            raise RuntimeError("GLMModel is not loaded.")
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

        if (
            self.glm.tokenizer.cls_token_id is not None
            and len(ids) > 0
            and ids[0] == self.glm.tokenizer.cls_token_id
        ):
            left_offset = 1

        right_excl = len(ids)
        if (
            self.glm.tokenizer.sep_token_id is not None
            and len(ids) > 0
            and ids[-1] == self.glm.tokenizer.sep_token_id
        ):
            right_excl = len(ids) - 1

        return left_offset, right_excl

    def masked_probs_full_vocab(self, seq: str, position: int) -> torch.Tensor:
        seq_list = list(seq)
        seq_list[position] = "[MASK]"
        inputs = self.tokenize("".join(seq_list))

        mask_id = self.glm.tokenizer.mask_token_id
        input_ids = inputs["input_ids"][0]
        mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) != 1:
            raise RuntimeError(f"Expected exactly 1 mask token, found {len(mask_positions)}")

        mask_pos = int(mask_positions[0].item())

        with torch.no_grad():
            logits = self.glm.model(**inputs).logits[0, mask_pos]

        return F.softmax(logits, dim=-1)

    @staticmethod
    def shift_score_max_abs_logodds(
        p_ref: torch.Tensor,
        p_alt: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        return torch.max(torch.abs(torch.log(p_alt + eps) - torch.log(p_ref + eps)))

    def pll_mean_loglik_on_positions(
        self,
        seq: str,
        positions: List[int],
        stride: int = 1,
    ) -> Dict[str, Any]:
        if not positions:
            return {"fitness": float("nan")}

        original_inputs = self.tokenize(seq)
        original_ids = original_inputs["input_ids"]
        left_off, right_excl = self.get_offsets(original_ids)

        if stride > 1:
            eval_positions = positions[::stride]
        else:
            eval_positions = positions

        total_logp = 0.0
        count = 0

        with torch.no_grad():
            for position in eval_positions:
                seq_list = list(seq)
                seq_list[position] = "[MASK]"
                inputs = self.tokenize("".join(seq_list))

                input_ids = inputs["input_ids"][0]
                mask_id = self.glm.tokenizer.mask_token_id
                mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]
                if len(mask_positions) != 1:
                    continue

                mask_pos = int(mask_positions[0].item())
                logits = self.glm.model(**inputs).logits[0, mask_pos]
                logp = F.log_softmax(logits, dim=-1)

                token_pos = left_off + position
                if token_pos < left_off or token_pos >= right_excl:
                    continue

                true_id = int(original_ids[0, token_pos].item())
                lp = float(logp[true_id].item())
                if not math.isfinite(lp):
                    continue

                total_logp += lp
                count += 1

        if count == 0:
            mean_logp = float("nan")
        else:
            mean_logp = float(total_logp / count)

        return {"fitness": mean_logp}


def mlm_pseudo_cross_entropy_on_positions(
    helper: GLMHelper,
    seq: str,
    positions: List[int],
    stride: int = 1,
) -> Dict[str, Any]:
    if not positions:
        return {"cross_entropy": float("nan")}

    original_inputs = helper.tokenize(seq)
    original_ids = original_inputs["input_ids"]
    left_off, right_excl = helper.get_offsets(original_ids)

    if stride > 1:
        eval_positions = positions[::stride]
    else:
        eval_positions = positions

    total_nll = 0.0
    count = 0

    with torch.no_grad():
        for position in eval_positions:
            seq_list = list(seq)
            seq_list[position] = "[MASK]"
            inputs = helper.tokenize("".join(seq_list))
            input_ids = inputs["input_ids"][0]

            mask_id = helper.glm.tokenizer.mask_token_id
            mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) != 1:
                continue

            mask_pos = int(mask_positions[0].item())
            logits = helper.glm.model(**inputs).logits[0, mask_pos]
            logp = F.log_softmax(logits, dim=-1)

            token_pos = left_off + position
            if token_pos < left_off or token_pos >= right_excl:
                continue

            true_id = int(original_ids[0, token_pos].item())
            nll = -float(logp[true_id].item())
            if not math.isfinite(nll):
                continue

            total_nll += nll
            count += 1

    if count == 0:
        ce = float("nan")
    else:
        ce = float(total_nll / count)

    return {"cross_entropy": ce}


def delta_likelihood_motif_only(
    helper: GLMHelper,
    ref_seq: str,
    alt_seq: str,
    positions: List[int],
) -> Dict[str, Any]:

    if len(ref_seq) != len(alt_seq):
        raise ValueError("ref_seq and alt_seq must have the same length.")

    if not positions:
        return {
            "delta": float("nan"),
            "reference_sum": float("nan"),
            "perturbed_sum": float("nan"),
        }

    ref_inputs = helper.tokenize(ref_seq)
    alt_inputs = helper.tokenize(alt_seq)

    with torch.no_grad():
        ref_logits = helper.glm.model(**ref_inputs).logits[0]
        alt_logits = helper.glm.model(**alt_inputs).logits[0]

    ref_logp = F.log_softmax(ref_logits, dim=-1)
    alt_logp = F.log_softmax(alt_logits, dim=-1)

    ref_ids = ref_inputs["input_ids"][0]
    alt_ids = alt_inputs["input_ids"][0]
    left_off, right_excl = helper.get_offsets(ref_inputs["input_ids"])

    token_positions: List[int] = []
    for position in positions:
        token_index = left_off + position
        if left_off <= token_index < right_excl:
            token_positions.append(token_index)

    if not token_positions:
        return {
            "delta": float("nan"),
            "reference_sum": float("nan"),
            "perturbed_sum": float("nan"),
        }

    idx = torch.tensor(token_positions, device=helper.device, dtype=torch.long)

    ref_sum = float(ref_logp[idx, ref_ids[idx]].sum().item())
    alt_sum = float(alt_logp[idx, alt_ids[idx]].sum().item())
    return {
        "delta": alt_sum - ref_sum,
        "reference_sum": ref_sum,
        "perturbed_sum": alt_sum,
    }


def influence_to_motif_only(
    helper: GLMHelper,
    ref_seq: str,
    alt_seq: str,
    motif_pos: List[int],
) -> Dict[str, Any]:
    if len(ref_seq) != len(alt_seq):
        raise ValueError("ref_seq and alt_seq must have the same length.")

    if not motif_pos:
        return {
            "influence_score": float("nan"),
            "query_positions": [],
            "targets_n": 0,
        }

    query_positions = []
    for index, (base_ref, base_alt) in enumerate(zip(ref_seq, alt_seq)):
        if base_ref != base_alt:
            query_positions.append(index)

    if not query_positions:
        return {
            "influence_score": 0.0,
            "query_positions": [],
            "targets_n": len(motif_pos),
        }

    total = 0.0
    for query_index in query_positions:
        scores: List[float] = []
        for motif_index in motif_pos:
            if motif_index == query_index:
                continue

            p_ref = helper.masked_probs_full_vocab(ref_seq, motif_index)
            p_alt = helper.masked_probs_full_vocab(alt_seq, motif_index)
            score_tensor = helper.shift_score_max_abs_logodds(p_ref, p_alt)
            score_value = float(score_tensor.item())
            scores.append(score_value)

        if scores:
            mean_score = float(np.mean(scores))
        else:
            mean_score = 0.0
        total += mean_score

    return {
        "influence_score": float(total),
        "query_positions": query_positions,
        "targets_n": len(motif_pos),
        "metric": "max_abs_logodds",
        "reduce": "mean_over_targets_then_sum_over_queries",
    }


def main() -> None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    ensure_directory(OUT_DIR)
    out_eval_dir = os.path.join(OUT_DIR, "evaluation")
    ensure_directory(out_eval_dir)

    print("Loading GLMModel...")
    glm = GLMModel(
        model_path=MODEL_DIR,
        fasta_file=TRAIN_FASTA_FOR_MODEL,
        force_retrain=False,
    )
    if glm.model is None:
        raise RuntimeError("Model not loaded.")
    helper = GLMHelper(glm)

    print("Loading REF and ALT FASTA files...")
    ref_headers, ref_seqs = load_fasta(EVAL_REF_FASTA)
    alt_headers, alt_seqs = load_fasta(EVAL_ALT_FASTA)

    print("Building pair pools...")
    pools = build_pair_pools(ref_headers, ref_seqs, alt_headers, alt_seqs)

    print("Pool sizes per bucket:")
    for bucket_name in ["A_only", "B_only", "both_flanks", "both_between"]:
        size = len(pools[bucket_name])
        print(f"  {bucket_name}: {size}")

    selection_seed = GLOBAL_SEED + 999
    chosen_pairs, selection_source = choose_pairs_auto(
        pools=pools,
        seed=selection_seed,
        existing_eval_csv=EXISTING_EVAL_CSV_FOR_SELECTION,
        manifest_csv=SELECTION_MANIFEST_CSV,
    )
    print(f"[OK] Selection source: {selection_source}")
    print(f"[OK] Selected {len(chosen_pairs)} pairs")

    eval_rows: List[Dict[str, Any]] = []

    for index, pair_record in enumerate(chosen_pairs, start=1):
        ref_seq = pair_record.ref_seq
        alt_seq = pair_record.alt_seq
        meta = pair_record.meta_ref

        if ref_seq.count("-") != 0:
            raise RuntimeError(f"REF contains deletions for seq_id={pair_record.seq_id}")
        if alt_seq.count("-") <= 0:
            raise RuntimeError(f"ALT has no deletions for seq_id={pair_record.seq_id}")
        if len(ref_seq) != len(alt_seq):
            raise RuntimeError(f"Length mismatch for seq_id={pair_record.seq_id}")

        motif_pos = motif_positions(meta, seq_len=len(ref_seq))

        ce_ref = mlm_pseudo_cross_entropy_on_positions(
            helper, ref_seq, positions=motif_pos, stride=CE_STRIDE
        )
        ce_alt = mlm_pseudo_cross_entropy_on_positions(
            helper, alt_seq, positions=motif_pos, stride=CE_STRIDE
        )

        fit_ref = helper.pll_mean_loglik_on_positions(
            ref_seq, positions=motif_pos, stride=PLL_STRIDE
        )
        fit_alt = helper.pll_mean_loglik_on_positions(
            alt_seq, positions=motif_pos, stride=PLL_STRIDE
        )

        delta_motif = delta_likelihood_motif_only(
            helper, ref_seq, alt_seq, positions=motif_pos
        )

        influence_result: Dict[str, Any] = {
            "influence_score": float("nan"),
            "query_positions": [],
            "targets_n": len(motif_pos),
            "metric": "max_abs_logodds",
        }
        if COMPUTE_INFLUENCE:
            influence_result = influence_to_motif_only(
                helper, ref_seq, alt_seq, motif_pos
            )

        row = {
            "id": pair_record.seq_id,
            "label": pair_record.label,
            "bucket": pair_record.bucket,
            "alt_both_mode": pair_record.alt_both_mode,
            "posA": meta.posA,
            "posB": meta.posB,
            "gap": meta.gap,
            "deletions_ref": ref_seq.count("-"),
            "deletions_alt": alt_seq.count("-"),
            "motif_positions_n": len(motif_pos),
            "ce_ref_motif": ce_ref["cross_entropy"],
            "ce_alt_motif": ce_alt["cross_entropy"],
            "fitness_pll_mean_logp_ref_motif": fit_ref["fitness"],
            "fitness_pll_mean_logp_alt_motif": fit_alt["fitness"],
            "ref_sum_motif": delta_motif["reference_sum"],
            "alt_sum_motif": delta_motif["perturbed_sum"],
            "delta_loglik_motif": delta_motif["delta"],
            "influence_score_motif": influence_result["influence_score"],
            "influence_metric": influence_result.get("metric", "max_abs_logodds"),
            "n_query_positions": len(influence_result.get("query_positions", [])),
            "ref_sequence_used": ref_seq,
            "alt_sequence_used": alt_seq,
        }
        eval_rows.append(row)

        if index % 25 == 0 or index == len(chosen_pairs):
            print(f"  [{index}/{len(chosen_pairs)}] done")

    base_fields = [key for key in eval_rows[0].keys()
                   if key not in ("ref_sequence_used", "alt_sequence_used")]
    fieldnames = base_fields + ["ref_sequence_used", "alt_sequence_used"]

    eval_csv_path = os.path.join(
        out_eval_dir,
        f"eval_motif_only__PAIRED__ALT_DUPBOTH__A{N_A_ONLY}_B{N_B_ONLY}"
        f"_bothFlanks{N_BOTH_FLANKS}_bothBetween{N_BOTH_BETWEEN}.csv",
    )

    with open(eval_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    print(f"[OK] Saved: {eval_csv_path}")
    print("DONE")
if __name__ == "__main__":
    main()

