import random
import math
from dataclasses import dataclass
from typing import List, Optional

DNA = ["A", "C", "G", "T"]

MOTIF_A_BASE = "ATATTCA"
MOTIF_B_BASE = "GTACTGC"


def rand_dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA) for _ in range(n))


def mutate_motif(motif: str, mutation_rate: float, rng: random.Random) -> str:
    """Randomly mutates nucleotides in the motif."""
    if mutation_rate <= 0:
        return motif
    s = list(motif)
    for i in range(len(s)):
        if rng.random() < mutation_rate:
            s[i] = rng.choice(DNA)
    return "".join(s)


@dataclass
class Sequence:
    seq: str
    label: str
    deletions: int
    pos_a: Optional[int]
    pos_b: Optional[int]
    gap: Optional[int]


def augmentation(seq: str, frac: float, rng: random.Random) -> str:
    """Adds deletions (-) to the sequence."""
    if frac <= 0:
        return seq
    n = len(seq)
    k = max(1, math.floor(frac * n))
    positions = list(range(n))
    rng.shuffle(positions)
    replace_pos = positions[:k]
    s = list(seq)
    for p in replace_pos:
        s[p] = '-'
    return "".join(s)


def make_example(
        length: int,
        mode: str,
        gaps: List[int],
        deletions: float,
        if_deletions: bool,
        rng: random.Random,
        motif_mutation_rate: float = 0.0
) -> Sequence:
    this_motif_a = mutate_motif(MOTIF_A_BASE, motif_mutation_rate, rng)
    this_motif_b = mutate_motif(MOTIF_B_BASE, motif_mutation_rate, rng)

    if length < max(len(this_motif_a), len(this_motif_b)) + 20:
        raise ValueError("Sequence length too short.")

    if mode == "no_motif":
        seq_clean = rand_dna(length, rng)
        pos_a, pos_b, gap = None, None, None

    else:
        if mode == "both":
            gap = rng.choice([g for g in gaps if g % 10 == 0])
            total_len = len(this_motif_a) + gap + len(this_motif_b)
            if total_len > length: total_len = length  # Fallback or error

            start = rng.randint(0, length - total_len)

            prefix = rand_dna(start, rng)
            between = rand_dna(gap, rng)
            suffix_len = length - start - total_len
            suffix = rand_dna(max(0, suffix_len), rng)

            seq_clean = prefix + this_motif_a + between + this_motif_b + suffix
            pos_a = start
            pos_b = start + len(this_motif_a) + gap

        elif mode == "A_only":
            start = rng.randint(0, length - len(this_motif_a))
            prefix = rand_dna(start, rng)
            suffix = rand_dna(length - start - len(this_motif_a), rng)
            seq_clean = prefix + this_motif_a + suffix
            pos_a = start
            pos_b, gap = None, None

        elif mode == "B_only":
            start = rng.randint(0, length - len(this_motif_b))
            prefix = rand_dna(start, rng)
            suffix = rand_dna(length - start - len(this_motif_b), rng)
            seq_clean = prefix + this_motif_b + suffix
            pos_b = start
            pos_a, gap = None, None

    seq_final_list = list(seq_clean)

    protected_indices = set()
    if pos_a is not None:
        for i in range(pos_a, pos_a + len(this_motif_a)): protected_indices.add(i)
    if pos_b is not None:
        for i in range(pos_b, pos_b + len(this_motif_b)): protected_indices.add(i)

    if if_deletions:
        num_dels = int(length * deletions)

        candidates = [i for i in range(length) if i not in protected_indices]

        if len(candidates) >= num_dels:
            rng.shuffle(candidates)
            del_indices = candidates[:num_dels]
            for idx in del_indices:
                seq_final_list[idx] = "-"

    seq_final = "".join(seq_final_list)
    actual_deletions = seq_final.count("-")

    return Sequence(
        seq=seq_final,
        label=mode,
        deletions=actual_deletions,
        pos_a=pos_a,
        pos_b=pos_b,
        gap=gap
    )


def generate_dataset(n: int, length: int = 150, seed: int = 42) -> List[Sequence]:
    rng = random.Random(seed)
    data = []
    probs = [0.4, 0.25, 0.25, 0.1]

    for _ in range(n):
        r = rng.random()
        if r < probs[0]:
            mode = "both"
        elif r < sum(probs[:2]):
            mode = "A_only"
        elif r < sum(probs[:3]):
            mode = "B_only"
        else:
            mode = "no_motif"

        del_rate = rng.uniform(0.05, 0.15)

        ex = make_example(
            length=length,
            mode=mode,
            gaps=list(range(10, 101, 10)),
            deletions=del_rate,
            if_deletions=True,
            rng=rng,
            motif_mutation_rate=0.0
        )
        data.append(ex)
    return data
