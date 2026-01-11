from typing import List, Tuple
import re

def normalize_seq(seq: str) -> str:
    # Convert RNA U->T, keep only letters
    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGTN]", "", seq)
    return seq

def ambiguous_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    return seq.count("N") / len(seq)

def filter_sequences(
    seqs: List[Tuple[str, str]],
    min_len: int,
    max_ambiguous_frac: float
) -> List[Tuple[str, str]]:
    out = []
    seen = set()
    for acc, s in seqs:
        s = normalize_seq(s)
        if len(s) < min_len:
            continue
        if ambiguous_fraction(s) > max_ambiguous_frac:
            continue
        # deduplicate by exact sequence
        if s in seen:
            continue
        seen.add(s)
        out.append((acc, s))
    return out
