"""
Biological utility functions — pure NumPy, no BioPython dependency.

Covers: GC content, one-hot encoding, Hamming/Jukes-Cantor distances,
consensus sequence, codon usage, and translation.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUCLEOTIDES = "ATGC"
COMPLEMENT = str.maketrans("ATGCatgc", "TACGtacg")

ONE_HOT_MAP: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}
ONE_HOT_REVERSE: Dict[int, str] = {v: k for k, v in ONE_HOT_MAP.items()}

STANDARD_CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# ---------------------------------------------------------------------------
# Sequence statistics
# ---------------------------------------------------------------------------

def gc_content(sequence: str) -> float:
    """Return the GC fraction of a nucleotide sequence (0–1)."""
    seq = sequence.upper()
    total = sum(seq.count(nt) for nt in NUCLEOTIDES)
    if total == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / total


def nucleotide_frequencies(sequence: str) -> Dict[str, float]:
    """Return per-nucleotide frequencies."""
    seq = sequence.upper()
    total = sum(seq.count(nt) for nt in NUCLEOTIDES)
    if total == 0:
        return {nt: 0.0 for nt in NUCLEOTIDES}
    return {nt: seq.count(nt) / total for nt in NUCLEOTIDES}


def shannon_entropy(sequence: str) -> float:
    """Shannon entropy of nucleotide composition (bits, max ≈ 2.0)."""
    freqs = nucleotide_frequencies(sequence)
    probs = np.array([v for v in freqs.values() if v > 0])
    return float(-np.sum(probs * np.log2(probs)))


def sequence_complexity(sequence: str, window: int = 10) -> float:
    """Mean per-window Shannon entropy as a proxy for sequence complexity."""
    seq = sequence.upper()
    if len(seq) < window:
        return shannon_entropy(seq)
    return float(np.mean([
        shannon_entropy(seq[i: i + window])
        for i in range(len(seq) - window + 1)
    ]))


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def hamming_distance(seq1: str, seq2: str, ignore_gaps: bool = True) -> int:
    """Hamming distance between two equal-length sequences."""
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Sequences must have equal length ({len(seq1)} vs {len(seq2)})."
        )
    skip = {"-", "N"} if ignore_gaps else set()
    return sum(
        a != b
        for a, b in zip(seq1.upper(), seq2.upper())
        if a not in skip and b not in skip
    )


def jukes_cantor_distance(seq1: str, seq2: str) -> float:
    """Jukes-Cantor corrected nucleotide distance. Returns np.inf at saturation."""
    n = min(len(seq1), len(seq2))
    if n == 0:
        return 0.0
    p = hamming_distance(seq1[:n], seq2[:n]) / n
    if p >= 0.75:
        return np.inf
    return -0.75 * np.log(1.0 - (4.0 / 3.0) * p)


def pairwise_distance_matrix(sequences: List[str]) -> np.ndarray:
    """Compute an N×N Jukes-Cantor distance matrix."""
    n = len(sequences)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jukes_cantor_distance(sequences[i], sequences[j])
            mat[i, j] = mat[j, i] = d
    return mat


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def one_hot_encode(sequence: str, length: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode a nucleotide sequence.

    Returns
    -------
    np.ndarray, shape (L, 4)  — columns: A, T, G, C
    """
    seq = sequence.upper()
    if length is not None:
        seq = (seq + "-" * length)[:length]
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nt in enumerate(seq):
        idx = ONE_HOT_MAP.get(nt)
        if idx is not None:
            arr[i, idx] = 1.0
        else:
            arr[i, :] = 0.25  # uniform for ambiguous / gap
    return arr


def one_hot_decode(arr: np.ndarray) -> str:
    """Decode a (L, 4) one-hot array back to a nucleotide string."""
    indices = np.argmax(arr, axis=-1)
    return "".join(ONE_HOT_REVERSE.get(int(i), "N") for i in indices)


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

def consensus_sequence(sequences: List[str], threshold: float = 0.5) -> str:
    """Majority-vote consensus sequence."""
    if not sequences:
        return ""
    length = len(sequences[0])
    consensus = []
    for i in range(length):
        col = [s[i].upper() for s in sequences if i < len(s)]
        counts = Counter(c for c in col if c in NUCLEOTIDES)
        total = sum(counts.values())
        if total == 0:
            consensus.append("N")
            continue
        best_nt, best_count = counts.most_common(1)[0]
        consensus.append(best_nt if best_count / total >= threshold else "N")
    return "".join(consensus)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(nucleotide_sequence: str,
              codon_table: Optional[Dict[str, str]] = None) -> str:
    """Translate a nucleotide sequence to amino acids (stop = '*')."""
    table = codon_table or STANDARD_CODON_TABLE
    seq = nucleotide_sequence.upper().replace("-", "")
    return "".join(table.get(seq[i: i + 3], "X") for i in range(0, len(seq) - 2, 3))


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.upper().translate(COMPLEMENT)[::-1]


# ---------------------------------------------------------------------------
# Codon usage
# ---------------------------------------------------------------------------

def codon_usage_bias(sequence: str) -> Dict[str, float]:
    """Relative synonymous codon usage (RSCU) for a coding sequence."""
    seq = sequence.upper().replace("-", "")
    codon_counts: Counter = Counter(
        seq[i: i + 3]
        for i in range(0, len(seq) - 2, 3)
        if all(c in NUCLEOTIDES for c in seq[i: i + 3])
    )
    aa_to_codons: Dict[str, List[str]] = {}
    for codon, aa in STANDARD_CODON_TABLE.items():
        if aa != "*":
            aa_to_codons.setdefault(aa, []).append(codon)

    rscu: Dict[str, float] = {}
    for aa, codons in aa_to_codons.items():
        total = sum(codon_counts.get(c, 0) for c in codons)
        n_syn = len(codons)
        for codon in codons:
            expected = total / n_syn if n_syn > 0 else 0
            rscu[codon] = codon_counts.get(codon, 0) / expected if expected > 0 else 0.0
    return rscu
