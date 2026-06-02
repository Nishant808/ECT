"""
Sequence Masking.

Identifies and masks low-quality or problematic positions in viral
genome sequences before feature extraction.

Masking strategies
------------------
* **Ambiguity masking**: replace IUPAC ambiguity codes (N, R, Y, …) with
  a sentinel character (default ``"N"``).
* **Low-complexity masking**: mask homopolymer runs and low-complexity
  windows using a Shannon-entropy threshold.
* **Coverage masking**: mask positions that fall below a minimum read
  depth (requires a per-position depth array).
* **Known-artifact masking**: mask positions listed in a user-supplied
  BED-like exclusion file (e.g. primer binding sites).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MASK_CHAR = "N"
AMBIGUOUS_IUPAC: Set[str] = set("NRYWSKMBDHV")
VALID_NUCLEOTIDES: Set[str] = set("ATGC")


# ---------------------------------------------------------------------------
# Individual masking strategies
# ---------------------------------------------------------------------------

class AmbiguityMasker:
    """Replace IUPAC ambiguity codes with ``MASK_CHAR``."""

    def mask(self, sequence: str) -> str:
        return "".join(
            c if c in VALID_NUCLEOTIDES else MASK_CHAR for c in sequence.upper()
        )

    def mask_batch(self, sequences: List[str]) -> List[str]:
        return [self.mask(s) for s in sequences]


class LowComplexityMasker:
    """
    Mask low-complexity windows using Shannon entropy.

    Parameters
    ----------
    window_size : int
        Sliding window length (nucleotides).
    entropy_threshold : float
        Windows with Shannon entropy below this value are masked.
        Maximum entropy for 4 nucleotides is log2(4) ≈ 2.0.
    """

    def __init__(self, window_size: int = 10, entropy_threshold: float = 0.8):
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold

    def mask(self, sequence: str) -> str:
        seq = sequence.upper()
        seq_list = list(seq)
        n = len(seq)

        for i in range(n - self.window_size + 1):
            window = seq[i : i + self.window_size]
            if self._entropy(window) < self.entropy_threshold:
                for j in range(i, i + self.window_size):
                    seq_list[j] = MASK_CHAR

        return "".join(seq_list)

    def mask_batch(self, sequences: List[str]) -> List[str]:
        return [self.mask(s) for s in sequences]

    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(window: str) -> float:
        counts = np.array([window.count(nt) for nt in "ATGC"], dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log2(probs)))


class CoverageMasker:
    """
    Mask positions below a minimum read depth.

    Parameters
    ----------
    min_depth : int
        Positions with depth < ``min_depth`` are masked.
    """

    def __init__(self, min_depth: int = 10):
        self.min_depth = min_depth

    def mask(self, sequence: str, depth_array: np.ndarray) -> str:
        if len(depth_array) != len(sequence):
            raise ValueError(
                f"depth_array length ({len(depth_array)}) must match "
                f"sequence length ({len(sequence)})."
            )
        seq_list = list(sequence.upper())
        for i, depth in enumerate(depth_array):
            if depth < self.min_depth:
                seq_list[i] = MASK_CHAR
        return "".join(seq_list)


class ArtifactMasker:
    """
    Mask known artifact positions from a BED-like exclusion file.

    The file should have one interval per line: ``start\\tend`` (0-based,
    half-open), or a single column of 0-based positions.
    """

    def __init__(self, exclusion_file: str):
        self.excluded_positions: Set[int] = self._load(exclusion_file)

    def mask(self, sequence: str) -> str:
        seq_list = list(sequence.upper())
        for pos in self.excluded_positions:
            if pos < len(seq_list):
                seq_list[pos] = MASK_CHAR
        return "".join(seq_list)

    def mask_batch(self, sequences: List[str]) -> List[str]:
        return [self.mask(s) for s in sequences]

    # ------------------------------------------------------------------
    @staticmethod
    def _load(path: str) -> Set[int]:
        positions: Set[int] = set()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Exclusion file not found: {path}")
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    start, end = int(parts[0]), int(parts[1])
                    positions.update(range(start, end))
                else:
                    positions.add(int(parts[0]))
        return positions


# ---------------------------------------------------------------------------
# Composite masker
# ---------------------------------------------------------------------------

class CompositeMasker:
    """
    Apply multiple masking strategies in sequence.

    Parameters
    ----------
    mask_ambiguous : bool
    mask_low_complexity : bool
    low_complexity_window : int
    low_complexity_threshold : float
    exclusion_file : str, optional
    """

    def __init__(
        self,
        mask_ambiguous: bool = True,
        mask_low_complexity: bool = True,
        low_complexity_window: int = 10,
        low_complexity_threshold: float = 0.8,
        exclusion_file: Optional[str] = None,
    ):
        self.maskers = []

        if mask_ambiguous:
            self.maskers.append(AmbiguityMasker())

        if mask_low_complexity:
            self.maskers.append(
                LowComplexityMasker(
                    window_size=low_complexity_window,
                    entropy_threshold=low_complexity_threshold,
                )
            )

        if exclusion_file:
            self.maskers.append(ArtifactMasker(exclusion_file))

    def mask(self, sequence: str) -> str:
        for masker in self.maskers:
            sequence = masker.mask(sequence)
        return sequence

    def mask_batch(self, sequences: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """
        Mask a batch of sequences and return masking statistics.

        Returns
        -------
        masked_sequences : list of str
        stats : dict
            ``{"total_positions": int, "masked_positions": int, "mask_fraction": float}``
        """
        masked = [self.mask(s) for s in sequences]

        total_pos = sum(len(s) for s in sequences)
        masked_pos = sum(s.count(MASK_CHAR) for s in masked)
        stats = {
            "total_positions": total_pos,
            "masked_positions": masked_pos,
            "mask_fraction": masked_pos / total_pos if total_pos > 0 else 0.0,
        }

        logger.info(
            "Masking complete: %d/%d positions masked (%.1f%%)",
            masked_pos,
            total_pos,
            stats["mask_fraction"] * 100,
        )
        return masked, stats
