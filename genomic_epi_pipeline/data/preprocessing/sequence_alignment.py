"""
Sequence Alignment Preprocessing.

Aligns a collection of viral sequences to a reference genome so that
positional features are comparable across samples.

Alignment backends
------------------
* **Pairwise** (default, no external dependency): Smith-Waterman via
  ``Bio.pairwise2`` for small datasets.
* **MAFFT** (recommended for >1 000 sequences): calls the ``mafft``
  command-line tool if it is available on ``$PATH``.

After alignment, the module trims leading/trailing gap columns and
optionally masks low-coverage positions.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import pairwise2, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

GAP_CHAR = "-"
AMBIGUOUS_CHARS = set("NRYWSKMBDHV")


class PairwiseAligner:
    """
    Align each query sequence to a single reference using Smith-Waterman.

    Parameters
    ----------
    reference_sequence : str
        The reference genome sequence (IUPAC nucleotides, upper-case).
    match_score : float
    mismatch_penalty : float
    gap_open : float
    gap_extend : float
    """

    def __init__(
        self,
        reference_sequence: str,
        match_score: float = 2.0,
        mismatch_penalty: float = -1.0,
        gap_open: float = -2.0,
        gap_extend: float = -0.5,
    ):
        self.reference = reference_sequence.upper()
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def align(self, sequences: List[str]) -> Tuple[List[str], List[float]]:
        """
        Align a list of sequences to the reference.

        Returns
        -------
        aligned_sequences : list of str
            Each sequence padded/trimmed to the reference length.
        identity_scores : list of float
            Fraction of positions identical to the reference (0–1).
        """
        aligned: List[str] = []
        scores: List[float] = []

        for seq in sequences:
            aln_seq, score = self._align_one(seq.upper())
            aligned.append(aln_seq)
            scores.append(score)

        logger.info(
            "Pairwise alignment complete: %d sequences, mean identity %.3f",
            len(sequences),
            float(np.mean(scores)),
        )
        return aligned, scores

    # ------------------------------------------------------------------
    def _align_one(self, query: str) -> Tuple[str, float]:
        alignments = pairwise2.align.globalms(
            self.reference,
            query,
            self.match_score,
            self.mismatch_penalty,
            self.gap_open,
            self.gap_extend,
            one_alignment_only=True,
        )
        if not alignments:
            # Fallback: pad/truncate to reference length
            padded = (query + GAP_CHAR * len(self.reference))[: len(self.reference)]
            return padded, 0.0

        aln = alignments[0]
        # Extract the aligned query (second sequence in the alignment)
        aligned_query = aln.seqB
        # Trim to reference length
        aligned_query = aligned_query[: len(self.reference)]
        # Pad if shorter
        aligned_query = aligned_query.ljust(len(self.reference), GAP_CHAR)

        # Calculate identity
        matches = sum(
            r == q and r != GAP_CHAR
            for r, q in zip(self.reference, aligned_query)
        )
        identity = matches / len(self.reference)
        return aligned_query, identity


class MAFFTAligner:
    """
    Multiple sequence alignment using MAFFT (must be installed on PATH).

    Parameters
    ----------
    reference_path : str, optional
        Path to a reference FASTA.  If provided, ``--addfragments`` mode
        is used to add query sequences to the reference alignment.
    threads : int
        Number of CPU threads for MAFFT.
    """

    def __init__(self, reference_path: Optional[str] = None, threads: int = 4):
        if shutil.which("mafft") is None:
            raise EnvironmentError(
                "MAFFT is not installed or not on PATH. "
                "Install with: conda install -c bioconda mafft"
            )
        self.reference_path = reference_path
        self.threads = threads

    def align(self, sequences: List[str], seq_ids: Optional[List[str]] = None) -> List[str]:
        """
        Align sequences using MAFFT.

        Parameters
        ----------
        sequences : list of str
        seq_ids : list of str, optional
            Identifiers for the FASTA headers.

        Returns
        -------
        list of str
            Aligned sequences in the same order as input.
        """
        if seq_ids is None:
            seq_ids = [f"seq_{i}" for i in range(len(sequences))]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_fasta = Path(tmpdir) / "input.fasta"
            output_fasta = Path(tmpdir) / "aligned.fasta"

            # Write input FASTA
            with open(input_fasta, "w") as fh:
                for sid, seq in zip(seq_ids, sequences):
                    fh.write(f">{sid}\n{seq}\n")

            # Build MAFFT command
            cmd = ["mafft", "--auto", f"--thread", str(self.threads)]
            if self.reference_path:
                cmd += ["--addfragments", str(input_fasta), self.reference_path]
            else:
                cmd.append(str(input_fasta))

            with open(output_fasta, "w") as out_fh:
                subprocess.run(cmd, stdout=out_fh, stderr=subprocess.DEVNULL, check=True)

            # Parse output
            aligned_map: Dict[str, str] = {}
            for record in SeqIO.parse(str(output_fasta), "fasta"):
                aligned_map[record.id] = str(record.seq).upper()

        # Return in original order
        return [aligned_map.get(sid, sequences[i]) for i, sid in enumerate(seq_ids)]


class AlignmentPostprocessor:
    """
    Post-process a multiple sequence alignment (MSA).

    Operations
    ----------
    * Remove all-gap columns.
    * Trim leading/trailing gap-only columns.
    * Mask positions with >``max_gap_fraction`` gaps as ``N``.
    * Filter out sequences with >``max_seq_gap_fraction`` gaps.
    """

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        max_seq_gap_fraction: float = 0.1,
        mask_ambiguous: bool = True,
    ):
        self.max_gap_fraction = max_gap_fraction
        self.max_seq_gap_fraction = max_seq_gap_fraction
        self.mask_ambiguous = mask_ambiguous

    def process(self, aligned_sequences: List[str]) -> Tuple[List[str], List[int]]:
        """
        Clean an MSA.

        Returns
        -------
        cleaned_sequences : list of str
        kept_indices : list of int
            Indices of sequences that passed the gap filter.
        """
        if not aligned_sequences:
            return [], []

        aln_len = max(len(s) for s in aligned_sequences)
        # Pad to uniform length
        padded = [s.ljust(aln_len, GAP_CHAR) for s in aligned_sequences]

        # Convert to numpy char array for vectorised operations
        arr = np.array([list(s) for s in padded])

        # --- Column filter: remove columns with too many gaps ---
        gap_mask = arr == GAP_CHAR
        col_gap_frac = gap_mask.mean(axis=0)
        keep_cols = col_gap_frac <= self.max_gap_fraction
        arr = arr[:, keep_cols]

        # --- Row filter: remove sequences with too many gaps ---
        row_gap_frac = (arr == GAP_CHAR).mean(axis=1)
        keep_rows = np.where(row_gap_frac <= self.max_seq_gap_fraction)[0]

        if len(keep_rows) == 0:
            logger.warning("All sequences were filtered out by gap threshold.")
            return [], []

        arr = arr[keep_rows, :]

        # --- Mask ambiguous characters ---
        if self.mask_ambiguous:
            for ch in AMBIGUOUS_CHARS:
                arr[arr == ch] = "N"

        cleaned = ["".join(row) for row in arr]
        logger.info(
            "Alignment post-processing: %d/%d sequences kept, %d/%d columns kept.",
            len(keep_rows),
            len(aligned_sequences),
            int(keep_cols.sum()),
            aln_len,
        )
        return cleaned, keep_rows.tolist()
