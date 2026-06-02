"""
Standalone evaluation metrics for viral evolution prediction.

All functions accept plain Python / NumPy inputs and return floats or dicts.
No model or torch dependency.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ONE_HOT_MAP = {"A": 0, "T": 1, "G": 2, "C": 3}


# ---------------------------------------------------------------------------
# Sequence-level accuracy
# ---------------------------------------------------------------------------

def nucleotide_accuracy(
    predicted: List[str],
    actual: List[str],
    ignore_gaps: bool = True,
) -> float:
    """
    Fraction of positions where predicted nucleotide matches actual.

    Parameters
    ----------
    predicted, actual : list of str
        Nucleotide sequences (equal length per pair).
    ignore_gaps : bool
        Skip positions where either sequence has ``-`` or ``N``.

    Returns
    -------
    float in [0, 1]
    """
    skip = {"-", "N"} if ignore_gaps else set()
    total = correct = 0
    for pred, act in zip(predicted, actual):
        for p, a in zip(pred.upper(), act.upper()):
            if p in skip or a in skip:
                continue
            total += 1
            if p == a:
                correct += 1
    return correct / total if total > 0 else 0.0


def sequence_mse(
    predicted: List[str],
    actual: List[str],
) -> float:
    """
    Mean squared normalised Hamming distance across sequence pairs.

    Each pair contributes ``(hamming / length)^2``.
    """
    if not predicted:
        return float("inf")
    total = 0.0
    for pred, act in zip(predicted, actual):
        n = min(len(pred), len(act))
        if n == 0:
            continue
        ham = sum(p != a for p, a in zip(pred.upper()[:n], act.upper()[:n]))
        total += (ham / n) ** 2
    return total / len(predicted)


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

def log_likelihood(
    predicted_probs: List[np.ndarray],
    actual_sequences: List[str],
    eps: float = 1e-10,
) -> float:
    """
    Mean per-position log-likelihood of actual nucleotides under predicted distributions.

    Parameters
    ----------
    predicted_probs : list of np.ndarray, each shape (L, 4)
        Predicted probability distributions (A, T, G, C).
    actual_sequences : list of str
    eps : float
        Floor to avoid log(0).

    Returns
    -------
    float  (negative = worse; 0 = perfect)
    """
    total_ll = 0.0
    total_pos = 0
    for probs, seq in zip(predicted_probs, actual_sequences):
        for i, nt in enumerate(seq.upper()):
            if i >= len(probs):
                break
            idx = ONE_HOT_MAP.get(nt)
            if idx is None:
                continue
            p = float(np.clip(probs[i, idx], eps, 1.0))
            total_ll += np.log(p)
            total_pos += 1
    return total_ll / total_pos if total_pos > 0 else float("-inf")


def perplexity(
    predicted_probs: List[np.ndarray],
    actual_sequences: List[str],
) -> float:
    """
    Perplexity = exp(-mean_log_likelihood).  Lower is better.
    """
    ll = log_likelihood(predicted_probs, actual_sequences)
    if ll == float("-inf"):
        return float("inf")
    return float(np.exp(-ll))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def coverage_at_level(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    actual_values: np.ndarray,
) -> float:
    """
    Fraction of actual values that fall within [lower, upper] bounds.

    Parameters
    ----------
    lower_bounds, upper_bounds, actual_values : np.ndarray
        All same shape.

    Returns
    -------
    float in [0, 1]
    """
    inside = (actual_values >= lower_bounds) & (actual_values <= upper_bounds)
    return float(np.mean(inside))


def calibration_error(
    expected_coverages: Sequence[float],
    empirical_coverages: Sequence[float],
) -> float:
    """
    Mean absolute calibration error (MACE).

    Parameters
    ----------
    expected_coverages : sequence of float
        Nominal confidence levels, e.g. [0.5, 0.6, …, 0.95].
    empirical_coverages : sequence of float
        Observed coverage at each level.

    Returns
    -------
    float  (0 = perfect calibration)
    """
    exp = np.array(expected_coverages)
    emp = np.array(empirical_coverages)
    return float(np.mean(np.abs(exp - emp)))


def compute_all_metrics(
    predicted_sequences: List[str],
    actual_sequences: List[str],
    predicted_probs: Optional[List[np.ndarray]] = None,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    actual_values: Optional[np.ndarray] = None,
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute all available metrics and return as a flat dict.

    Parameters
    ----------
    predicted_sequences, actual_sequences : list of str
    predicted_probs : list of (L, 4) arrays, optional
    lower_bounds, upper_bounds, actual_values : np.ndarray, optional
        For calibration coverage.
    confidence_levels : list of float, optional
        Nominal levels for calibration error.

    Returns
    -------
    dict  {metric_name: value}
    """
    results: Dict[str, float] = {}

    # Sequence accuracy
    results["nucleotide_accuracy"] = nucleotide_accuracy(predicted_sequences, actual_sequences)
    results["sequence_mse"] = sequence_mse(predicted_sequences, actual_sequences)

    # Probabilistic metrics
    if predicted_probs is not None:
        results["log_likelihood"] = log_likelihood(predicted_probs, actual_sequences)
        results["perplexity"] = perplexity(predicted_probs, actual_sequences)

    # Calibration
    if (lower_bounds is not None and upper_bounds is not None
            and actual_values is not None):
        results["coverage_95"] = coverage_at_level(lower_bounds, upper_bounds, actual_values)

    if confidence_levels is not None:
        # Placeholder: caller should supply empirical coverages separately
        pass

    logger.info("Metrics computed: %s", {k: f"{v:.4f}" for k, v in results.items()})
    return results
