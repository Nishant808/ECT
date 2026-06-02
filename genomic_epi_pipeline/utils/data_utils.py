"""
Data utility functions: temporal splits, batch iteration, checkpointing.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal train / val / test split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    date_col: str = "date",
    val_cutoff: Optional[str] = None,
    test_cutoff: Optional[str] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train / val / test by date (no leakage).

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    val_cutoff, test_cutoff : str, optional
        ISO date strings.  If provided, used directly.
        Otherwise, fractions of the sorted date range are used.
    val_frac, test_frac : float
        Fractions used when cutoffs are not provided.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    dates = df[date_col].sort_values()

    if test_cutoff is None:
        test_cutoff = dates.quantile(1 - test_frac)
    else:
        test_cutoff = pd.Timestamp(test_cutoff)

    if val_cutoff is None:
        val_cutoff = dates.quantile(1 - test_frac - val_frac)
    else:
        val_cutoff = pd.Timestamp(val_cutoff)

    train_df = df[df[date_col] < val_cutoff]
    val_df = df[(df[date_col] >= val_cutoff) & (df[date_col] < test_cutoff)]
    test_df = df[df[date_col] >= test_cutoff]

    logger.info(
        "Temporal split: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Sequence deduplication
# ---------------------------------------------------------------------------

def deduplicate_sequences(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate sequences, keeping the first (or last) occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=[sequence_col], keep=keep).reset_index(drop=True)
    logger.info("Deduplication: %d → %d sequences", before, len(df))
    return df


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def batch_iterator(
    data: Any,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterator:
    """
    Yield successive batches from a list, numpy array, or DataFrame.

    Parameters
    ----------
    data : list | np.ndarray | pd.DataFrame
    batch_size : int
    shuffle : bool
    seed : int

    Yields
    ------
    Batches of the same type as ``data``.
    """
    n = len(data)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start: start + batch_size]
        if isinstance(data, pd.DataFrame):
            yield data.iloc[batch_idx]
        elif isinstance(data, np.ndarray):
            yield data[batch_idx]
        else:
            yield [data[i] for i in batch_idx]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    obj: Any,
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Pickle an object to ``path`` and optionally write a JSON sidecar.

    Parameters
    ----------
    obj : Any
        Object to serialise.
    path : str
        Destination path (e.g. ``/mnt/results/checkpoint.pkl``).
    metadata : dict, optional
        Extra metadata written to ``<path>.meta.json``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Checkpoint saved: %s", p)

    if metadata is not None:
        meta_path = p.with_suffix(".meta.json")
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)


def load_checkpoint(path: str) -> Any:
    """Load a pickled checkpoint."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(p, "rb") as fh:
        obj = pickle.load(fh)
    logger.info("Checkpoint loaded: %s", p)
    return obj


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def save_results_csv(df: pd.DataFrame, filename: str, results_dir: str = "/mnt/results") -> str:
    """Save a DataFrame as CSV to the results directory."""
    out = Path(results_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved CSV: %s (%d rows)", out, len(df))
    return str(out)


def save_results_json(data: Dict, filename: str, results_dir: str = "/mnt/results") -> str:
    """Save a dict as JSON to the results directory."""
    out = Path(results_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved JSON: %s", out)
    return str(out)


# ---------------------------------------------------------------------------
# Sequence padding / truncation
# ---------------------------------------------------------------------------

def pad_sequences(
    sequences: List[str],
    target_length: Optional[int] = None,
    pad_char: str = "N",
) -> List[str]:
    """Pad (or truncate) all sequences to ``target_length``."""
    if target_length is None:
        target_length = max(len(s) for s in sequences)
    return [(s + pad_char * target_length)[:target_length] for s in sequences]


def encode_sequences_batch(sequences: List[str], length: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode a list of sequences into a (N, L, 4) float32 array.

    Delegates to ``bio_utils.one_hot_encode`` for each sequence.
    """
    from .bio_utils import one_hot_encode  # local import to avoid circular

    if length is None:
        length = max(len(s) for s in sequences)
    return np.stack([one_hot_encode(s, length=length) for s in sequences])
