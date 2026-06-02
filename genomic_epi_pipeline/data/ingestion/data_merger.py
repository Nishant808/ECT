"""
Data Merger: joins viral sequence records with environmental data.

Matching strategy
-----------------
1. Parse collection dates from sequence metadata.
2. For each sequence, find the closest environmental record by date
   within a configurable tolerance window (default ±7 days).
3. Optionally aggregate environmental data over a look-back window
   (e.g. 14-day rolling mean) before merging.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column name constants (must match ingestion modules)
SEQ_DATE_COL = "date"
SEQ_LOC_COL = "location"
ENV_DATE_COL = "date"
ENV_LOC_COL = "location"


class DataMerger:
    """
    Merge sequence and environmental DataFrames.

    Parameters
    ----------
    date_tolerance_days : int
        Maximum number of days between a sequence collection date and an
        environmental record for the match to be accepted.
    lookback_days : int, optional
        If provided, environmental features are averaged over the
        ``lookback_days`` days preceding each sequence date.
    require_location_match : bool
        If ``True`` (default), only merge records with the same location.
        If ``False``, fall back to date-only matching when no location match
        is found.
    """

    def __init__(
        self,
        date_tolerance_days: int = 7,
        lookback_days: Optional[int] = None,
        require_location_match: bool = True,
    ):
        self.date_tolerance_days = date_tolerance_days
        self.lookback_days = lookback_days
        self.require_location_match = require_location_match

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        sequence_df: pd.DataFrame,
        environmental_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge sequence and environmental DataFrames.

        Parameters
        ----------
        sequence_df : pd.DataFrame
            Must contain at least ``date`` and ``location`` columns.
        environmental_df : pd.DataFrame
            Must contain at least ``date``, ``location``, and numeric
            environmental feature columns.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame.  Rows without an environmental match within
            the tolerance window are dropped with a warning.
        """
        seq_df = sequence_df.copy()
        env_df = environmental_df.copy()

        # Ensure datetime types
        seq_df[SEQ_DATE_COL] = pd.to_datetime(seq_df[SEQ_DATE_COL])
        env_df[ENV_DATE_COL] = pd.to_datetime(env_df[ENV_DATE_COL])

        # Optionally compute rolling averages over the look-back window
        if self.lookback_days is not None:
            env_df = self._rolling_aggregate(env_df)

        # Identify numeric environmental feature columns
        env_feature_cols = [
            c
            for c in env_df.columns
            if c not in (ENV_DATE_COL, ENV_LOC_COL)
            and pd.api.types.is_numeric_dtype(env_df[c])
        ]

        merged_rows = []
        n_unmatched = 0

        for _, seq_row in seq_df.iterrows():
            env_row = self._find_best_match(seq_row, env_df, env_feature_cols)
            if env_row is None:
                n_unmatched += 1
                continue
            merged_row = {**seq_row.to_dict(), **{c: env_row[c] for c in env_feature_cols}}
            merged_rows.append(merged_row)

        if n_unmatched > 0:
            logger.warning(
                "%d sequence records had no environmental match within ±%d days and were dropped.",
                n_unmatched,
                self.date_tolerance_days,
            )

        if not merged_rows:
            raise ValueError(
                "No sequence records could be matched to environmental data. "
                "Check date ranges and location names."
            )

        merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)
        logger.info(
            "Merged %d sequence records with environmental data (%d dropped).",
            len(merged_df),
            n_unmatched,
        )
        return merged_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_best_match(
        self,
        seq_row: pd.Series,
        env_df: pd.DataFrame,
        feature_cols: list,
    ) -> Optional[pd.Series]:
        """Return the closest environmental record for a single sequence row."""
        seq_date = seq_row[SEQ_DATE_COL]
        seq_loc = seq_row.get(SEQ_LOC_COL)

        # Filter by location first (if required)
        if self.require_location_match and seq_loc is not None:
            candidates = env_df[env_df[ENV_LOC_COL] == seq_loc]
            if candidates.empty and not self.require_location_match:
                candidates = env_df  # fall back to all locations
        else:
            candidates = env_df

        if candidates.empty:
            return None

        # Find closest date within tolerance
        date_diffs = (candidates[ENV_DATE_COL] - seq_date).abs()
        min_diff = date_diffs.min()

        if min_diff > pd.Timedelta(days=self.date_tolerance_days):
            return None

        best_idx = date_diffs.idxmin()
        return candidates.loc[best_idx]

    def _rolling_aggregate(self, env_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace point-in-time environmental values with rolling means.

        Computes a ``lookback_days``-day trailing mean per location.
        """
        numeric_cols = [
            c
            for c in env_df.columns
            if c not in (ENV_DATE_COL, ENV_LOC_COL)
            and pd.api.types.is_numeric_dtype(env_df[c])
        ]

        result_parts = []
        for loc, group in env_df.groupby(ENV_LOC_COL):
            group = group.sort_values(ENV_DATE_COL).copy()
            group[numeric_cols] = (
                group[numeric_cols]
                .rolling(window=self.lookback_days, min_periods=1)
                .mean()
            )
            result_parts.append(group)

        return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def merge_sequence_and_environment(
    sequence_df: pd.DataFrame,
    environmental_df: pd.DataFrame,
    date_tolerance_days: int = 7,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper around :class:`DataMerger`.

    Parameters
    ----------
    sequence_df : pd.DataFrame
    environmental_df : pd.DataFrame
    date_tolerance_days : int
    lookback_days : int, optional

    Returns
    -------
    pd.DataFrame
    """
    merger = DataMerger(
        date_tolerance_days=date_tolerance_days,
        lookback_days=lookback_days,
    )
    return merger.merge(sequence_df, environmental_df)
