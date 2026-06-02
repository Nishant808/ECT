"""
Prediction Engine — high-level wrapper around ViralEvolutionPredictor
and MonteCarloSimulator.

Exposes a single ``predict()`` method that accepts sequences + an
environmental DataFrame and returns a tidy DataFrame of predictions
with confidence intervals.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    High-level prediction interface for viral evolution forecasting.

    Parameters
    ----------
    model : ViralEvolutionPredictor
        A trained (or partially trained) predictor.
    feature_pipeline : FeaturePipeline
        A fitted feature pipeline used to encode inputs.
    num_mc_samples : int
        Number of Monte Carlo samples for uncertainty estimation.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model,
        feature_pipeline,
        num_mc_samples: int = 200,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.num_mc_samples = num_mc_samples
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        sequences: List[str],
        env_df: pd.DataFrame,
        horizon_days: int = 90,
        sequence_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Predict viral evolution for a set of sequences over a time horizon.

        Parameters
        ----------
        sequences : list of str
            Input viral sequences.
        env_df : pd.DataFrame
            Environmental data with columns: date, location, temperature,
            humidity, population_density.
        horizon_days : int
            Number of days to forecast.
        sequence_ids : list of str, optional
            Identifiers for each input sequence.

        Returns
        -------
        pd.DataFrame
            Columns: sequence_id, step, predicted_sequence,
                     mean_fitness, fitness_ci_lower, fitness_ci_upper,
                     nucleotide_entropy (mean per-position uncertainty).
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        # Encode inputs
        encoded_seqs, env_features = self._encode_inputs(sequences, env_df)

        # Build environmental trajectory for the forecast horizon
        env_trajectory = self._build_env_trajectory(env_df, horizon_days)

        rows: List[Dict] = []

        for seq_idx, (seq_id, enc_seq) in enumerate(zip(sequence_ids, encoded_seqs)):
            seq_tensor = torch.tensor(enc_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            env_tensor = torch.tensor(env_features[seq_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
            env_traj_tensor = torch.tensor(env_trajectory, dtype=torch.float32).to(self.device)

            traj_results = self._run_trajectory(seq_tensor, env_tensor, env_traj_tensor)

            for step_idx, step_data in enumerate(traj_results):
                rows.append({
                    "sequence_id": seq_id,
                    "step": step_idx,
                    "horizon_days": (step_idx + 1) * (horizon_days // max(len(traj_results), 1)),
                    "predicted_sequence": step_data["sequence"],
                    "mean_fitness": step_data["mean_fitness"],
                    "fitness_ci_lower": step_data["fitness_ci_lower"],
                    "fitness_ci_upper": step_data["fitness_ci_upper"],
                    "nucleotide_entropy": step_data["nucleotide_entropy"],
                    "n_mutations_from_input": step_data["n_mutations"],
                })

        return pd.DataFrame(rows)

    def predict_single_step(
        self,
        sequences: List[str],
        env_df: pd.DataFrame,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        One-step-ahead prediction with uncertainty.

        Returns
        -------
        predicted_sequences : list of str
        mean_probs : np.ndarray, shape (N, L, 4)
        std_probs : np.ndarray, shape (N, L, 4)
        """
        encoded_seqs, env_features = self._encode_inputs(sequences, env_df)

        all_probs: List[np.ndarray] = []
        predicted_seqs: List[str] = []

        for enc_seq, env_feat in zip(encoded_seqs, env_features):
            seq_t = torch.tensor(enc_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            env_t = torch.tensor(env_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

            sample_probs = []
            with torch.no_grad():
                for _ in range(self.num_mc_samples):
                    out = self.model(seq_t, env_t)
                    sample_probs.append(out["mutation_probabilities"].cpu().numpy())

            stacked = np.stack(sample_probs, axis=0)  # (S, 1, L, 4)
            mean_p = stacked.mean(axis=0)[0]           # (L, 4)
            all_probs.append(mean_p)

            # Greedy decode
            pred_seq = "".join("ATGC"[int(np.argmax(mean_p[i]))] for i in range(mean_p.shape[0]))
            predicted_seqs.append(pred_seq)

        mean_probs = np.stack(all_probs)
        std_probs = np.stack([
            np.stack([p for p in all_probs], axis=0).std(axis=0)
        ])

        return predicted_seqs, mean_probs, std_probs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_inputs(
        self,
        sequences: List[str],
        env_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode sequences and extract environmental features."""
        from ..utils.bio_utils import one_hot_encode

        seq_len = max(len(s) for s in sequences)
        encoded_seqs = np.stack([one_hot_encode(s, length=seq_len) for s in sequences])

        # Extract numeric environmental features
        num_cols = [c for c in env_df.columns
                    if c not in ("date", "location")
                    and pd.api.types.is_numeric_dtype(env_df[c])]

        if not num_cols:
            # Fallback: zeros
            env_features = np.zeros((len(sequences), 3), dtype=np.float32)
        else:
            env_mean = env_df[num_cols].mean().values.astype(np.float32)
            env_features = np.tile(env_mean, (len(sequences), 1))

        return encoded_seqs, env_features

    def _build_env_trajectory(
        self,
        env_df: pd.DataFrame,
        horizon_days: int,
        n_steps: int = 12,
    ) -> np.ndarray:
        """Build an environmental feature trajectory for the forecast horizon."""
        num_cols = [c for c in env_df.columns
                    if c not in ("date", "location")
                    and pd.api.types.is_numeric_dtype(env_df[c])]

        if not num_cols:
            return np.zeros((n_steps, 3), dtype=np.float32)

        # Use the last available values and add mild seasonal drift
        last_vals = env_df[num_cols].iloc[-1].values.astype(np.float32)
        trajectory = np.tile(last_vals, (n_steps, 1))

        # Add small Gaussian noise to simulate environmental variation
        rng = np.random.default_rng(42)
        trajectory += rng.normal(0, 0.05 * np.abs(last_vals) + 1e-6, trajectory.shape)

        return trajectory

    def _run_trajectory(
        self,
        seq_tensor: torch.Tensor,
        env_tensor: torch.Tensor,
        env_trajectory: torch.Tensor,
        n_steps: int = 12,
    ) -> List[Dict]:
        """Run multi-step trajectory with Monte Carlo uncertainty."""
        results = []
        current_seq = seq_tensor.clone()

        with torch.no_grad():
            for step in range(n_steps):
                env_step = env_trajectory[step: step + 1] if step < len(env_trajectory) else env_tensor

                # MC samples
                sample_probs = []
                sample_fitness = []
                for _ in range(self.num_mc_samples):
                    out = self.model(current_seq, env_step)
                    sample_probs.append(out["mutation_probabilities"].cpu().numpy())
                    sample_fitness.append(out["fitness_scores"].cpu().numpy().flatten()[0])

                stacked = np.stack(sample_probs)[:, 0]  # (S, L, 4)
                mean_p = stacked.mean(axis=0)            # (L, 4)

                # Entropy as uncertainty proxy
                entropy = -np.sum(mean_p * np.log(mean_p + 1e-10), axis=-1).mean()

                # Greedy decode
                pred_seq = "".join("ATGC"[int(np.argmax(mean_p[i]))] for i in range(mean_p.shape[0]))

                # Count mutations vs input
                input_seq = "".join("ATGC"[int(np.argmax(current_seq[0, i].cpu().numpy()))]
                                    for i in range(current_seq.shape[1]))
                n_mut = sum(a != b for a, b in zip(input_seq, pred_seq))

                fitness_arr = np.array(sample_fitness)
                results.append({
                    "sequence": pred_seq,
                    "mean_fitness": float(fitness_arr.mean()),
                    "fitness_ci_lower": float(np.percentile(fitness_arr, 2.5)),
                    "fitness_ci_upper": float(np.percentile(fitness_arr, 97.5)),
                    "nucleotide_entropy": float(entropy),
                    "n_mutations": n_mut,
                })

                # Update current sequence (one-hot from greedy decode)
                next_enc = np.zeros_like(mean_p)
                for i in range(mean_p.shape[0]):
                    next_enc[i, int(np.argmax(mean_p[i]))] = 1.0
                current_seq = torch.tensor(next_enc, dtype=torch.float32).unsqueeze(0).to(self.device)

        return results
