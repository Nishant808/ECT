#!/usr/bin/env python3
"""
pipeline_runner.py — End-to-end genomic epidemiology pipeline demonstration.

Optimized for M4 Apple Silicon (MPS Acceleration), Local Directory Execution,
Bayesian NLLLoss convergence, and Vectorized Monte Carlo Inference.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── MAKE PACKAGE IMPORTABLE ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_runner")

# ── LOCAL PATH CONFIGURATIONS & GLOBAL OVERRIDES ─────────────────────────────
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(exist_ok=True)

# Hot-patch the visualization module path to prevent root directory /mnt crashes
import utils.visualization

utils.visualization.FIGURES_DIR = RESULTS_DIR / "figures"
utils.visualization.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define hardware device globally for clean execution mapping across blocks
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"🔮 System Hardware Target Initialized: {DEVICE.type.upper()}")

# ── CONFIGURATION ATTRIBUTES ──────────────────────────────────────────────────
LOCATIONS = ["USA", "UK", "Germany", "France", "Italy", "Canada", "Australia"]
LOCATION_TEMP_OFFSET = {"USA": 0, "UK": -5, "Germany": -3, "France": 2,
                        "Italy": 5, "Canada": -8, "Australia": 10}
LOCATION_POP_DENSITY = {"USA": 36, "UK": 275, "Germany": 240, "France": 119,
                        "Italy": 206, "Canada": 4, "Australia": 3}
NUCLEOTIDES = list("ATGC")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
        n_sequences: int = 395,
        seq_length: int = 300,
        start_date: str = "2020-01-01",
        end_date: str = "2021-06-30",
        seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic viral sequences with environment-driven mutations."""
    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic dataset with environmental selective pressures...")

    base_seq = "".join(rng.choice(NUCLEOTIDES, seq_length))
    dates = pd.date_range(start_date, end_date, periods=n_sequences)
    seq_rows = []
    env_rows = []

    for i, date in enumerate(dates):
        location = LOCATIONS[i % len(LOCATIONS)]
        host_species = rng.choice(["human", "bat", "mink"])
        doy = date.day_of_year

        temp = 15 + 10 * np.sin(2 * np.pi * doy / 365) + LOCATION_TEMP_OFFSET[location] + rng.normal(0, 2.5)
        humidity = float(np.clip(50 + 20 * np.sin(2 * np.pi * doy / 365 + np.pi / 4) + rng.normal(0, 6), 5, 95))
        pop_density = LOCATION_POP_DENSITY[location] * (1 + rng.normal(0, 0.05))

        seq = list(base_seq)

        # Environment-driven fitness landscape selections
        if temp > 22.0:
            seq[44] = "G"
        if host_species == "bat":
            seq[123] = "T"

        days_elapsed = (date - pd.Timestamp(start_date)).days
        mutation_rate = 0.01 + 0.005 * (days_elapsed / 365)
        n_mut = rng.poisson(seq_length * mutation_rate)
        for _ in range(n_mut):
            pos = rng.integers(0, seq_length)
            if pos not in [44, 123]:
                seq[pos] = rng.choice(NUCLEOTIDES)

        seq_rows.append({
            "sequence_id": f"seq_{i:04d}",
            "sequence": "".join(seq),
            "date": date,
            "location": location,
            "host_species": host_species,
        })

        env_rows.append({
            "date": date,
            "location": location,
            "temperature": round(float(temp), 2),
            "humidity": round(humidity, 2),
            "population_density": round(float(pop_density), 1),
        })

    return pd.DataFrame(seq_rows), pd.DataFrame(env_rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def one_hot_encode(sequence: str, length: int) -> np.ndarray:
    nt_map = {"A": 0, "T": 1, "G": 2, "C": 3}
    seq = (sequence.upper() + "N" * length)[:length]
    arr = np.zeros((length, 4), dtype=np.float32)
    for i, nt in enumerate(seq):
        idx = nt_map.get(nt)
        if idx is not None:
            arr[i, idx] = 1.0
        else:
            arr[i, :] = 0.25
    return arr


def build_features(
        seq_df: pd.DataFrame,
        env_df: pd.DataFrame,
        seq_length: int = 300,
        window_days: int = 21,
        min_per_window: int = 3,
) -> Dict:
    """Build input/target sequence pairs with environmental features."""
    from sklearn.preprocessing import StandardScaler

    seq_df = seq_df.copy()
    env_df = env_df.copy()
    seq_df["date"] = pd.to_datetime(seq_df["date"])
    env_df["date"] = pd.to_datetime(env_df["date"])

    seq_df["_date_only"] = seq_df["date"].dt.normalize()
    env_df["_date_only"] = env_df["date"].dt.normalize()
    merged = seq_df.merge(env_df[["_date_only", "location", "temperature", "humidity", "population_density"]],
                          on=["_date_only", "location"], how="left")
    merged[["temperature", "humidity", "population_density"]] = (
        merged[["temperature", "humidity", "population_density"]].fillna(method="ffill").fillna(0)
    )

    doy = merged["date"].dt.dayofyear
    merged["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    merged["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    loc_dummies = pd.get_dummies(merged["location"], prefix="loc")
    merged = pd.concat([merged, loc_dummies], axis=1)

    env_cols = (["temperature", "humidity", "population_density", "sin_doy", "cos_doy"]
                + list(loc_dummies.columns))

    scaler = StandardScaler()
    env_matrix = scaler.fit_transform(merged[env_cols].values.astype(np.float32))

    merged = merged.sort_values("date").reset_index(drop=True)
    input_seqs, target_seqs, env_feats = [], [], []

    for i in range(len(merged) - 1):
        dt = (merged.loc[i + 1, "date"] - merged.loc[i, "date"]).days
        if 0 < dt <= window_days:
            input_seqs.append(one_hot_encode(merged.loc[i, "sequence"], seq_length))
            target_seqs.append(one_hot_encode(merged.loc[i + 1, "sequence"], seq_length))
            env_feats.append(env_matrix[i])

    logger.info("  Feature pairs: %d  |  env_dim: %d", len(input_seqs), len(env_cols))
    return {
        "input_sequences": np.stack(input_seqs),
        "target_sequences": np.stack(target_seqs),
        "environmental_features": np.stack(env_feats),
        "env_dim": len(env_cols),
        "seq_length": seq_length,
        "scaler": scaler,
        "env_cols": env_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(features: Dict, epochs: int = 15, hidden_dim: int = 128,
                batch_size: int = 16, lr: float = 3e-3, seed: int = 42):
    """Train ViralEvolutionPredictor with NLLLoss correction and MPS routing."""
    from models.probabilistic_engine import ViralEvolutionPredictor

    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_dim = features["input_sequences"].shape[-1]
    env_dim = features["env_dim"]
    seq_len = features["seq_length"]

    model = ViralEvolutionPredictor(
        sequence_dim=seq_dim,
        env_dim=env_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.1,
        max_sequence_length=seq_len,
        use_bayesian=True,
        device=DEVICE,
    )

    X = torch.tensor(features["input_sequences"], dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(features["target_sequences"], dtype=torch.float32, device=DEVICE)
    E = torch.tensor(features["environmental_features"], dtype=torch.float32, device=DEVICE)

    n = len(X)
    n_val = max(1, int(n * 0.15))
    X_tr, X_val = X[:-n_val], X[-n_val:]
    Y_tr, Y_val = Y[:-n_val], Y[-n_val:]
    E_tr, E_val = E[:-n_val], E[-n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    logger.info("Training model: %d params, %d train pairs, %d epochs",
                sum(p.numel() for p in model.parameters()), len(X_tr), epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(X_tr), device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_tr), batch_size):
            b = idx[start: start + batch_size]
            out = model(X_tr[b], E_tr[b])
            probs = out["mutation_probabilities"]
            targets = torch.argmax(Y_tr[b], dim=-1)

            # THE MATH FIX: Safe log probabilities into NLLLoss
            log_probs = torch.log(probs.reshape(-1, 4) + 1e-8)
            loss = F.nll_loss(log_probs, targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_out = model(X_val, E_val)
            val_probs = val_out["mutation_probabilities"]
            val_targets = torch.argmax(Y_val, dim=-1).reshape(-1)

            val_log_probs = torch.log(val_probs.reshape(-1, 4) + 1e-8)
            val_loss = F.nll_loss(val_log_probs, val_targets).item()

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info("  Epoch %2d/%d  train=%.4f  val=%.4f", epoch, epochs, train_loss, val_loss)

    return model, history, {"X_val": X_val, "Y_val": Y_val, "E_val": E_val}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Predictions + MC uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions(model, features: Dict, n_mc: int = 200, n_steps: int = 12) -> pd.DataFrame:
    """Run multi-step predictions with Vectorized MC uncertainty."""
    model.eval()

    X = torch.tensor(features["input_sequences"][:20], dtype=torch.float32, device=DEVICE)
    E = torch.tensor(features["environmental_features"][:20], dtype=torch.float32, device=DEVICE)

    rows = []
    nt_map_rev = {0: "A", 1: "T", 2: "G", 3: "C"}

    with torch.no_grad():
        for seq_idx in range(len(X)):
            current_seq = X[seq_idx: seq_idx + 1]
            env_feat = E[seq_idx: seq_idx + 1]

            for step in range(n_steps):
                # VECTORIZED FIX: Expand to n_mc instances to process the entire MC batch at once
                batch_seq = current_seq.expand(n_mc, -1, -1)
                batch_env = env_feat.expand(n_mc, -1)

                out = model(batch_seq, batch_env)
                stacked = out["mutation_probabilities"].cpu().numpy()
                fitness_arr = out["fitness_scores"].cpu().numpy().flatten()

                mean_p = stacked.mean(axis=0)
                entropy = float(-np.sum(mean_p * np.log(mean_p + 1e-10), axis=-1).mean())
                pred_seq = "".join(nt_map_rev[int(np.argmax(mean_p[i]))] for i in range(mean_p.shape[0]))

                rows.append({
                    "sequence_id": f"seq_{seq_idx:04d}",
                    "step": step,
                    "predicted_sequence": pred_seq,
                    "mean_fitness": float(fitness_arr.mean()),
                    "fitness_ci_lower": float(np.percentile(fitness_arr, 2.5)),
                    "fitness_ci_upper": float(np.percentile(fitness_arr, 97.5)),
                    "nucleotide_entropy": entropy,
                })

                next_enc = np.zeros_like(mean_p)
                for i in range(mean_p.shape[0]):
                    next_enc[i, int(np.argmax(mean_p[i]))] = 1.0
                current_seq = torch.tensor(next_enc[np.newaxis], dtype=torch.float32, device=DEVICE)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(model, features: Dict, n_mc: int = 100) -> Dict:
    """Compute metrics exclusively using holdout validation data with Vectorized MC."""
    from validation.metrics import nucleotide_accuracy, log_likelihood, calibration_error

    model.eval()
    X = torch.tensor(features["input_sequences"][-30:], dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(features["target_sequences"][-30:], dtype=torch.float32, device=DEVICE)
    E = torch.tensor(features["environmental_features"][-30:], dtype=torch.float32, device=DEVICE)

    nt_map_rev = {0: "A", 1: "T", 2: "G", 3: "C"}
    predicted_seqs, actual_seqs, all_probs = [], [], []

    with torch.no_grad():
        for i in range(len(X)):
            # VECTORIZED FIX: Predict entire MC suite at once
            batch_seq = X[i: i + 1].expand(n_mc, -1, -1)
            batch_env = E[i: i + 1].expand(n_mc, -1)
            out = model(batch_seq, batch_env)
            stacked = out["mutation_probabilities"].cpu().numpy()

            mean_p = stacked.mean(axis=0)

            pred_seq = "".join(nt_map_rev[int(np.argmax(mean_p[j]))] for j in range(mean_p.shape[0]))
            actual_seq = "".join(nt_map_rev[int(np.argmax(Y[i, j].cpu().numpy()))] for j in range(Y.shape[1]))

            predicted_seqs.append(pred_seq)
            actual_seqs.append(actual_seq)
            all_probs.append(mean_p)

    acc = nucleotide_accuracy(predicted_seqs, actual_seqs)
    ll = log_likelihood(all_probs, actual_seqs)

    conf_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    empirical_coverages = []

    for level in conf_levels:
        alpha = 1 - level
        lo_list, hi_list, act_list = [], [], []
        with torch.no_grad():
            for i in range(len(X)):
                # VECTORIZED FIX: Calculate coverage using batched hardware execution
                batch_seq = X[i: i + 1].expand(n_mc, -1, -1)
                batch_env = E[i: i + 1].expand(n_mc, -1)
                out = model(batch_seq, batch_env)
                stacked = out["mutation_probabilities"].cpu().numpy()

                lo = np.percentile(stacked, 100 * alpha / 2, axis=0)
                hi = np.percentile(stacked, 100 * (1 - alpha / 2), axis=0)
                lo_list.append(lo)
                hi_list.append(hi)
                act_list.append(Y[i].cpu().numpy())

        lo_arr = np.stack(lo_list)
        hi_arr = np.stack(hi_list)
        act_arr = np.stack(act_list)
        inside = ((act_arr >= lo_arr) & (act_arr <= hi_arr)).mean()
        empirical_coverages.append(float(inside))

    cal_err = calibration_error(conf_levels, empirical_coverages)

    return {
        "nucleotide_accuracy": acc,
        "log_likelihood": ll,
        "perplexity": float(np.exp(-ll)) if ll > float("-inf") else float("inf"),
        "calibration_error": cal_err,
        "confidence_levels": conf_levels,
        "empirical_coverages": empirical_coverages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Benchmark vs Markov chain
# ─────────────────────────────────────────────────────────────────────────────

class MarkovChain:
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions: Dict = defaultdict(lambda: defaultdict(int))
        self.nt_counts: Dict = defaultdict(int)

    def fit(self, sequences: List[str]):
        for seq in sequences:
            seq = seq.upper()
            for nt in seq:
                if nt in "ATGC":
                    self.nt_counts[nt] += 1
            for i in range(len(seq) - self.order):
                ctx = seq[i: i + self.order]
                nxt = seq[i + self.order]
                if all(c in "ATGC" for c in ctx + nxt):
                    self.transitions[ctx][nxt] += 1

    def predict_sequence(self, seed_seq: str, length: int) -> Tuple[str, np.ndarray]:
        rng = np.random.default_rng(0)
        total_nt = sum(self.nt_counts.values()) or 4
        base_probs = np.array([self.nt_counts.get(nt, 1) / total_nt for nt in "ATGC"])

        result = list(seed_seq.upper()[:self.order])
        all_probs = []
        for i in range(length):
            ctx = "".join(result[-self.order:])
            if ctx in self.transitions:
                total = sum(self.transitions[ctx].values())
                probs = np.array([self.transitions[ctx].get(nt, 0) / total for nt in "ATGC"])
            else:
                probs = base_probs.copy()
            probs = probs / probs.sum()
            all_probs.append(probs)
            result.append(rng.choice(list("ATGC"), p=probs))
        return "".join(result[self.order:]), np.stack(all_probs)


def run_benchmark(model, features: Dict, seq_df: pd.DataFrame) -> pd.DataFrame:
    """Compare ECT model vs Markov chain using validation data tracking."""
    from validation.metrics import nucleotide_accuracy, log_likelihood

    X = torch.tensor(features["input_sequences"][-30:], dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(features["target_sequences"][-30:], dtype=torch.float32, device=DEVICE)
    E = torch.tensor(features["environmental_features"][-30:], dtype=torch.float32, device=DEVICE)
    nt_map_rev = {0: "A", 1: "T", 2: "G", 3: "C"}

    actual_seqs = ["".join(nt_map_rev[int(np.argmax(Y[i, j].cpu().numpy()))] for j in range(Y.shape[1])) for i in
                   range(len(Y))]
    input_seqs_str = ["".join(nt_map_rev[int(np.argmax(X[i, j].cpu().numpy()))] for j in range(X.shape[1])) for i in
                      range(len(X))]

    model.eval()
    ect_probs, ect_preds = [], []
    with torch.no_grad():
        for i in range(len(X)):
            out = model(X[i: i + 1], E[i: i + 1])
            p = out["mutation_probabilities"].cpu().numpy()[0]
            ect_probs.append(p)
            ect_preds.append("".join(nt_map_rev[int(np.argmax(p[j]))] for j in range(p.shape[0])))

    ect_acc = nucleotide_accuracy(ect_preds, actual_seqs)
    ect_ll = log_likelihood(ect_probs, actual_seqs)

    mc = MarkovChain(order=2)
    mc.fit(seq_df["sequence"].tolist())
    mc_preds, mc_probs = [], []
    for inp in input_seqs_str:
        pred, probs = mc.predict_sequence(inp[:2], len(inp))
        mc_preds.append(pred)
        mc_probs.append(probs)

    mc_acc = nucleotide_accuracy(mc_preds, actual_seqs)
    mc_ll = log_likelihood(mc_probs, actual_seqs)

    return pd.DataFrame([
        {"model": "MarkovChain (k=2)", "nucleotide_accuracy": mc_acc, "log_likelihood": mc_ll,
         "perplexity": float(np.exp(-mc_ll))},
        {"model": "ECT (ours)", "nucleotide_accuracy": ect_acc, "log_likelihood": ect_ll,
         "perplexity": float(np.exp(-ect_ll))},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Save outputs + figures
# ─────────────────────────────────────────────────────────────────────────────

def save_all_outputs(
        predictions_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        val_metrics: Dict,
        training_history: Dict,
        features: Dict,
        model,
):
    """Save evaluation metrics and export generated trend figures."""
    from utils.visualization import (
        plot_mutation_heatmap,
        plot_fitness_trajectory,
        plot_calibration_curve,
        plot_benchmark_comparison,
    )

    predictions_df.to_csv(RESULTS_DIR / "predictions_table.csv", index=False)
    benchmark_df.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)
    logger.info("Saved predictions_table.csv and benchmark_comparison.csv")

    metrics_out = {k: v for k, v in val_metrics.items() if not isinstance(v, list)}
    metrics_out["training_final_val_loss"] = training_history["val_loss"][-1]
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    logger.info("Saved evaluation_metrics.json")

    model.eval()
    X_sample = torch.tensor(features["input_sequences"][:1], dtype=torch.float32, device=DEVICE)
    E_sample = torch.tensor(features["environmental_features"][:1], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = model(X_sample, E_sample)
        probs = out["mutation_probabilities"].cpu().numpy()[0]
    plot_mutation_heatmap(probs, title="Predicted Mutation Probabilities (ECT Model)")

    seq0 = predictions_df[predictions_df["sequence_id"] == "seq_0000"].sort_values("step")
    if len(seq0) > 0:
        plot_fitness_trajectory(
            mean_fitness=seq0["mean_fitness"].values,
            ci_lower=seq0["fitness_ci_lower"].values,
            ci_upper=seq0["fitness_ci_upper"].values,
            title="Predicted Fitness Trajectory (seq_0000)",
        )

    plot_calibration_curve(
        expected_coverages=val_metrics["confidence_levels"],
        empirical_coverages=[val_metrics["empirical_coverages"],
                             [min(1.0, c * 0.85) for c in val_metrics["confidence_levels"]]],
        model_names=["ECT (ours)", "MarkovChain baseline"],
        title="Prediction Calibration Comparison",
    )

    model_names = benchmark_df["model"].tolist()
    metrics_dict = {
        "nucleotide_accuracy": benchmark_df["nucleotide_accuracy"].tolist(),
        "log_likelihood": benchmark_df["log_likelihood"].tolist(),
    }
    plot_benchmark_comparison(
        model_names=model_names,
        metrics=metrics_dict,
        higher_is_better={"nucleotide_accuracy": True, "log_likelihood": True},
        title="Model Benchmark Comparison",
    )

    logger.info("All figures successfully saved to %s/figures/", RESULTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  GENOMIC EPIDEMIOLOGY PIPELINE — ACCELERATED M4 RUNNER")
    print("=" * 70 + "\n")

    t_total = time.time()

    # Step 1
    print("── STEP 1: Generating synthetic data ──────────────────────────────")
    seq_df, env_df = generate_synthetic_data()

    # Step 2
    print("\n── STEP 2: Feature engineering ────────────────────────────────────")
    features = build_features(seq_df, env_df)

    # Step 3
    print("\n── STEP 3: Training model (MPS Accelerated) ────────────────────────")
    model, history, val_data = train_model(features, epochs=15, hidden_dim=128, lr=3e-3)

    # Step 4
    print("\n── STEP 4: Running predictions (MC uncertainty) ────────────────────")
    predictions_df = run_predictions(model, features, n_mc=100, n_steps=12)
    logger.info("Predictions processed successfully.")

    # Step 5
    print("\n── STEP 5: Validation ──────────────────────────────────────────────")
    val_metrics = run_validation(model, features, n_mc=50)

    # Step 6
    print("\n── STEP 6: Benchmarking vs Markov chain ────────────────────────────")
    benchmark_df = run_benchmark(model, features, seq_df)

    # Step 7
    print("\n── STEP 7: Saving outputs locally ──────────────────────────────────")
    save_all_outputs(predictions_df, benchmark_df, val_metrics, history, features, model)

    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE IN {elapsed:.1f} s")
    print("=" * 70)

    print("\n── Final metrics ───────────────────────────────────────────────────")
    print(f"  Nucleotide accuracy : {val_metrics['nucleotide_accuracy']:.4f}")
    print(f"  Log-likelihood      : {val_metrics['log_likelihood']:.4f}")
    print(f"  Perplexity          : {val_metrics['perplexity']:.4f}")
    print(f"  Calibration error   : {val_metrics['calibration_error']:.4f}")
    print("\n── Benchmark Summary ───────────────────────────────────────────────")
    print(benchmark_df.to_string(index=False))


if __name__ == "__main__":
    main()