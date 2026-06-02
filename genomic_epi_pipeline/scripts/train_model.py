#!/usr/bin/env python3
"""
train_model.py — CLI script to train the ViralEvolutionPredictor.

Usage
-----
python scripts/train_model.py \
    --sequences data/sequences.csv \
    --environment data/environment.csv \
    --output /mnt/results/model_checkpoint.pkl \
    --epochs 20 \
    --hidden-dim 256 \
    --batch-size 16

The script:
  1. Loads sequence + environmental CSVs.
  2. Runs FeaturePipeline.fit_transform.
  3. Trains ViralEvolutionPredictor.
  4. Saves the model checkpoint and training metrics.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.feature_pipeline import FeaturePipeline
from models.probabilistic_engine import ViralEvolutionPredictor
from utils.data_utils import temporal_split, batch_iterator, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_model")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ViralEvolutionPredictor,
    input_seqs: torch.Tensor,
    env_feats: torch.Tensor,
    target_seqs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    n = len(input_seqs)
    indices = np.random.permutation(n)

    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        x = input_seqs[idx].to(device)
        e = env_feats[idx].to(device)
        y = target_seqs[idx].to(device)

        optimizer.zero_grad()
        outputs = model(x, e, y)

        # Cross-entropy on nucleotide predictions
        probs = outputs["mutation_probabilities"]  # (B, L, 4)
        targets_idx = torch.argmax(y, dim=-1)      # (B, L)
        loss = F.cross_entropy(probs.reshape(-1, 4), targets_idx.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: ViralEvolutionPredictor,
    input_seqs: torch.Tensor,
    env_feats: torch.Tensor,
    target_seqs: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(input_seqs.to(device), env_feats.to(device))
        probs = outputs["mutation_probabilities"]
        targets_idx = torch.argmax(target_seqs.to(device), dim=-1)
        loss = F.cross_entropy(probs.reshape(-1, 4), targets_idx.reshape(-1))
    return loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ViralEvolutionPredictor")
    parser.add_argument("--sequences", required=True, help="Path to sequences CSV")
    parser.add_argument("--environment", required=True, help="Path to environment CSV")
    parser.add_argument("--output", default="/mnt/results/model_checkpoint.pkl",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ---- Load data ----
    logger.info("Loading data...")
    seq_df = pd.read_csv(args.sequences)
    env_df = pd.read_csv(args.environment)

    # ---- Feature engineering ----
    logger.info("Running feature pipeline...")
    pipeline = FeaturePipeline(
        sequence_config={"encoding_method": "one_hot", "window_size": 100, "overlap": 50},
        environmental_config={"continuous_method": "standard", "seasonal_features": True},
        temporal_config={"time_window_days": 30, "min_sequences_per_window": 3},
    )
    features = pipeline.fit_transform(seq_df, env_df)

    input_seqs = torch.tensor(features["input_sequences"], dtype=torch.float32)
    target_seqs = torch.tensor(features["target_sequences"], dtype=torch.float32)
    env_feats = torch.tensor(features["environmental_features"], dtype=torch.float32)

    # Flatten window dimension if present
    if input_seqs.ndim == 4:
        B, W, L, C = input_seqs.shape
        input_seqs = input_seqs[:, 0, :, :]
        target_seqs = target_seqs[:, 0, :, :]

    seq_dim = input_seqs.shape[-1]
    env_dim = env_feats.shape[-1]
    logger.info("Tensors: input=%s env=%s target=%s", input_seqs.shape, env_feats.shape, target_seqs.shape)

    # ---- Train / val split ----
    n = len(input_seqs)
    n_val = max(1, int(n * 0.15))
    train_x, val_x = input_seqs[:-n_val], input_seqs[-n_val:]
    train_e, val_e = env_feats[:-n_val], env_feats[-n_val:]
    train_y, val_y = target_seqs[:-n_val], target_seqs[-n_val:]

    # ---- Build model ----
    model = ViralEvolutionPredictor(
        sequence_dim=seq_dim,
        env_dim=env_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=0.1,
        use_bayesian=False,
        device=args.device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ---- Training loop ----
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_x, train_e, train_y,
                                     optimizer, args.batch_size, device)
        val_loss = evaluate(model, val_x, val_e, val_y, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                {"model_state": model.state_dict(), "config": vars(args), "epoch": epoch},
                args.output,
                metadata={"best_val_loss": best_val, "epoch": epoch},
            )

    elapsed = time.time() - t0
    logger.info("Training complete in %.1f s. Best val loss: %.4f", elapsed, best_val)

    # Save training history
    hist_path = Path(args.output).with_suffix(".history.json")
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("History saved: %s", hist_path)


if __name__ == "__main__":
    main()
