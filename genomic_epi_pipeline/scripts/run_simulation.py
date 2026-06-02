#!/usr/bin/env python3
"""
run_simulation.py — Load a trained checkpoint and run forward predictions.

Usage
-----
python scripts/run_simulation.py \
    --checkpoint /mnt/results/model_checkpoint.pkl \
    --sequences data/sequences.csv \
    --environment data/environment.csv \
    --output /mnt/results/predictions_table.csv \
    --horizon 90 \
    --mc-samples 200
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.probabilistic_engine import ViralEvolutionPredictor
from simulation.prediction_engine import PredictionEngine
from features.feature_pipeline import FeaturePipeline
from utils.data_utils import load_checkpoint, save_results_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_simulation")


def main():
    parser = argparse.ArgumentParser(description="Run viral evolution simulation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", default="/mnt/results/predictions_table.csv")
    parser.add_argument("--horizon", type=int, default=90, help="Forecast horizon in days")
    parser.add_argument("--mc-samples", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # ---- Load data ----
    logger.info("Loading sequences and environment data...")
    seq_df = pd.read_csv(args.sequences)
    env_df = pd.read_csv(args.environment)

    sequences = seq_df["sequence"].tolist()
    seq_ids = seq_df.get("sequence_id", pd.Series([f"seq_{i}" for i in range(len(sequences))])).tolist()

    # ---- Load checkpoint ----
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt["config"]

    # Rebuild feature pipeline (needs to be re-fitted on the same data)
    pipeline = FeaturePipeline(
        sequence_config={"encoding_method": "one_hot", "window_size": 100, "overlap": 50},
        environmental_config={"continuous_method": "standard", "seasonal_features": True},
        temporal_config={"time_window_days": 30, "min_sequences_per_window": 3},
    )
    features = pipeline.fit_transform(seq_df, env_df)

    seq_dim = features["input_sequences"].shape[-1]
    env_dim = features["environmental_features"].shape[-1]

    # Rebuild model
    model = ViralEvolutionPredictor(
        sequence_dim=seq_dim,
        env_dim=env_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 4),
        dropout_rate=0.1,
        use_bayesian=False,
        device=args.device,
    )
    model.load_state_dict(ckpt["model_state"])
    logger.info("Model loaded (epoch %d)", ckpt.get("epoch", "?"))

    # ---- Run predictions ----
    engine = PredictionEngine(
        model=model,
        feature_pipeline=pipeline,
        num_mc_samples=args.mc_samples,
        device=args.device,
    )

    logger.info("Running predictions for %d sequences, horizon=%d days...",
                len(sequences), args.horizon)
    predictions_df = engine.predict(
        sequences=sequences,
        env_df=env_df,
        horizon_days=args.horizon,
        sequence_ids=seq_ids,
    )

    # ---- Save output ----
    out_path = save_results_csv(predictions_df, Path(args.output).name)
    logger.info("Predictions saved: %s (%d rows)", out_path, len(predictions_df))
    print(predictions_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
