#!/usr/bin/env python3
"""
evaluate_predictions.py — Compute evaluation metrics from a predictions CSV.

Usage
-----
python scripts/evaluate_predictions.py \
    --predictions /mnt/results/predictions_table.csv \
    --ground-truth data/sequences.csv \
    --output /mnt/results/evaluation_report.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.metrics import nucleotide_accuracy, sequence_mse, compute_all_metrics
from utils.data_utils import save_results_csv, save_results_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_predictions")


def main():
    parser = argparse.ArgumentParser(description="Evaluate viral evolution predictions")
    parser.add_argument("--predictions", required=True, help="Predictions CSV from run_simulation.py")
    parser.add_argument("--ground-truth", required=True, help="Ground truth sequences CSV")
    parser.add_argument("--output", default="/mnt/results/evaluation_report.csv")
    args = parser.parse_args()

    # ---- Load data ----
    pred_df = pd.read_csv(args.predictions)
    gt_df = pd.read_csv(args.ground_truth)

    logger.info("Predictions: %d rows", len(pred_df))
    logger.info("Ground truth: %d sequences", len(gt_df))

    # Use step=0 (one-step-ahead) predictions for evaluation
    step0 = pred_df[pred_df["step"] == 0].copy()

    # Match by sequence_id
    gt_map = dict(zip(gt_df.get("sequence_id", gt_df.index.astype(str)), gt_df["sequence"]))

    matched_pred = []
    matched_actual = []
    for _, row in step0.iterrows():
        sid = row["sequence_id"]
        if sid in gt_map:
            matched_pred.append(row["predicted_sequence"])
            matched_actual.append(gt_map[sid])

    if not matched_pred:
        logger.warning("No matching sequence IDs found. Using positional matching.")
        matched_pred = step0["predicted_sequence"].tolist()
        matched_actual = gt_df["sequence"].tolist()[: len(matched_pred)]

    logger.info("Evaluating %d matched pairs...", len(matched_pred))

    # ---- Compute metrics ----
    metrics = compute_all_metrics(matched_pred, matched_actual)

    # Per-sequence metrics
    per_seq_rows = []
    for pred, actual in zip(matched_pred, matched_actual):
        per_seq_rows.append({
            "nucleotide_accuracy": nucleotide_accuracy([pred], [actual]),
            "sequence_mse": sequence_mse([pred], [actual]),
            "length": len(actual),
        })
    per_seq_df = pd.DataFrame(per_seq_rows)

    # ---- Save outputs ----
    save_results_csv(per_seq_df, Path(args.output).name)
    save_results_json(metrics, "evaluation_metrics.json")

    # Print summary
    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"  {k:<30} {v:.4f}")
    print(f"\nPer-sequence stats:")
    print(per_seq_df.describe().to_string())


if __name__ == "__main__":
    main()
