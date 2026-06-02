"""
Visualization utilities for the genomic epidemiology pipeline.

All functions save figures to ``/mnt/results/figures/`` by default and
return the output path.  Requires matplotlib and seaborn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette (Phylo brand colours + colourblind-safe extras)
PALETTE = ["#0279EE", "#FF9400", "#75A025", "#E9ED4C", "#FD9BED", "#000000"]
sns.set_theme(style="whitegrid", palette=PALETTE)


# ---------------------------------------------------------------------------
# 1. Mutation probability heatmap
# ---------------------------------------------------------------------------

def plot_mutation_heatmap(
    probs: np.ndarray,
    positions: Optional[List[int]] = None,
    title: str = "Predicted Mutation Probabilities",
    filename: str = "mutation_heatmap.png",
    top_n_positions: int = 50,
) -> str:
    """
    Heatmap of per-position nucleotide probabilities.

    Parameters
    ----------
    probs : np.ndarray, shape (L, 4)
        Predicted probabilities for A, T, G, C at each position.
    positions : list of int, optional
        Genomic positions for x-axis labels.
    title : str
    filename : str
    top_n_positions : int
        Show only the top-N most variable positions.

    Returns
    -------
    str  — path to saved PNG
    """
    L = probs.shape[0]
    if positions is None:
        positions = list(range(L))

    # Select most variable positions
    variability = 1 - np.max(probs, axis=1)  # 1 - max_prob as variability proxy
    top_idx = np.argsort(variability)[-top_n_positions:][::-1]
    top_idx = np.sort(top_idx)

    data = probs[top_idx, :]
    xlabels = [str(positions[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(min(20, top_n_positions * 0.4 + 2), 4))
    sns.heatmap(
        data.T,
        ax=ax,
        xticklabels=xlabels,
        yticklabels=["A", "T", "G", "C"],
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.3,
        cbar_kws={"label": "Probability"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Genomic Position", fontsize=11)
    ax.set_ylabel("Nucleotide", fontsize=11)
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()

    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mutation heatmap: %s", out)
    return str(out)


# ---------------------------------------------------------------------------
# 2. Fitness trajectory
# ---------------------------------------------------------------------------

def plot_fitness_trajectory(
    mean_fitness: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    time_labels: Optional[List[str]] = None,
    title: str = "Predicted Fitness Trajectory",
    filename: str = "fitness_trajectory.png",
) -> str:
    """
    Line plot of fitness over time with optional confidence ribbon.

    Parameters
    ----------
    mean_fitness : np.ndarray, shape (T,)
    ci_lower, ci_upper : np.ndarray, shape (T,), optional
    time_labels : list of str, optional
    title, filename : str

    Returns
    -------
    str — path to saved PNG
    """
    T = len(mean_fitness)
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, mean_fitness, color=PALETTE[0], linewidth=2, label="Mean fitness")

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(x, ci_lower, ci_upper, alpha=0.25, color=PALETTE[0], label="95% CI")

    if time_labels is not None:
        step = max(1, T // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(time_labels[::step], rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xlabel("Time Step", fontsize=11)

    ax.set_ylabel("Fitness Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fitness trajectory: %s", out)
    return str(out)


# ---------------------------------------------------------------------------
# 3. Calibration curve (reliability diagram)
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    expected_coverages: List[float],
    empirical_coverages: List[float],
    model_names: Optional[List[str]] = None,
    title: str = "Prediction Calibration",
    filename: str = "calibration_curve.png",
) -> str:
    """
    Reliability diagram: empirical vs. expected coverage.

    Parameters
    ----------
    expected_coverages : list of float
        Nominal confidence levels, e.g. [0.5, 0.6, …, 0.95].
    empirical_coverages : list of float or list of lists
        Observed coverage at each level.  Pass a list of lists to
        overlay multiple models.
    model_names : list of str, optional
    title, filename : str

    Returns
    -------
    str — path to saved PNG
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Handle single model (flat list) vs multiple models (list of lists)
    if empirical_coverages and not isinstance(empirical_coverages[0], (list, np.ndarray)):
        empirical_coverages = [empirical_coverages]
        if model_names is None:
            model_names = ["Model"]

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(empirical_coverages))]

    for i, (emp, name) in enumerate(zip(empirical_coverages, model_names)):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(expected_coverages, emp, "o-", color=color, linewidth=2,
                markersize=6, label=name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration curve: %s", out)
    return str(out)


# ---------------------------------------------------------------------------
# 4. Benchmark comparison bar chart
# ---------------------------------------------------------------------------

def plot_benchmark_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Benchmark Comparison",
    filename: str = "benchmark_comparison.png",
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Grouped bar chart comparing models across multiple metrics.

    Parameters
    ----------
    model_names : list of str
    metrics : dict  {metric_name: [value_per_model]}
    title, filename : str
    higher_is_better : dict {metric_name: bool}, optional
        Used to annotate axes with ↑/↓ arrows.

    Returns
    -------
    str — path to saved PNG
    """
    if higher_is_better is None:
        higher_is_better = {}

    n_models = len(model_names)
    n_metrics = len(metrics)
    metric_names = list(metrics.keys())

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))

    for i, model in enumerate(model_names):
        vals = [metrics[m][i] for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model, color=PALETTE[i % len(PALETTE)], alpha=0.85)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and not np.isinf(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    # X-axis labels with direction arrows
    xlabels = []
    for m in metric_names:
        arrow = " ↑" if higher_is_better.get(m, True) else " ↓"
        xlabels.append(m.replace("_", " ").title() + arrow)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved benchmark comparison: %s", out)
    return str(out)
