# Environmental Conditioning for Viral Evolution Prediction

[![License](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)

A probabilistic machine learning framework that integrates macroscopic ecological and environmental selective pressures (temperature, humidity, host-species reservoirs, and population density) with microscopic nucleotide mutation dynamics using an **Environmental-Conditioned Transformer** (ECT).

This repository contains the complete implementation, simulation engine, and validation pipeline developed as an independent, open-source bioinformatics research project.

---

## 🗺️ Project Architecture

The codebase implements a decoupled, production-ready pipeline structured as follows:

```text
ECT/
├── MANUSCRIPT_DRAFT.tex     # Publication-grade LaTeX manuscript
├── references.bib           # Cross-checked BibTeX citation database
├── pipeline_runner.py       # Central orchestrator / execution entry point
├── requirements.txt         # Package dependencies
├── model_config.yaml        # Hyperparameter configuration
├── settings.py              # Global runtime settings
│
├── genomic_epi_pipeline/
│   ├── data/                # Data ingestion and alignment pipelines
│   │   ├── ingestion/       # Sequence fetching and environmental scraping
│   │   └── preprocessing/   # Masking and sequence alignment modules
│   ├── features/            # Feature tokenizers and vector normalization
│   ├── models/              # PyTorch ECT neural network architectures
│   ├── simulation/          # Monte Carlo trajectory engines & hindcasting
│   ├── utils/               # Bio-informatics helpers and visualization assets
│   ├── validation/          # Epistemic uncertainty and calibration metrics
│   └── results/
│       └── figures/         # Auto-generated high-resolution evaluation charts

```

---

## 📈 Core Performance Benchmarks

The ECT framework was validated against a second-order Baseline Markov Chain using strict chronological temporal splitting over an 18-month simulation window:

| Model | Nucleotide Accuracy | Log-Likelihood | Perplexity |
| --- | --- | --- | --- |
| **Markov Chain (k=2)** | 21.74% | -1.4868 | 4.4229 |
| **ECT (Proposed Framework)** | **97.24%** | **-0.1077** | **1.1137** |

### Key Analytical Findings

* **Macro-to-Micro Routing:** The ECT successfully maps complex environmental thresholds (e.g., seasonal temperature forcing and host-species reservoir transitions) directly into point-mutation probabilities without explicit hardcoding.
* **Hardware Acceleration:** By expanding temporal simulation vectors into parallel instances using `X_batch = X.expand(N, -1, -1)`, the Monte Carlo engine bypasses sequential loop overhead, executing fully vectorized sampling on Apple Silicon hardware acceleration layers (`mps`).
* **Epistemic Limitations:** The network exhibits standard deep learning overconfidence (Calibration Error = 0.7417), pinpointing boundaries for post-hoc temperature scaling prior to real-world deployment.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9 or higher
* PyTorch (configured for CUDA or Apple Silicon MPS acceleration)

### Installation

1. Clone this repository to your local cluster or workstation:

```bash
   git clone [https://github.com/Nishant808/ECT.git](https://github.com/Nishant808/ECT.git)
   cd ECT

```

2. Install the required bioinformatics and deep learning dependencies:

```bash
   pip install -r requirements.txt

```

### Execution

To initialize the pipeline, generate the synthetic environmental tracking arrays, train the variational inference model layers, and export the comparative visualization charts into the results folder, run the central orchestrator:

```bash
python pipeline_runner.py

```

---

## 📊 Auto-Generated Outputs

Upon successful execution, the pipeline deposits high-resolution evaluation figures into `genomic_epi_pipeline/results/figures/`:

* `benchmark_comparison.png`: Metric comparison against the Markov baseline.
* `calibration_curve.png`: Reliability diagram measuring empirical probability coverage.
* `mutation_heatmap.png`: Positional nucleotide probability shifts across high-entropy positions.
* `fitness_trajectory.png`: Temporal tracking of spatial thermodynamic stability bounds.

---

## ✉️ Contact & Support

**Author:** Nishant Thalwal

**Email:** [nishant.thalwal@stud.th-deg.de](mailto:nishant.thalwal@stud.th-deg.de)

*Independent Researcher*
