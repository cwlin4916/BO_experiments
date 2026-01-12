# Sparse Compositional Bayesian Optimization Benchmark

A physics-informed benchmarking suite for evaluating Bayesian Optimization (BO) algorithms on high-dimensional, sparse chemical formulation problems.

## 1. Overview
This repository implements a synthetic data generator based on **Scheffé's Quadratic Mixture Models** [Scheffé, 1958]. Unlike standard BBOB benchmarks, this suite respects the geometric constraints of the simplex ($\sum x_i = 1$) and models chemical phenomena like synergism and antagonism.

### Mathematical Formulation
The ground truth oracle is defined as:
$$\eta(x) = \sum_{i \in \mathcal{A}} \beta_i x_i + \sum_{i,j \in \mathcal{A}, i<j} \beta_{ij} x_i x_j + \epsilon$$
Where $\mathcal{A}$ is a sparse subset of active ingredients ($|\mathcal{A}| \ll D$).

## 2. Code Structure & Overview

### Core Logic
*   **`src/scheffe_generator.py`**: The heart of the benchmark. It processes the ground truth oracle using **Scheffé's Quadratic Mixture Models**.
    *   **Variants**: Supports three distinct problem types:
        *   `A` (Linear / Main Effects Dominant)
        *   `B` (Sparse Synergism / PBT-like)
        *   `C` (Antagonism)
    *   **Features**: Handles sparsity (active vs. inactive ingredients), geometric constraints (simplex), and noise injection.

### Scripts
*   **`scripts/generate_data.py`**: The data factory.
    *   Generates "frozen" benchmark datasets in batch.
    *   Ensures reproducibility by fixing seeds for each dataset instance.
    *   Saves content as `.pt` files containing: configuration, initial training data (X, Y), and the hidden ground truth (optimum X and f*).

*   **`scripts/benchmark_bo2.py`**: The evaluation engine.
    *   Loads the generated `.pt` datasets.
    *   Runs a standard Bayesian Optimization loop using **BoTorch** (SingleTaskGP, LogExpectedImprovement).
    *   Tracks **Inference Regret** (difference between global optimum and the model's best guess).
    *   Aggregates results across variants and produces summary plots (`benchmark_results.png`).

*   **`scripts/visualize_interactive.py`**: A helper tool to intuitively understand the landscapes. Provides a GUI using `matplotlib.widgets` to adjust mixture coefficients and visualize the resulting 1D slices of the response surface.

*   **`scripts/check_validity.py`**: Unit tests to ensure the generator respects simplex constraints ($\sum x_i = 1$) and sparsity rules.

### Output
*   `datasets/`: Stores the generated frozen datasets (organized by dimension `D` and count `N`).
*   `results/`: Destination for benchmark plots, GIFs, and raw result tensors.

## 3. Usage

### 1. Prerequisite (One-time)
If you (or your collaborators) haven't done it yet, install the package in editable mode:
```bash
pip install -e .
```

### 2. Running the Scripts
Run these commands from the **root directory**:

*   **Benchmark Loop (creates GIF in `results/`):**
    ```bash
    python scripts/benchmark_bo.py
    ```

*   **Sanity Check (creates PNG in `results/`):**
    ```bash
    python scripts/check_validity.py
    ```

*   **Interactive Visualization:**
    ```bash
    python scripts/visualize_interactive.py
    ```