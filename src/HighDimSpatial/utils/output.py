"""Output helpers (saving results and plotting summaries)."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from HighDimSpatial.config import resolve_data_path


def save_dataframe(
    df: pd.DataFrame,
    name: str,
    output_dir: Optional[Path] = None,
    include_slurm_id: bool = True,
) -> Path:
    """Save a DataFrame with a timestamped filename.

    Args:
        df: DataFrame to save.
        name: Base name (e.g., notebook/script name).
        output_dir: Optional output directory. Defaults to data/processed.
        include_slurm_id: Include SLURM_JOB_ID in filename when available.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.getenv("SLURM_JOB_ID", "no_slurm_id") if include_slurm_id else "no_slurm_id"

    if output_dir is None:
        output_dir = resolve_data_path("processed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{name}_job_{slurm_job_id}_time_{timestamp}.csv"
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    return filepath


def plot_param_histograms(df: pd.DataFrame, ground_truth_df: Optional[pd.DataFrame] = None) -> None:
    """Plot histograms for each parameter column with optional ground-truth overlay."""
    columns = df.columns
    n_params = len(columns)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)

    plt.figure(figsize=(15, 3 * n_rows))
    for i, col in enumerate(columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(df[col], bins=30, color="skyblue", edgecolor="black")
        if ground_truth_df is not None and col in ground_truth_df.columns:
            plt.axvline(x=ground_truth_df[col].iloc[0], color="red", linestyle="--", linewidth=2)
        plt.title(f"{col}")
        plt.xlabel(f"{col}")
        plt.ylabel("Frequency")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
