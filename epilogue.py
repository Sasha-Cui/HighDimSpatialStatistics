"""Legacy epilogue helper.

If executed with `df_to_plot` and `notebook_name` in globals, this will save
outputs to `data/processed/` with a timestamped filename. New code should call
`HighDimSpatial.utils.output.save_dataframe` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_repo_root = Path(__file__).resolve().parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from HighDimSpatial.utils.output import plot_param_histograms, save_dataframe


def save_dataframe_with_histograms(
    df_to_plot: pd.DataFrame,
    notebook_name: str,
    histograms_are_plotted: bool = False,
    ground_truth_df: Optional[pd.DataFrame] = None,
) -> None:
    path = save_dataframe(df_to_plot, notebook_name)
    print(f"DataFrame saved to {path}")
    if histograms_are_plotted:
        plot_param_histograms(df_to_plot, ground_truth_df=ground_truth_df)


# Legacy behavior: if run via %run with expected globals, execute automatically
if "df_to_plot" in globals() and "notebook_name" in globals():
    save_dataframe_with_histograms(
        df_to_plot=globals()["df_to_plot"],
        notebook_name=globals()["notebook_name"],
        histograms_are_plotted=globals().get("histograms_are_plotted", False),
        ground_truth_df=globals().get("ground_truth_df"),
    )
