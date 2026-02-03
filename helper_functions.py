"""Legacy helper functions shim.

This module preserves the original API used by notebooks, but delegates
implementation to the cleaned `HighDimSpatial` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_repo_root = Path(__file__).resolve().parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from HighDimSpatial.config import DEFAULT_DTYPE, get_device
from HighDimSpatial.data.io import extract_gene_values as _extract_gene_values
from HighDimSpatial.data.real import load_real_data
from HighDimSpatial.data.smoothing import epanechnikov_kernel, kernel_smoothing
from HighDimSpatial.data.synthetic import generate_genton_synthetic
from HighDimSpatial.kernels.approx import (
    adjust_matrix_with_nugget,
    approx_matern_kernel_cross,
    approx_matern_kernel_cross_legacy,
    approx_matern_kernel_marginal,
    approx_matern_kernel_marginal_old,
)
from HighDimSpatial.kernels.matern import (
    compute_matern_covariance,
    compute_parameter_matrices,
    matern_kernel,
)
from HighDimSpatial.kernels.psd import (
    cross_psd_condition_checker,
    marginal_approx_psd_condition_checker,
    marginal_psd_condition_checker,
)
from HighDimSpatial.metrics.likelihood import negative_log_likelihood
from HighDimSpatial.metrics.validation import validation_metric, validation_metric_marginal
from HighDimSpatial.plotting.plots import plot_gp_data
from HighDimSpatial.simulation.generate import (
    Genton_parametrisation,
    Genton_parametrisation_fixed_locations,
    simulate_gp_data,
    simulate_locations,
)
from HighDimSpatial.utils.dataframe import store_as_df
from HighDimSpatial.utils.linalg import is_positive_definite


device = get_device()


def isolate_gene_values(adata, gene_name):
    return _extract_gene_values(adata, gene_name, device=device, dtype=DEFAULT_DTYPE)


def load_data(gene_list, head=0, puck_list="all", filename="mouse_ovary_slide_seq_young_estrus.h5ad", subdir="raw"):
    """Legacy wrapper for loading real data.

    Place the .h5ad file under `data/raw/` or pass `subdir=None` with a full path.
    """
    puck_list = None if puck_list == "all" else puck_list
    result = load_real_data(
        gene_list=gene_list,
        filename=filename,
        subdir=subdir,
        head=head,
        puck_list=puck_list,
        prefer_cuda=torch.cuda.is_available(),
    )
    return result.adata, result.X, result.Y, result.gene_list


def load_synthetic_data(number_of_locations, dims=2):
    """Legacy wrapper for Genton synthetic data generation."""
    return Genton_parametrisation(number_of_locations, dims)


__all__ = [
    "device",
    "isolate_gene_values",
    "load_data",
    "load_synthetic_data",
    "plot_gp_data",
    "is_positive_definite",
    "matern_kernel",
    "adjust_matrix_with_nugget",
    "approx_matern_kernel_marginal",
    "approx_matern_kernel_marginal_old",
    "compute_matern_covariance",
    "approx_matern_kernel_cross",
    "approx_matern_kernel_cross_legacy",
    "compute_parameter_matrices",
    "simulate_locations",
    "simulate_gp_data",
    "Genton_parametrisation",
    "Genton_parametrisation_fixed_locations",
    "store_as_df",
    "negative_log_likelihood",
    "cross_psd_condition_checker",
    "marginal_approx_psd_condition_checker",
    "marginal_psd_condition_checker",
    "epanechnikov_kernel",
    "kernel_smoothing",
    "validation_metric",
    "validation_metric_marginal",
]
