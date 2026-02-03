"""Synthetic data generation helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from HighDimSpatial.simulation.generate import Genton_parametrisation, simulate_locations, simulate_gp_data
from HighDimSpatial.kernels.matern import compute_matern_covariance
from HighDimSpatial.utils.linalg import symmetrize


@dataclass
class SyntheticDataResult:
    X: torch.Tensor
    Y: torch.Tensor
    K: torch.Tensor
    alpha_matrix: torch.Tensor
    nu_matrix: torch.Tensor
    sigma_matrix: torch.Tensor


def generate_genton_synthetic(number_of_locations: int, dims: int = 2) -> SyntheticDataResult:
    X, Y, K, alpha_matrix, nu_matrix, sigma_matrix = Genton_parametrisation(number_of_locations, dims)
    return SyntheticDataResult(X=X, Y=Y, K=K, alpha_matrix=alpha_matrix, nu_matrix=nu_matrix, sigma_matrix=sigma_matrix)
