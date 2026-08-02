"""Validation metrics for fitted models."""
from __future__ import annotations

import torch

from HighDimSpatial.kernels.approx import approx_matern_kernel_marginal


def extract_location_major_marginal_covariance(
    covariance: torch.Tensor,
    feature: int,
    number_of_variables: int,
) -> torch.Tensor:
    """Extract one feature's spatial covariance from location-major stacking."""
    if covariance.ndim != 2 or covariance.size(0) != covariance.size(1):
        raise ValueError("covariance must be square")
    if number_of_variables < 1 or covariance.size(0) % number_of_variables != 0:
        raise ValueError("covariance dimension must be divisible by number_of_variables")
    if not 0 <= feature < number_of_variables:
        raise ValueError("feature index is out of range")
    return covariance[feature::number_of_variables, feature::number_of_variables]


def validation_metric_marginal(
    optimized_marginal_params: dict,
    X_test: torch.Tensor,
    K_test: torch.Tensor,
) -> float:
    """Relative squared Frobenius distance between predicted and true covariances."""
    alpha = optimized_marginal_params["alpha"]
    nu = optimized_marginal_params["nu"]
    sigma = optimized_marginal_params["sigma"]

    device = X_test.device
    if alpha.device != device:
        alpha = alpha.to(device)
    if nu.device != device:
        nu = nu.to(device)
    if sigma.device != device:
        sigma = sigma.to(device)

    K_pred = approx_matern_kernel_marginal(X_test, alpha, nu, sigma)
    frob_diff = torch.norm(K_pred - K_test, p="fro") ** 2
    frob_true = torch.norm(K_test, p="fro") ** 2
    return (frob_diff / frob_true).item()


# Backwards-compat name
validation_metric = validation_metric_marginal
