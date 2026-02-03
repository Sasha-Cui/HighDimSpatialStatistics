"""Validation metrics for fitted models."""
from __future__ import annotations

import torch

from HighDimSpatial.kernels.approx import approx_matern_kernel_marginal


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
