"""Likelihood utilities."""
from __future__ import annotations

import torch


def negative_log_likelihood(y: torch.Tensor, cov_matrix: torch.Tensor) -> torch.Tensor:
    """Compute zero-mean Gaussian negative log-likelihood.

    A matrix-valued ``y`` is flattened in location-major (row-major) order.
    """
    n = y.numel()
    if cov_matrix.shape != (n, n):
        raise ValueError(
            f"covariance shape {tuple(cov_matrix.shape)} does not match {n} observations"
        )
    L = torch.linalg.cholesky(cov_matrix, upper=False)
    y_vector = y.reshape(-1, 1).to(device=cov_matrix.device, dtype=cov_matrix.dtype)
    alpha = torch.linalg.solve_triangular(L, y_vector, upper=False)
    log_likelihood = 0.5 * torch.sum(alpha ** 2)
    log_likelihood += torch.sum(torch.log(torch.diag(L)))
    log_likelihood += 0.5 * n * torch.log(
        torch.tensor(2 * torch.pi, device=cov_matrix.device, dtype=cov_matrix.dtype)
    )
    return log_likelihood
