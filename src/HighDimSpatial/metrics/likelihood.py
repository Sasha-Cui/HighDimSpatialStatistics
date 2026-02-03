"""Likelihood utilities."""
from __future__ import annotations

import torch


def negative_log_likelihood(y: torch.Tensor, cov_matrix: torch.Tensor) -> torch.Tensor:
    """Compute negative log-likelihood under a zero-mean Gaussian."""
    n = y.shape[0]
    L = torch.linalg.cholesky(cov_matrix, upper=False)
    alpha = torch.linalg.solve_triangular(L, y.reshape(-1, 1), upper=False)
    log_likelihood = 0.5 * torch.sum(alpha ** 2)
    log_likelihood += torch.sum(torch.log(torch.diag(L)))
    log_likelihood += 0.5 * n * torch.log(torch.tensor(2 * torch.pi, device=y.device, dtype=y.dtype))
    return log_likelihood
