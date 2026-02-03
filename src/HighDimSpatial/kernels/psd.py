"""PSD checking helpers."""
from __future__ import annotations

import torch

from HighDimSpatial.kernels.approx import approx_matern_kernel_marginal
from HighDimSpatial.kernels.matern import compute_matern_covariance, compute_parameter_matrices, matern_kernel
from HighDimSpatial.utils.linalg import is_positive_definite, symmetrize


def cross_psd_condition_checker(
    Delta_A: torch.Tensor,
    Delta_B: torch.Tensor,
    rho_A: torch.Tensor,
    rho_B: torch.Tensor,
    rho_V: torch.Tensor,
    W: torch.Tensor,
    alpha: torch.Tensor,
    nu: torch.Tensor,
    sigma: torch.Tensor,
    X_batch: torch.Tensor,
) -> bool:
    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(
        Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma
    )
    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
    K = symmetrize(K) + torch.eye(K.size(0), device=K.device) * 1e-8
    return is_positive_definite(K)


def marginal_approx_psd_condition_checker(
    X_batch: torch.Tensor,
    alpha_i: torch.Tensor,
    nu_i: torch.Tensor,
    sigma_i: torch.Tensor,
) -> bool:
    K = approx_matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
    return is_positive_definite(K)


def marginal_psd_condition_checker(
    X_batch: torch.Tensor,
    alpha_i: torch.Tensor,
    nu_i: torch.Tensor,
    sigma_i: torch.Tensor,
) -> bool:
    K = matern_kernel(torch.cdist(X_batch, X_batch), alpha_i, nu_i, sigma_i)
    K += torch.eye(K.size(0), device=K.device) * 1e-9
    return is_positive_definite(K)
