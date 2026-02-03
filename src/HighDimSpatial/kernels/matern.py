"""Matérn kernel utilities."""
from __future__ import annotations

import torch
from scipy.special import kv, kvp

from HighDimSpatial.utils.linalg import symmetrize


class BesselKFunction(torch.autograd.Function):
    """Autograd wrapper for the modified Bessel function of the second kind (kv)."""

    @staticmethod
    def forward(ctx, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v, x)
        v_np = v.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        out = torch.tensor(kv(v_np, x_np), dtype=torch.float64)
        return out.to(x.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, x = ctx.saved_tensors
        v_np = v.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()

        epsilon_x = 1e-12
        grad_x = (kv(v_np, x_np + epsilon_x) - kv(v_np, x_np - epsilon_x)) / (2 * epsilon_x)
        grad_x = torch.tensor(grad_x, dtype=torch.float64).to(x.device)

        grad_v = torch.tensor(kvp(v_np, x_np), dtype=torch.float64).to(v.device)

        grad_input_x = grad_output * grad_x
        grad_input_v = grad_output * grad_v
        return grad_input_v, grad_input_x


def matern_kernel(
    pairwise_distances: torch.Tensor,
    length_scale: torch.Tensor,
    nu: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Compute the Matérn covariance matrix (supports broadcasting).

    Args:
        pairwise_distances: Pairwise distances (n, n) or broadcasted.
        length_scale: Length-scale parameter.
        nu: Smoothness parameter.
        sigma: Standard deviation parameter.
        epsilon: Small perturbation to avoid nu == 0.5 edge case.
    """
    if pairwise_distances.device != length_scale.device:
        length_scale = length_scale.to(pairwise_distances.device)
    if nu.device != pairwise_distances.device:
        nu = nu.to(pairwise_distances.device)
    if sigma.device != pairwise_distances.device:
        sigma = sigma.to(pairwise_distances.device)

    sigma2 = sigma ** 2
    nu = torch.where(nu == 0.5, nu + epsilon, nu)

    scaled_distances = torch.sqrt(2 * nu) * (pairwise_distances / length_scale)
    scaled_distances = torch.clamp(scaled_distances, min=1e-10, max=1e6)

    bessel_term = BesselKFunction.apply(nu, scaled_distances)
    scaling_term = (2 ** (1.0 - nu)) / torch.exp(torch.lgamma(nu))
    covariance_matrix = sigma2 * scaling_term * (scaled_distances ** nu) * bessel_term

    covariance_matrix = torch.where(pairwise_distances == 0, sigma2, covariance_matrix)
    return covariance_matrix


def compute_matern_covariance(
    alpha_matrix: torch.Tensor,
    nu_matrix: torch.Tensor,
    sigma_matrix: torch.Tensor,
    X: torch.Tensor,
) -> torch.Tensor:
    """Compute the block Matérn covariance matrix for multivariate processes."""
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    pairwise_distances = torch.cdist(X, X)
    pairwise_distances.fill_diagonal_(0)

    pairwise_distances_expanded = pairwise_distances.unsqueeze(0).unsqueeze(0).expand(p, p, n_locations, n_locations)
    alpha_expanded = alpha_matrix.unsqueeze(-1).unsqueeze(-1)
    nu_expanded = nu_matrix.unsqueeze(-1).unsqueeze(-1)
    sigma_expanded = sigma_matrix.unsqueeze(-1).unsqueeze(-1)

    K_blocks = matern_kernel(pairwise_distances_expanded, alpha_expanded, nu_expanded, sigma_expanded)
    K = K_blocks.permute(0, 2, 1, 3).reshape(p * n_locations, p * n_locations)
    return K


def compute_parameter_matrices(
    Delta_A: torch.Tensor,
    Delta_B: torch.Tensor,
    rho_A: torch.Tensor,
    rho_B: torch.Tensor,
    rho_V: torch.Tensor,
    W: torch.Tensor,
    alpha: torch.Tensor,
    nu: torch.Tensor,
    sigma: torch.Tensor,
    dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute alpha/nu/sigma matrices from cross parameters."""
    p = W.size(0)

    alpha_i_squared = (alpha.unsqueeze(1) ** 2)
    alpha_j_squared = (alpha.unsqueeze(0) ** 2)
    alpha_matrix = torch.sqrt((alpha_i_squared + alpha_j_squared) / 2 + Delta_B * (1 - rho_B))

    nu_matrix = ((nu.unsqueeze(1) + nu.unsqueeze(0)) / 2 + Delta_A * (1 - rho_A))

    W_i = W.unsqueeze(1)
    W_j = W.unsqueeze(0)
    sigma_matrix = (
        W_i
        * W_j
        * rho_V
        * alpha_matrix ** (-2 * Delta_A - (nu.unsqueeze(0) + nu.unsqueeze(1)))
        * torch.exp(
            torch.lgamma((nu.unsqueeze(0) + nu.unsqueeze(1)) / 2 + dim / 2)
            + torch.lgamma(nu_matrix)
            - torch.lgamma(nu_matrix + dim / 2)
        )
    )

    alpha_matrix = alpha_matrix + torch.diag(alpha - torch.diag(alpha_matrix))
    nu_matrix = nu_matrix + torch.diag(nu - torch.diag(nu_matrix))
    sigma_matrix = sigma_matrix + torch.diag(sigma - torch.diag(sigma_matrix))

    return alpha_matrix, nu_matrix, sigma_matrix
