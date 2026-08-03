"""Matérn kernel utilities.

The canonical parameter convention in this module is the one used by
Apanasovich, Genton, and Sun (2012): ``alpha`` is a *decay scale*, so a larger
value gives faster decay,

``M(h; nu, alpha) = 2**(1-nu) / Gamma(nu) * (alpha*h)**nu * K_nu(alpha*h)``.

Multivariate observations are stacked in location-major order throughout:
``Y.reshape(-1) == (Y[0, 0], ..., Y[0, p-1], Y[1, 0], ...)``.
"""
from __future__ import annotations

import torch
from scipy.special import kv, kvp

from HighDimSpatial.utils.linalg import symmetrize


class BesselKFunction(torch.autograd.Function):
    """Autograd wrapper for the modified Bessel function of the second kind (kv)."""

    @staticmethod
    def forward(ctx, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if v.device != x.device:
            raise ValueError("v and x must be on the same device")
        ctx.save_for_backward(v, x)
        v_np = v.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        dtype = torch.promote_types(v.dtype, x.dtype)
        return torch.as_tensor(kv(v_np, x_np), dtype=dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, x = ctx.saved_tensors
        v_np = v.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()

        # scipy.special.kvp differentiates with respect to the argument x.
        grad_x = torch.as_tensor(kvp(v_np, x_np, n=1), dtype=grad_output.dtype, device=x.device)

        # SciPy has no derivative with respect to the order v.  A scale-aware
        # central difference is stable enough for the compact parameter ranges
        # used here and, crucially, differentiates the correct variable.
        step_v = 1e-5 * (1.0 + abs(v_np))
        grad_v_np = (kv(v_np + step_v, x_np) - kv(v_np - step_v, x_np)) / (2.0 * step_v)
        grad_v = torch.as_tensor(grad_v_np, dtype=grad_output.dtype, device=v.device)

        # Broadcasting happens inside scipy.special.kv.  Custom autograd
        # functions must explicitly reduce broadcast dimensions on return.
        grad_input_x = (grad_output * grad_x).sum_to_size(x.shape).to(x.dtype)
        grad_input_v = (grad_output * grad_v).sum_to_size(v.shape).to(v.dtype)
        return grad_input_v, grad_input_x


def matern_correlation(
    pairwise_distances: torch.Tensor,
    alpha: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """Compute a Matérn correlation under the AGS decay-scale convention.

    ``alpha`` is an inverse range: larger values imply faster decay.  The
    function supports broadcasting and is differentiable in ``alpha`` and
    ``nu`` through the custom Bessel wrapper above.
    """
    device = pairwise_distances.device
    alpha = alpha.to(device)
    nu = nu.to(device)

    if torch.any(pairwise_distances < 0):
        raise ValueError("pairwise distances must be nonnegative")
    if torch.any(alpha <= 0):
        raise ValueError("alpha must be positive")
    if torch.any(nu <= 0):
        raise ValueError("nu must be positive")

    scaled_distances = torch.clamp(alpha * pairwise_distances, min=1e-10, max=1e6)
    bessel_term = BesselKFunction.apply(nu, scaled_distances)
    scaling_term = torch.pow(torch.as_tensor(2.0, dtype=nu.dtype, device=device), 1.0 - nu) / torch.exp(
        torch.lgamma(nu)
    )
    correlation = scaling_term * torch.pow(scaled_distances, nu) * bessel_term
    return torch.where(pairwise_distances == 0, torch.ones_like(correlation), correlation)


def matern_kernel(
    pairwise_distances: torch.Tensor,
    alpha: torch.Tensor,
    nu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute a marginal Matérn covariance (supports broadcasting).

    Args:
        pairwise_distances: Pairwise distances (n, n) or broadcasted.
        alpha: Positive decay-scale (inverse-range) parameter.
        nu: Smoothness parameter.
        sigma: Standard deviation parameter.
    """
    sigma = sigma.to(pairwise_distances.device)
    if torch.any(sigma < 0):
        raise ValueError("marginal standard deviation sigma must be nonnegative")
    return sigma.square() * matern_correlation(pairwise_distances, alpha, nu)


def matern_cross_kernel(
    pairwise_distances: torch.Tensor,
    alpha: torch.Tensor,
    nu: torch.Tensor,
    zero_lag_covariance: torch.Tensor,
) -> torch.Tensor:
    """Compute a (possibly signed) Matérn cross-covariance.

    Unlike :func:`matern_kernel`, the amplitude is a zero-lag covariance, not
    a standard deviation, and therefore must not be squared.
    """
    amplitude = zero_lag_covariance.to(pairwise_distances.device)
    return amplitude * matern_correlation(pairwise_distances, alpha, nu)


def compute_matern_covariance(
    alpha_matrix: torch.Tensor,
    nu_matrix: torch.Tensor,
    sigma_matrix: torch.Tensor,
    X: torch.Tensor,
) -> torch.Tensor:
    """Compute a location-major multivariate Matérn covariance matrix.

    Historical callers supply marginal standard deviations on the diagonal of
    ``sigma_matrix`` and signed zero-lag covariances off diagonal.  This mixed
    convention is retained at the public boundary for compatibility, then
    converted to an unambiguous zero-lag covariance matrix internally.
    """
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    pairwise_distances = torch.cdist(X, X)
    pairwise_distances.fill_diagonal_(0)

    pairwise_distances_expanded = pairwise_distances.unsqueeze(0).unsqueeze(0).expand(p, p, n_locations, n_locations)
    alpha_expanded = alpha_matrix.unsqueeze(-1).unsqueeze(-1)
    nu_expanded = nu_matrix.unsqueeze(-1).unsqueeze(-1)
    zero_lag_covariance = sigma_matrix.clone()
    diagonal = torch.arange(p, device=sigma_matrix.device)
    zero_lag_covariance[diagonal, diagonal] = sigma_matrix.diagonal().square()
    amplitude_expanded = zero_lag_covariance.unsqueeze(-1).unsqueeze(-1)

    K_blocks = matern_cross_kernel(pairwise_distances_expanded, alpha_expanded, nu_expanded, amplitude_expanded)
    # K_blocks[a, b, i, j] -> K[(i, a), (j, b)] so Y.reshape(-1)
    # uses the same location-major convention as the covariance.
    K = K_blocks.permute(2, 0, 3, 1).reshape(p * n_locations, p * n_locations)
    return symmetrize(K)


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
