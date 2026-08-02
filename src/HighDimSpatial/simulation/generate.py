"""Synthetic data generation utilities."""
from __future__ import annotations

import torch

from HighDimSpatial.kernels.matern import compute_matern_covariance
from HighDimSpatial.utils.linalg import symmetrize


def simulate_locations(
    number_of_locations: int,
    dimensions: int = 2,
    range_min: float = -3.0,
    range_max: float = 3.0,
) -> torch.Tensor:
    """Simulate random locations within a range."""
    return torch.empty(number_of_locations, dimensions, dtype=torch.float64).uniform_(range_min, range_max)


def simulate_gp_data(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Simulate ``(location, variable)`` data from a location-major covariance."""
    n_locations = X.size(0)
    p = K.size(0) // n_locations
    mean = torch.zeros(K.size(0), dtype=torch.float64, device=K.device)
    K = symmetrize(K)
    Y = torch.distributions.MultivariateNormal(mean, covariance_matrix=K).rsample()
    return Y.reshape(n_locations, p)


def Genton_parametrisation(
    number_of_locations: int,
    dims: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Genton parametrisation (p=3) synthetic data."""
    true_vals = [
        1.2, 0.01, 1.0,
        0.6, 0.02, 1.0,
        0.3, 0.03, 1.0,
        1.093, 0.0205, -0.286,
        1.092, 0.0263, -0.181,
        0.990, 0.0282, 0.274,
    ]
    (
        true_nu1, true_a1, true_sigma1,
        true_nu2, true_a2, true_sigma2,
        true_nu3, true_a3, true_sigma3,
        true_nu12, true_a12, true_sigma12,
        true_nu13, true_a13, true_sigma13,
        true_nu23, true_a23, true_sigma23,
    ) = true_vals

    alpha_matrix = torch.tensor([
        [true_a1, true_a12, true_a13],
        [true_a12, true_a2, true_a23],
        [true_a13, true_a23, true_a3],
    ])
    nu_matrix = torch.tensor([
        [true_nu1, true_nu12, true_nu13],
        [true_nu12, true_nu2, true_nu23],
        [true_nu13, true_nu23, true_nu3],
    ])
    sigma_matrix = torch.tensor([
        [true_sigma1, true_sigma12, true_sigma13],
        [true_sigma12, true_sigma2, true_sigma23],
        [true_sigma13, true_sigma23, true_sigma3],
    ])

    X = simulate_locations(number_of_locations, dims)
    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    K = symmetrize(K)
    Y = simulate_gp_data(X, K).detach()
    return X, Y, K, alpha_matrix, nu_matrix, sigma_matrix


def Genton_parametrisation_fixed_locations(
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Genton parametrisation with fixed locations."""
    true_vals = [
        1.2, 0.01, 1.0,
        0.6, 0.02, 1.0,
        0.3, 0.03, 1.0,
        1.093, 0.0205, -0.286,
        1.092, 0.0263, -0.181,
        0.990, 0.0282, 0.274,
    ]
    (
        true_nu1, true_a1, true_sigma1,
        true_nu2, true_a2, true_sigma2,
        true_nu3, true_a3, true_sigma3,
        true_nu12, true_a12, true_sigma12,
        true_nu13, true_a13, true_sigma13,
        true_nu23, true_a23, true_sigma23,
    ) = true_vals

    alpha_matrix = torch.tensor([
        [true_a1, true_a12, true_a13],
        [true_a12, true_a2, true_a23],
        [true_a13, true_a23, true_a3],
    ])
    nu_matrix = torch.tensor([
        [true_nu1, true_nu12, true_nu13],
        [true_nu12, true_nu2, true_nu23],
        [true_nu13, true_nu23, true_nu3],
    ])
    sigma_matrix = torch.tensor([
        [true_sigma1, true_sigma12, true_sigma13],
        [true_sigma12, true_sigma2, true_sigma23],
        [true_sigma13, true_sigma23, true_sigma3],
    ])

    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    K = symmetrize(K)
    nugget = 1e-10
    K = K + nugget * torch.eye(K.shape[0], device=K.device)
    Y = simulate_gp_data(X, K).detach()
    return Y, K
