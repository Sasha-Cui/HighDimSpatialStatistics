"""Approximate Matérn kernels and PSD adjustments."""
from __future__ import annotations

import torch

from HighDimSpatial.kernels.matern import matern_cross_kernel, matern_kernel
from HighDimSpatial.utils.linalg import symmetrize


def _distance_grid_and_indices(
    X: torch.Tensor,
    number_of_distances: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a quantization grid and indices, including equal-distance cases."""
    if number_of_distances < 1:
        raise ValueError("number_of_distances must be positive")
    distances = torch.pdist(X, p=2).detach()
    if distances.numel() == 0:
        return torch.empty(0, dtype=torch.float64, device=X.device), torch.empty(
            0, dtype=torch.long, device=X.device
        )

    min_dist = distances.min()
    max_dist = distances.max()
    grid = torch.linspace(
        min_dist, max_dist, number_of_distances, dtype=torch.float64, device=X.device
    )
    if torch.isclose(max_dist, min_dist):
        indices = torch.zeros_like(distances, dtype=torch.long)
    else:
        normalized = (distances - min_dist) / (max_dist - min_dist)
        indices = (normalized * (number_of_distances - 1)).round().long()
    return grid, indices


def adjust_matrix_with_nugget(K: torch.Tensor, nugget: float) -> torch.Tensor:
    """Ensure PSD by shifting negative eigenvalues and adding a nugget."""
    eigenvalues = torch.linalg.eigvalsh(K)
    smallest = eigenvalues.min().item()
    if smallest < 0:
        K = K + (-smallest) * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    K = K + nugget * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    return K


def approx_matern_kernel_marginal(
    X: torch.Tensor,
    alpha_i: torch.Tensor,
    nu_i: torch.Tensor,
    sigma_i: torch.Tensor,
    epsilon: float = 1e-3,
    number_of_distances: int = 500,
) -> torch.Tensor:
    """Approximate marginal Matérn kernel using precomputed distances."""
    if X.size(0) == 1:
        return (sigma_i.square() + epsilon).reshape(1, 1)

    distances_grid, indices = _distance_grid_and_indices(X, number_of_distances)
    kernel_dict = torch.empty(number_of_distances, dtype=torch.float64, device=X.device)
    for i, dist in enumerate(distances_grid):
        kernel_dict[i] = matern_kernel(dist, alpha_i, nu_i, sigma_i)

    n = X.size(0)
    K = torch.zeros((n, n), dtype=torch.float64, device=X.device)
    triu_indices = torch.triu_indices(n, n, offset=1, device=X.device)
    K[triu_indices[0], triu_indices[1]] = kernel_dict[indices]
    K = K + K.mT
    K += torch.diag(sigma_i ** 2 + torch.full((n,), 0 * epsilon, dtype=K.dtype, device=K.device))
    K = adjust_matrix_with_nugget(symmetrize(K), epsilon)
    return K


def approx_matern_kernel_marginal_old(
    X: torch.Tensor,
    alpha_i: torch.Tensor,
    nu_i: torch.Tensor,
    sigma_i: torch.Tensor,
    epsilon: float = 1e-9,
    number_of_distances: int = 500,
) -> torch.Tensor:
    """Legacy marginal approximation using full cdist."""
    pairwise_distances = torch.cdist(X, X).detach()
    pairwise_distances_no_diag = pairwise_distances[~torch.eye(pairwise_distances.size(0), dtype=bool, device=X.device)]

    max_dist = torch.max(pairwise_distances_no_diag).detach()
    min_dist = torch.min(pairwise_distances_no_diag).detach()
    distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=X.device)

    kernel_dict = torch.empty(number_of_distances, dtype=torch.float64, device=X.device)
    for i, dist in enumerate(distances_grid):
        kernel_dict[i] = matern_kernel(dist, alpha_i, nu_i, sigma_i)

    normalized_distances = ((pairwise_distances - min_dist) / (max_dist - min_dist)).detach()
    indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()
    indices = torch.where(indices < 0, torch.tensor(-1, device=X.device), indices).detach()
    diag_indices = torch.arange(pairwise_distances.size(0), device=X.device)
    indices[diag_indices, diag_indices] = -1

    K = torch.empty_like(pairwise_distances, device=X.device)
    for i in range(pairwise_distances.size(0)):
        for j in range(pairwise_distances.size(1)):
            index = indices[i, j].item()
            if index == -1:
                K[i, j] = sigma_i ** 2
            else:
                K[i, j] = kernel_dict[index]
    K += torch.eye(K.size(0), device=K.device) * epsilon
    return symmetrize(K)


def approx_matern_kernel_cross(
    alpha_matrix: torch.Tensor,
    nu_matrix: torch.Tensor,
    sigma_matrix: torch.Tensor,
    X: torch.Tensor,
    epsilon: float = 1e-9,
    number_of_distances: int = 500,
) -> torch.Tensor:
    """Approximate a signed, location-major cross-covariance Matérn kernel.

    As in :func:`compute_matern_covariance`, ``sigma_matrix`` contains marginal
    standard deviations on its diagonal and signed zero-lag covariances off
    diagonal.  Distance quantization itself is retained for legacy
    reproducibility; it is not guaranteed to preserve positive definiteness.
    """
    device = X.device
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    zero_lag_covariance = sigma_matrix.clone()
    diagonal = torch.arange(p, device=device)
    zero_lag_covariance[diagonal, diagonal] = sigma_matrix.diagonal().square()

    if n_locations == 1:
        return symmetrize(zero_lag_covariance) + epsilon * torch.eye(
            p, dtype=zero_lag_covariance.dtype, device=device
        )

    distances_grid, indices = _distance_grid_and_indices(X, number_of_distances)
    kernel_dict = torch.empty((p, p, number_of_distances), dtype=torch.float64, device=device)

    for i in range(p):
        for j in range(p):
            for k, dist in enumerate(distances_grid):
                kernel_dict[i, j, k] = matern_cross_kernel(
                    dist, alpha_matrix[i, j], nu_matrix[i, j], zero_lag_covariance[i, j]
                )

    K_blocks = torch.zeros((p, p, n_locations, n_locations), dtype=torch.float64, device=device)
    triu_indices = torch.triu_indices(n_locations, n_locations, 1, device=device)

    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        index = indices[idx].item()
        K_blocks[:, :, i, j] = kernel_dict[:, :, index]

    K_blocks += K_blocks.transpose(2, 3).clone()
    for i in range(n_locations):
        K_blocks[:, :, i, i] = zero_lag_covariance

    K_approx = K_blocks.permute(2, 0, 3, 1).reshape(p * n_locations, p * n_locations)
    K_approx = symmetrize(K_approx)
    return K_approx + epsilon * torch.eye(p * n_locations, dtype=K_approx.dtype, device=device)


def approx_matern_kernel_cross_legacy(
    alpha_matrix: torch.Tensor,
    nu_matrix: torch.Tensor,
    sigma_matrix: torch.Tensor,
    X: torch.Tensor,
    epsilon: float = 1e-9,
    number_of_distances: int = 500,
) -> torch.Tensor:
    """Legacy cross-kernel using swapped parameter order (nu/alpha).

    This matches historical notebook behavior.
    """
    device = X.device
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    zero_lag_covariance = sigma_matrix.clone()
    diagonal = torch.arange(p, device=device)
    zero_lag_covariance[diagonal, diagonal] = sigma_matrix.diagonal().square()

    if n_locations == 1:
        return symmetrize(zero_lag_covariance) + epsilon * torch.eye(
            p, dtype=zero_lag_covariance.dtype, device=device
        )

    distances_grid, indices = _distance_grid_and_indices(X, number_of_distances)
    kernel_dict = torch.empty((p, p, number_of_distances), dtype=torch.float64, device=device)

    for i in range(p):
        for j in range(p):
            for k, dist in enumerate(distances_grid):
                kernel_dict[i, j, k] = matern_cross_kernel(
                    dist, nu_matrix[i, j], alpha_matrix[i, j], zero_lag_covariance[i, j]
                )

    K_blocks = torch.zeros((p, p, n_locations, n_locations), dtype=torch.float64, device=device)
    triu_indices = torch.triu_indices(n_locations, n_locations, 1, device=device)

    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        index = indices[idx].item()
        K_blocks[:, :, i, j] = kernel_dict[:, :, index]

    K_blocks += K_blocks.transpose(2, 3).clone()
    for i in range(n_locations):
        K_blocks[:, :, i, i] = zero_lag_covariance

    K_approx = K_blocks.permute(2, 0, 3, 1).reshape(p * n_locations, p * n_locations)
    K_approx = symmetrize(K_approx)
    return K_approx + epsilon * torch.eye(p * n_locations, dtype=K_approx.dtype, device=device)
