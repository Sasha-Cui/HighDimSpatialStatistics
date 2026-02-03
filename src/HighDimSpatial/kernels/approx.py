"""Approximate Matérn kernels and PSD adjustments."""
from __future__ import annotations

import torch

from HighDimSpatial.kernels.matern import matern_kernel
from HighDimSpatial.utils.linalg import symmetrize


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
    pairwise_distances_condensed = torch.pdist(X, p=2).detach()
    max_dist = pairwise_distances_condensed.max().detach()
    min_dist = pairwise_distances_condensed.min().detach()

    distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=X.device)
    kernel_dict = torch.empty(number_of_distances, dtype=torch.float64, device=X.device)
    for i, dist in enumerate(distances_grid):
        kernel_dict[i] = matern_kernel(dist, alpha_i, nu_i, sigma_i)

    normalized_distances = ((pairwise_distances_condensed - min_dist) / (max_dist - min_dist)).detach()
    indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()

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
    """Approximate cross-covariance Matérn kernel (correct parameter order)."""
    device = X.device
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    pairwise_distances_condensed = torch.pdist(X, p=2).detach()
    max_dist = pairwise_distances_condensed.max().detach()
    min_dist = pairwise_distances_condensed.min().detach()

    distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=device)
    kernel_dict = torch.empty((p, p, number_of_distances), dtype=torch.float64, device=device)

    for i in range(p):
        for j in range(p):
            for k, dist in enumerate(distances_grid):
                kernel_dict[i, j, k] = matern_kernel(dist, alpha_matrix[i, j], nu_matrix[i, j], sigma_matrix[i, j])

    normalized_distances = ((pairwise_distances_condensed - min_dist) / (max_dist - min_dist)).detach()
    indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()

    K_blocks = torch.zeros((p, p, n_locations, n_locations), dtype=torch.float64, device=device)
    triu_indices = torch.triu_indices(n_locations, n_locations, 1, device=device)

    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        index = indices[idx].item()
        if index == -1:
            K_blocks[:, :, i, j] = sigma_matrix ** 2
        else:
            K_blocks[:, :, i, j] = kernel_dict[:, :, index]

    K_blocks += K_blocks.transpose(2, 3).clone()
    for i in range(n_locations):
        K_blocks[:, :, i, i] = sigma_matrix ** 2

    K_blocks[:, :, range(n_locations), range(n_locations)] += epsilon
    K_approx = K_blocks.permute(0, 2, 1, 3).reshape(p * n_locations, p * n_locations)
    return symmetrize(K_approx)


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

    pairwise_distances_condensed = torch.pdist(X, p=2).detach()
    max_dist = pairwise_distances_condensed.max().detach()
    min_dist = pairwise_distances_condensed.min().detach()

    distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=device)
    kernel_dict = torch.empty((p, p, number_of_distances), dtype=torch.float64, device=device)

    for i in range(p):
        for j in range(p):
            for k, dist in enumerate(distances_grid):
                kernel_dict[i, j, k] = matern_kernel(dist, nu_matrix[i, j], alpha_matrix[i, j], sigma_matrix[i, j])

    normalized_distances = ((pairwise_distances_condensed - min_dist) / (max_dist - min_dist)).detach()
    indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()

    K_blocks = torch.zeros((p, p, n_locations, n_locations), dtype=torch.float64, device=device)
    triu_indices = torch.triu_indices(n_locations, n_locations, 1, device=device)

    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        index = indices[idx].item()
        if index == -1:
            K_blocks[:, :, i, j] = sigma_matrix ** 2
        else:
            K_blocks[:, :, i, j] = kernel_dict[:, :, index]

    K_blocks += K_blocks.transpose(2, 3).clone()
    for i in range(n_locations):
        K_blocks[:, :, i, i] = sigma_matrix ** 2

    K_blocks[:, :, range(n_locations), range(n_locations)] += epsilon
    K_approx = K_blocks.permute(0, 2, 1, 3).reshape(p * n_locations, p * n_locations)
    return symmetrize(K_approx)
