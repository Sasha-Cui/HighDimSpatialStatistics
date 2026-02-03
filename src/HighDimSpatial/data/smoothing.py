"""Kernel smoothing and subsampling utilities."""
from __future__ import annotations

from typing import List, Tuple

import torch


def epanechnikov_kernel(distance: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    scaled = distance / bandwidth
    return 0.75 * (1 - scaled ** 2) * (scaled < 1).to(distance.dtype)


def kernel_smoothing(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidth: torch.Tensor,
    number_of_grids: int = 10,
    min_grid_count: int = 100,
    max_grid_count: int = 5000,
    print_logs: bool = False,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Grid-based kernel smoothing with Epanechnikov kernel."""
    if number_of_grids <= 0:
        raise ValueError("number_of_grids must be greater than 0")

    device = X.device
    dtype = X.dtype

    x_min, _ = X[:, 0].min(dim=0)
    y_min, _ = X[:, 1].min(dim=0)
    x_max, _ = X[:, 0].max(dim=0)
    y_max, _ = X[:, 1].max(dim=0)

    if print_logs:
        print(f"Device: {device}")
        print(f"x_min: {x_min:.3g}, y_min: {y_min:.3g}, x_max: {x_max:.3g}, y_max: {y_max:.3g}")

    X_groups: list[torch.Tensor] = []
    Y_groups: list[torch.Tensor] = []
    grid_scales = torch.round(torch.sqrt(torch.linspace(min_grid_count, max_grid_count, steps=number_of_grids)))

    for i, grid_scale in enumerate(grid_scales):
        if print_logs:
            print(f"Processing grid {i+1}/{len(grid_scales)} with {int(grid_scale)} points per axis")

        x_coords = torch.linspace(x_min, x_max, int(grid_scale), device=device, dtype=dtype)
        y_coords = torch.linspace(y_min, y_max, int(grid_scale), device=device, dtype=dtype)
        grid_points = torch.cartesian_prod(x_coords, y_coords).to(device)

        X_group: list[torch.Tensor] = []
        Y_group: list[torch.Tensor] = []

        for grid_point in grid_points:
            distances = torch.cdist(grid_point.unsqueeze(0), X, p=2).squeeze()
            weights = epanechnikov_kernel(distances, bandwidth=bandwidth)
            weights_sum = weights.sum()

            if weights_sum > 0:
                weights = weights / weights_sum
                Y_group_value = (Y * weights.unsqueeze(1)).sum(dim=0)
                X_group.append(grid_point)
                Y_group.append(Y_group_value)
            elif print_logs:
                print(f"Grid point {grid_point} removed due to zero weights.")

        if X_group and Y_group:
            X_groups.append(torch.stack(X_group))
            Y_groups.append(torch.stack(Y_group))

        if print_logs:
            x_shape = X_groups[-1].shape if X_groups else (0,)
            y_shape = Y_groups[-1].shape if Y_groups else (0,)
            print(f"X_group size: {x_shape}, Y_group size: {y_shape}")

    return X_groups, Y_groups
