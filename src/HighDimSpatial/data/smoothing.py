"""Kernel smoothing and subsampling utilities."""
from __future__ import annotations

from typing import List, Tuple

import torch

from HighDimSpatial.utils.linalg import symmetrize


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
    """Grid-based kernel smoothing with Epanechnikov kernel.

    This compatibility wrapper discards the linear smoothing operators.  New
    inferential code should call :func:`kernel_smoothing_with_operators`, since
    a Gaussian covariance must be transformed by those operators.
    """
    X_groups, Y_groups, _ = kernel_smoothing_with_operators(
        X,
        Y,
        bandwidth,
        number_of_grids=number_of_grids,
        min_grid_count=min_grid_count,
        max_grid_count=max_grid_count,
        print_logs=print_logs,
    )
    return X_groups, Y_groups


def kernel_smoothing_with_operators(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidth: torch.Tensor,
    number_of_grids: int = 10,
    min_grid_count: int = 100,
    max_grid_count: int = 5000,
    print_logs: bool = False,
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Smooth observations and return each explicit location operator ``S``.

    Every returned response satisfies ``Y_group == S_group @ Y``.  Therefore,
    if ``vec(Y)`` has location-major covariance ``K``, the smoothed covariance
    is ``(S ⊗ I_p) K (S ⊗ I_p).T`` rather than a Matérn covariance
    evaluated at the grid points.
    """
    if number_of_grids <= 0:
        raise ValueError("number_of_grids must be greater than 0")
    if X.ndim != 2 or X.size(1) != 2:
        raise ValueError("X must have shape (n_locations, 2)")
    if Y.ndim != 2 or Y.size(0) != X.size(0):
        raise ValueError("Y must have shape (n_locations, n_variables)")
    if torch.any(bandwidth <= 0):
        raise ValueError("bandwidth must be positive")

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
    S_groups: list[torch.Tensor] = []
    grid_scales = torch.round(torch.sqrt(torch.linspace(min_grid_count, max_grid_count, steps=number_of_grids)))

    for i, grid_scale in enumerate(grid_scales):
        if print_logs:
            print(f"Processing grid {i+1}/{len(grid_scales)} with {int(grid_scale)} points per axis")

        x_coords = torch.linspace(x_min, x_max, int(grid_scale), device=device, dtype=dtype)
        y_coords = torch.linspace(y_min, y_max, int(grid_scale), device=device, dtype=dtype)
        grid_points = torch.cartesian_prod(x_coords, y_coords).to(device)

        X_group: list[torch.Tensor] = []
        Y_group: list[torch.Tensor] = []
        S_group: list[torch.Tensor] = []

        for grid_point in grid_points:
            distances = torch.cdist(grid_point.unsqueeze(0), X, p=2).squeeze()
            weights = epanechnikov_kernel(distances, bandwidth=bandwidth)
            weights_sum = weights.sum()

            if weights_sum > 0:
                weights = weights / weights_sum
                Y_group_value = (Y * weights.unsqueeze(1)).sum(dim=0)
                X_group.append(grid_point)
                Y_group.append(Y_group_value)
                S_group.append(weights)
            elif print_logs:
                print(f"Grid point {grid_point} removed due to zero weights.")

        if X_group and Y_group:
            X_groups.append(torch.stack(X_group))
            Y_groups.append(torch.stack(Y_group))
            S_groups.append(torch.stack(S_group))

        if print_logs:
            x_shape = X_groups[-1].shape if X_groups else (0,)
            y_shape = Y_groups[-1].shape if Y_groups else (0,)
            print(f"X_group size: {x_shape}, Y_group size: {y_shape}")

    return X_groups, Y_groups, S_groups


def lift_location_operator(S: torch.Tensor, number_of_variables: int) -> torch.Tensor:
    """Lift a location operator to location-major multivariate vectorization."""
    if S.ndim != 2:
        raise ValueError("S must be a two-dimensional location operator")
    if number_of_variables < 1:
        raise ValueError("number_of_variables must be positive")
    identity = torch.eye(number_of_variables, dtype=S.dtype, device=S.device)
    return torch.kron(S.contiguous(), identity)


def transform_location_covariance(
    covariance: torch.Tensor,
    left_operator: torch.Tensor,
    number_of_variables: int,
    right_operator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform a location-major covariance under known linear smoothing.

    With ``right_operator=None`` this returns ``A K A.T``.  Supplying a second
    operator returns the cross-covariance ``A K B.T``, which makes explicit
    that overlapping smoothing resolutions are generally dependent.
    """
    right = left_operator if right_operator is None else right_operator
    left_lifted = lift_location_operator(left_operator, number_of_variables)
    right_lifted = lift_location_operator(right, number_of_variables)
    expected = left_lifted.size(1)
    if covariance.shape != (expected, expected):
        raise ValueError(
            f"covariance shape {tuple(covariance.shape)} is incompatible with "
            f"{left_operator.size(1)} locations and {number_of_variables} variables"
        )
    if right_lifted.size(1) != expected:
        raise ValueError("left and right operators must act on the same input locations")
    transformed = left_lifted @ covariance @ right_lifted.mT
    return symmetrize(transformed) if right_operator is None else transformed
