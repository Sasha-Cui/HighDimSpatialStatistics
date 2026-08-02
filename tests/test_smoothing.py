import torch

from HighDimSpatial.data.smoothing import (
    kernel_smoothing,
    kernel_smoothing_with_operators,
    transform_location_covariance,
)


def test_kernel_smoothing_shapes():
    X = torch.rand(100, 2, dtype=torch.float64)
    Y = torch.rand(100, 3, dtype=torch.float64)
    bandwidth = torch.tensor(0.5, dtype=torch.float64)

    X_groups, Y_groups = kernel_smoothing(X, Y, bandwidth, number_of_grids=3, min_grid_count=10, max_grid_count=20)
    assert len(X_groups) == 3
    assert len(Y_groups) == 3
    for Xg, Yg in zip(X_groups, Y_groups):
        assert Xg.shape[1] == 2
        assert Yg.shape[1] == 3


def test_smoothing_operator_reproduces_responses():
    X = torch.rand(40, 2, dtype=torch.float64)
    Y = torch.rand(40, 2, dtype=torch.float64)
    X_groups, Y_groups, S_groups = kernel_smoothing_with_operators(
        X,
        Y,
        torch.tensor(0.4, dtype=torch.float64),
        number_of_grids=2,
        min_grid_count=4,
        max_grid_count=9,
    )
    assert len(S_groups) == len(X_groups) == len(Y_groups)
    for Y_group, S_group in zip(Y_groups, S_groups):
        assert torch.allclose(Y_group, S_group @ Y)
        assert torch.allclose(S_group.sum(dim=1), torch.ones(S_group.size(0), dtype=torch.float64))


def test_covariance_is_transformed_by_smoothing_operator():
    S = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]], dtype=torch.float64)
    p = 2
    K = torch.eye(3 * p, dtype=torch.float64)
    actual = transform_location_covariance(K, S, p)
    expected = torch.kron(S @ S.mT, torch.eye(p, dtype=torch.float64))
    assert torch.allclose(actual, expected)


def test_distinct_smoothing_resolutions_are_correlated():
    S_left = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
    S_right = torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float64)
    K = torch.eye(3, dtype=torch.float64)
    cross_covariance = transform_location_covariance(
        K, S_left, number_of_variables=1, right_operator=S_right
    )
    assert torch.allclose(cross_covariance, torch.tensor([[0.25]], dtype=torch.float64))
