import torch

from HighDimSpatial.data.smoothing import kernel_smoothing


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
