import torch

from HighDimSpatial.fitting.marginal import optimize_marginal_parameters


def test_optimize_marginal_parameters_runs():
    X = torch.rand(10, 2, dtype=torch.float64)
    Y = torch.rand(10, 1, dtype=torch.float64)
    params = optimize_marginal_parameters(X, Y, number_of_groups=1, number_of_cycles=1, steps_per_batch=1)
    assert len(params) == 1
