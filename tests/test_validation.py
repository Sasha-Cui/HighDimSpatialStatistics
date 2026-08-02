import torch

from HighDimSpatial.kernels.approx import approx_matern_kernel_marginal
from HighDimSpatial.metrics.validation import (
    extract_location_major_marginal_covariance,
    validation_metric_marginal,
)


def test_validation_metric_zero_when_match():
    X = torch.rand(20, 2, dtype=torch.float64)
    alpha = torch.tensor(1.0, dtype=torch.float64)
    nu = torch.tensor(0.5, dtype=torch.float64)
    sigma = torch.tensor(0.2, dtype=torch.float64)

    K_test = approx_matern_kernel_marginal(X, alpha, nu, sigma)
    params = {"alpha": alpha, "nu": nu, "sigma": sigma}
    metric = validation_metric_marginal(params, X, K_test)
    assert metric < 1e-6


def test_extract_location_major_marginal_covariance():
    # Location-major order is (s0,v0), (s0,v1), (s1,v0), (s1,v1).
    K = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    actual = extract_location_major_marginal_covariance(K, feature=1, number_of_variables=2)
    expected = K[torch.tensor([1, 3])][:, torch.tensor([1, 3])]
    assert torch.equal(actual, expected)
