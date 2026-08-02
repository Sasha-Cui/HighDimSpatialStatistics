import math
import torch
from scipy.special import kv

from HighDimSpatial.kernels.approx import approx_matern_kernel_cross
from HighDimSpatial.kernels.matern import (
    BesselKFunction,
    compute_matern_covariance,
    matern_cross_kernel,
    matern_kernel,
)


def test_matern_kernel_diag():
    X = torch.rand(5, 2, dtype=torch.float64)
    dists = torch.cdist(X, X)
    alpha = torch.tensor(1.0, dtype=torch.float64)
    nu = torch.tensor(1.5, dtype=torch.float64)
    sigma = torch.tensor(2.0, dtype=torch.float64)

    K = matern_kernel(dists, alpha, nu, sigma)
    assert K.shape == (5, 5)
    assert torch.allclose(torch.diag(K), torch.full((5,), sigma ** 2, dtype=torch.float64))


def test_matern_uses_ags_decay_scale_convention():
    distance = torch.tensor(2.0, dtype=torch.float64)
    alpha = torch.tensor(0.7, dtype=torch.float64)
    nu = torch.tensor(1.3, dtype=torch.float64)
    sigma = torch.tensor(1.4, dtype=torch.float64)

    expected = sigma.item() ** 2 * 2 ** (1 - nu.item()) / math.gamma(nu.item())
    expected *= (alpha.item() * distance.item()) ** nu.item() * kv(
        nu.item(), alpha.item() * distance.item()
    )
    actual = matern_kernel(distance, alpha, nu, sigma)
    assert torch.allclose(actual, torch.tensor(expected, dtype=torch.float64), rtol=1e-10, atol=1e-12)


def test_cross_covariance_preserves_sign():
    distance = torch.tensor([0.0, 1.0], dtype=torch.float64)
    covariance = torch.tensor(-0.4, dtype=torch.float64)
    values = matern_cross_kernel(
        distance,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.8, dtype=torch.float64),
        covariance,
    )
    assert values[0] == covariance
    assert values[1] < 0


def test_compute_matern_covariance_shape():
    alpha_matrix = torch.ones(2, 2, dtype=torch.float64)
    nu_matrix = torch.ones(2, 2, dtype=torch.float64)
    sigma_matrix = torch.ones(2, 2, dtype=torch.float64)
    X = torch.rand(4, 2, dtype=torch.float64)

    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    assert K.shape == (8, 8)


def test_compute_matern_covariance_is_location_major():
    alpha_matrix = torch.ones(2, 2, dtype=torch.float64)
    nu_matrix = torch.ones(2, 2, dtype=torch.float64)
    sigma_matrix = torch.tensor([[2.0, -0.25], [-0.25, 3.0]], dtype=torch.float64)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    # The first 2x2 block is the covariance between both variables at location 0.
    expected_zero_lag = torch.tensor([[4.0, -0.25], [-0.25, 9.0]], dtype=torch.float64)
    assert torch.allclose(K[:2, :2], expected_zero_lag)
    assert K[0, 2] > 0  # variable 0 across locations
    assert K[0, 3] < 0  # cross-variable covariance across locations


def test_approx_cross_covariance_preserves_sign_and_layout():
    alpha_matrix = torch.ones(2, 2, dtype=torch.float64)
    nu_matrix = torch.ones(2, 2, dtype=torch.float64)
    sigma_matrix = torch.tensor([[2.0, -0.25], [-0.25, 3.0]], dtype=torch.float64)
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)

    K = approx_matern_kernel_cross(
        alpha_matrix, nu_matrix, sigma_matrix, X, epsilon=0.0, number_of_distances=8
    )
    expected_zero_lag = torch.tensor([[4.0, -0.25], [-0.25, 9.0]], dtype=torch.float64)
    assert torch.allclose(K[:2, :2], expected_zero_lag)
    assert K[0, 2] > 0
    assert K[0, 3] < 0


def test_approx_cross_handles_single_location():
    alpha_matrix = torch.ones(2, 2, dtype=torch.float64)
    nu_matrix = torch.ones(2, 2, dtype=torch.float64)
    sigma_matrix = torch.tensor([[1.0, -0.2], [-0.2, 2.0]], dtype=torch.float64)
    K = approx_matern_kernel_cross(
        alpha_matrix,
        nu_matrix,
        sigma_matrix,
        torch.zeros((1, 2), dtype=torch.float64),
        epsilon=0.0,
    )
    assert torch.allclose(K, torch.tensor([[1.0, -0.2], [-0.2, 4.0]], dtype=torch.float64))


def test_bessel_gradients_match_finite_differences():
    order = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    argument = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    value = BesselKFunction.apply(order, argument)
    value.backward()

    h = 1e-6
    expected_order = (kv(1.2 + h, 0.9) - kv(1.2 - h, 0.9)) / (2 * h)
    expected_argument = (kv(1.2, 0.9 + h) - kv(1.2, 0.9 - h)) / (2 * h)
    assert torch.allclose(order.grad, torch.tensor(expected_order), rtol=1e-5, atol=1e-7)
    assert torch.allclose(argument.grad, torch.tensor(expected_argument), rtol=1e-5, atol=1e-7)
