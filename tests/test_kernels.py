import torch

from HighDimSpatial.kernels.matern import matern_kernel, compute_matern_covariance


def test_matern_kernel_diag():
    X = torch.rand(5, 2, dtype=torch.float64)
    dists = torch.cdist(X, X)
    alpha = torch.tensor(1.0, dtype=torch.float64)
    nu = torch.tensor(1.5, dtype=torch.float64)
    sigma = torch.tensor(2.0, dtype=torch.float64)

    K = matern_kernel(dists, alpha, nu, sigma)
    assert K.shape == (5, 5)
    assert torch.allclose(torch.diag(K), torch.full((5,), sigma ** 2))


def test_compute_matern_covariance_shape():
    alpha_matrix = torch.ones(2, 2, dtype=torch.float64)
    nu_matrix = torch.ones(2, 2, dtype=torch.float64)
    sigma_matrix = torch.ones(2, 2, dtype=torch.float64)
    X = torch.rand(4, 2, dtype=torch.float64)

    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    assert K.shape == (8, 8)
