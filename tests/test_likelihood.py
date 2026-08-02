import torch

from HighDimSpatial.metrics.likelihood import negative_log_likelihood


def test_multivariate_likelihood_counts_every_scalar_observation():
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    covariance = torch.eye(4, dtype=torch.float64)

    actual = negative_log_likelihood(y, covariance)
    expected = 0.5 * y.square().sum() + 0.5 * y.numel() * torch.log(
        torch.tensor(2 * torch.pi, dtype=torch.float64)
    )
    assert torch.allclose(actual, expected)


def test_likelihood_rejects_layout_dimension_mismatch():
    y = torch.ones((3, 2), dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64)
    try:
        negative_log_likelihood(y, covariance)
    except ValueError as exc:
        assert "does not match 6 observations" in str(exc)
    else:
        raise AssertionError("dimension mismatch should raise ValueError")
