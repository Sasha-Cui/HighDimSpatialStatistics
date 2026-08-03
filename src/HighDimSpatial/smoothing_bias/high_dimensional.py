"""High-dimensional Gaussian likelihood concentration for spatial benchmarks.

The functions in this module implement the finite-candidate theorem used by the
SupportShift benchmark.  The observation dimension is ``p`` and ``N`` denotes
the number of independent spatial-field replicates.  Spatial coordinates within
each vector may be arbitrarily dependent; that dependence is retained through
the relative precision matrix

``A = Sigma_0**(1/2) @ Sigma**(-1) @ Sigma_0**(1/2)``.

No numerical jitter is added.  All declared covariance matrices must be
symmetric positive definite.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CandidateLikelihoodBound:
    """Quadratic-form geometry and concentration radius for one candidate."""

    frobenius_norm: float
    operator_norm: float
    stable_rank: float
    radius: float


@dataclass(frozen=True)
class UniformLikelihoodBound:
    """Simultaneous per-coordinate likelihood bound on a finite candidate set."""

    dimension: int
    sample_size: int
    candidate_count: int
    delta: float
    log_factor: float
    radius: float
    candidates: tuple[CandidateLikelihoodBound, ...]


def _positive_definite_matrix(value: ArrayLike, *, name: str) -> tuple[FloatArray, FloatArray]:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) / 2.0
    try:
        factor = cholesky(matrix, lower=True, check_finite=False)
    except LinAlgError as error:
        raise ValueError(f"{name} must be positive definite; no jitter is added") from error
    return matrix, factor


def _candidate_sequence(
    candidate_covariances: ArrayLike | list[ArrayLike] | tuple[ArrayLike, ...],
) -> tuple[FloatArray, ...]:
    array = np.asarray(candidate_covariances, dtype=float)
    if array.ndim == 2:
        values = (array,)
    elif array.ndim == 3 and array.shape[0] > 0:
        values = tuple(array[index] for index in range(array.shape[0]))
    else:
        raise ValueError(
            "candidate_covariances must have shape (p, p) or (K, p, p)"
        )
    return tuple(np.asarray(value, dtype=float) for value in values)


def relative_precision_matrix(
    true_covariance: ArrayLike,
    candidate_covariance: ArrayLike,
) -> FloatArray:
    """Return a symmetric relative precision in standard-normal coordinates.

    If ``X = L_0 Z`` with ``L_0 L_0.T = Sigma_0``, then

    ``X.T @ Sigma**(-1) @ X = Z.T @ A @ Z``

    for the returned ``A``.  Its eigenvalues are the generalized eigenvalues of
    ``(Sigma_0, Sigma)`` and do not depend on the chosen square root.
    """
    truth, truth_factor = _positive_definite_matrix(
        true_covariance, name="true_covariance"
    )
    candidate, candidate_factor = _positive_definite_matrix(
        candidate_covariance, name="candidate_covariance"
    )
    if truth.shape != candidate.shape:
        raise ValueError("true_covariance and candidate_covariance must have the same shape")
    solved = cho_solve((candidate_factor, True), truth_factor, check_finite=False)
    relative = truth_factor.T @ solved
    return (relative + relative.T) / 2.0


def _relative_precision_from_factors(
    truth_factor: FloatArray,
    candidate_factor: FloatArray,
) -> FloatArray:
    """Return ``L_0.T @ Sigma**(-1) @ L_0`` from Cholesky factors."""
    solved = cho_solve(
        (candidate_factor, True),
        truth_factor,
        check_finite=False,
    )
    relative = truth_factor.T @ solved
    return (relative + relative.T) / 2.0


def gaussian_likelihood_uniform_bound(
    true_covariance: ArrayLike,
    candidate_covariances: ArrayLike | list[ArrayLike] | tuple[ArrayLike, ...],
    *,
    sample_size: int,
    delta: float = 0.05,
) -> UniformLikelihoodBound:
    r"""Return the finite-grid Gaussian likelihood concentration certificate.

    Let ``Q_hat(theta)`` be the average zero-mean Gaussian negative
    log-likelihood divided by the vector dimension ``p``, and let ``Q(theta)``
    be its expectation under ``N(0, Sigma_0)``.  For a candidate set of size
    ``K``, the returned radius ``epsilon`` satisfies

    .. math::

       P\left\{\max_\theta |\widehat Q(\theta)-Q(\theta)|
       \leq \epsilon\right\}\geq 1-\delta,

    where each candidate radius is

    .. math::

       \frac{\|A_\theta\|_F}{p}\sqrt{\frac{t}{N}}
       +\frac{\|A_\theta\|_{\mathrm{op}}}{p}\frac{t}{N},
       \qquad t=\log(2K/\delta).

    The result is a direct Gaussian quadratic-form bound plus a union bound.
    """
    if isinstance(sample_size, bool) or int(sample_size) != sample_size or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    truth, truth_factor = _positive_definite_matrix(
        true_covariance,
        name="true_covariance",
    )
    candidates = _candidate_sequence(candidate_covariances)
    if any(candidate.shape != truth.shape for candidate in candidates):
        raise ValueError("every candidate covariance must match true_covariance")
    dimension = truth.shape[0]
    log_factor = float(np.log(2.0 * len(candidates) / delta))
    bounds: list[CandidateLikelihoodBound] = []
    for index, candidate_value in enumerate(candidates):
        _, candidate_factor = _positive_definite_matrix(
            candidate_value,
            name=f"candidate_covariances[{index}]",
        )
        relative = _relative_precision_from_factors(
            truth_factor,
            candidate_factor,
        )
        eigenvalues = np.linalg.eigvalsh(relative)
        operator_norm = float(np.max(np.abs(eigenvalues)))
        frobenius_norm = float(np.linalg.norm(eigenvalues))
        stable_rank = float((frobenius_norm / operator_norm) ** 2)
        radius = float(
            frobenius_norm / dimension * np.sqrt(log_factor / sample_size)
            + operator_norm / dimension * log_factor / sample_size
        )
        bounds.append(
            CandidateLikelihoodBound(
                frobenius_norm=frobenius_norm,
                operator_norm=operator_norm,
                stable_rank=stable_rank,
                radius=radius,
            )
        )
    return UniformLikelihoodBound(
        dimension=dimension,
        sample_size=int(sample_size),
        candidate_count=len(candidates),
        delta=float(delta),
        log_factor=log_factor,
        radius=max(bound.radius for bound in bounds),
        candidates=tuple(bounds),
    )


def normalized_gaussian_population_objectives(
    true_covariance: ArrayLike,
    candidate_covariances: ArrayLike | list[ArrayLike] | tuple[ArrayLike, ...],
) -> FloatArray:
    """Return population negative log-likelihoods per coordinate, sans constant."""
    truth, _ = _positive_definite_matrix(true_covariance, name="true_covariance")
    candidates = _candidate_sequence(candidate_covariances)
    values = np.empty(len(candidates), dtype=float)
    dimension = truth.shape[0]
    for index, candidate_value in enumerate(candidates):
        candidate, factor = _positive_definite_matrix(
            candidate_value, name=f"candidate_covariances[{index}]"
        )
        if candidate.shape != truth.shape:
            raise ValueError("every candidate covariance must match true_covariance")
        log_determinant = 2.0 * np.log(np.diag(factor)).sum()
        trace_term = np.trace(cho_solve((factor, True), truth, check_finite=False))
        values[index] = 0.5 * (log_determinant + trace_term) / dimension
    return values


def normalized_gaussian_sample_objectives(
    samples: ArrayLike,
    candidate_covariances: ArrayLike | list[ArrayLike] | tuple[ArrayLike, ...],
) -> FloatArray:
    """Return average sample negative log-likelihoods per coordinate.

    ``samples`` has shape ``(N, p)``.  The Gaussian constant is omitted because
    it is common to every candidate.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("samples must have shape (N, p) with N and p positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values")
    candidates = _candidate_sequence(candidate_covariances)
    objectives = np.empty(len(candidates), dtype=float)
    dimension = values.shape[1]
    for index, candidate_value in enumerate(candidates):
        candidate, factor = _positive_definite_matrix(
            candidate_value, name=f"candidate_covariances[{index}]"
        )
        if candidate.shape != (dimension, dimension):
            raise ValueError("every candidate covariance must match the sample dimension")
        whitened = solve_triangular(factor, values.T, lower=True, check_finite=False)
        mean_quadratic = np.square(whitened).sum(axis=0).mean()
        log_determinant = 2.0 * np.log(np.diag(factor)).sum()
        objectives[index] = 0.5 * (log_determinant + mean_quadratic) / dimension
    return objectives


__all__ = [
    "CandidateLikelihoodBound",
    "UniformLikelihoodBound",
    "gaussian_likelihood_uniform_bound",
    "normalized_gaussian_population_objectives",
    "normalized_gaussian_sample_objectives",
    "relative_precision_matrix",
]
