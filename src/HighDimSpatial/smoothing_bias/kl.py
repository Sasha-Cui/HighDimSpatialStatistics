"""Finite-design Gaussian criteria for studying ignored spatial smoothing.

This module deliberately uses NumPy/SciPy rather than the differentiable Torch
kernel stack.  Its role is to compute auditable finite-design KL targets and
one-parameter Monte Carlo fits.  Throughout, ``decay`` follows the repository's
Apanasovich--Genton--Sun convention: it is an inverse range and the Matérn
argument is ``decay * distance``.

No function in this module adds numerical jitter.  A singular or indefinite
covariance therefore raises an explicit error instead of silently changing the
statistical model.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import gammaln, kve


FloatArray = NDArray[np.float64]
CovarianceBuilder = Callable[[float], ArrayLike]


@dataclass(frozen=True)
class LogDecayFit:
    """Result and optimizer diagnostics for a bounded log-decay fit.

    ``objective`` is the population Gaussian criterion or sample negative
    log-likelihood at the reported optimum.  The covariance builder supplied to
    a fitting function receives ``decay`` (not ``log_decay``).
    """

    decay: float
    log_decay: float
    objective: float
    success: bool
    status: int
    message: str
    nfev: int
    nit: int
    log_decay_bounds: tuple[float, float]
    at_lower_bound: bool
    at_upper_bound: bool


def _as_locations(locations: ArrayLike, *, name: str = "locations") -> FloatArray:
    result = np.asarray(locations, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n,) or (n, d) with n positive")
    if result.shape[1] not in (1, 2):
        raise ValueError(f"{name} must contain one- or two-dimensional coordinates")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _as_smoothing_operator(operator: ArrayLike, n_locations: int) -> FloatArray:
    result = np.asarray(operator, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("smoothing_operator must be a nonempty two-dimensional matrix")
    if result.shape[1] != n_locations:
        raise ValueError(
            "smoothing_operator must have one column per latent location; "
            f"received {result.shape[1]} columns for {n_locations} locations"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("smoothing_operator must contain only finite values")
    if np.any(np.linalg.norm(result, axis=1) == 0.0):
        raise ValueError("smoothing_operator cannot contain an all-zero row")
    return result


def matern_correlation(
    pairwise_distances: ArrayLike,
    decay: float,
    nu: float,
) -> FloatArray | float:
    r"""Evaluate a Matérn correlation under the AGS decay convention.

    The formula is

    .. math::
       2^{1-\nu}\Gamma(\nu)^{-1}(a r)^\nu K_\nu(a r),

    where ``a`` is ``decay``.  Thus larger decay means shorter range.
    """

    distances = np.asarray(pairwise_distances, dtype=float)
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("pairwise_distances must be finite and nonnegative")
    decay = _positive_scalar(decay, name="decay")
    nu = _positive_scalar(nu, name="nu")

    scaled = decay * distances
    result = np.ones_like(scaled, dtype=float)
    positive = scaled > 0.0
    x = scaled[positive]
    if x.size:
        if nu == 0.5:
            correlation = np.exp(-x)
        else:
            # kve(nu, x) = exp(x) K_nu(x); the log representation avoids
            # the indeterminate x**nu * K_nu(x) at very large arguments.
            # Only replace values so small that the leading limit is at
            # floating-point accuracy.  A looser cutoff is inaccurate when
            # nu is close to zero because convergence to one is then slow.
            tiny = x < 1e-100
            correlation = np.ones_like(x)
            regular = ~tiny
            xr = x[regular]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                log_correlation = (
                    (1.0 - nu) * np.log(2.0)
                    - gammaln(nu)
                    + nu * np.log(xr)
                    + np.log(kve(nu, xr))
                    - xr
                )
                correlation[regular] = np.exp(log_correlation)
        if not np.all(np.isfinite(correlation)):
            raise ValueError("Matérn correlation evaluation produced a nonfinite value")
        # Roundoff near zero can put the result a few ulps above one.
        result[positive] = np.clip(correlation, 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def matern_covariance(
    locations: ArrayLike,
    variance: float,
    decay: float,
    nu: float,
    nugget: float = 0.0,
) -> FloatArray:
    """Construct a univariate Matérn covariance at 1D or 2D locations.

    ``variance`` is the process variance.  ``nugget`` is an independent-noise
    variance and is added to the diagonal.
    """

    coordinates = _as_locations(locations)
    variance = _positive_scalar(variance, name="variance")
    decay = _positive_scalar(decay, name="decay")
    nu = _positive_scalar(nu, name="nu")
    nugget = _nonnegative_scalar(nugget, name="nugget")
    distances = cdist(coordinates, coordinates, metric="euclidean")
    covariance = variance * np.asarray(matern_correlation(distances, decay, nu))
    if nugget:
        covariance = covariance.copy()
        covariance.flat[:: covariance.shape[0] + 1] += nugget
    return (covariance + covariance.T) / 2.0


def exact_smoothed_covariance(
    locations: ArrayLike,
    smoothing_operator: ArrayLike,
    variance: float,
    decay: float,
    nu: float,
    nugget: float = 0.0,
) -> FloatArray:
    """Covariance under the known rectangular observation operator ``S``.

    If the latent point observations have covariance ``K + nugget * I``, this
    returns ``S @ (K + nugget * I) @ S.T``.  In particular, independent noise
    is smoothed along with the process.
    """

    coordinates = _as_locations(locations)
    operator = _as_smoothing_operator(smoothing_operator, coordinates.shape[0])
    latent_covariance = matern_covariance(coordinates, variance, decay, nu, nugget)
    result = operator @ latent_covariance @ operator.T
    return (result + result.T) / 2.0


def naive_point_covariance(
    locations: ArrayLike,
    smoothing_operator: ArrayLike,
    variance: float,
    decay: float,
    nu: float,
    nugget: float = 0.0,
    *,
    output_locations: ArrayLike | None = None,
) -> FloatArray:
    """Point-level covariance that ignores the smoothing observation model.

    The rectangular operator is still required so that the number of outputs is
    explicit.  By default, each output coordinate is the weighted barycenter of
    a row of ``smoothing_operator``.  Supplying ``output_locations`` overrides
    those barycenters.  Unlike :func:`exact_smoothed_covariance`, ``nugget`` is
    added *after* smoothing as ``nugget * I``; this is the usual naive model.
    """

    coordinates = _as_locations(locations)
    operator = _as_smoothing_operator(smoothing_operator, coordinates.shape[0])
    n_outputs = operator.shape[0]
    if output_locations is None:
        row_sums = operator.sum(axis=1)
        if np.any(np.isclose(row_sums, 0.0, rtol=0.0, atol=1e-14)):
            raise ValueError(
                "each smoothing-operator row must have nonzero sum when output_locations "
                "are inferred"
            )
        effective_locations = (operator @ coordinates) / row_sums[:, None]
    else:
        effective_locations = _as_locations(output_locations, name="output_locations")
        if effective_locations.shape != (n_outputs, coordinates.shape[1]):
            raise ValueError(
                "output_locations must have one row per smoothing output and the same "
                "coordinate dimension as locations"
            )
    return matern_covariance(effective_locations, variance, decay, nu, nugget)


def _as_square_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return result


def _covariance_cholesky(covariance: ArrayLike, *, name: str) -> tuple[FloatArray, FloatArray]:
    matrix = _as_square_matrix(covariance, name=name)
    try:
        factor = cholesky(matrix, lower=True, check_finite=False)
    except LinAlgError as error:
        raise ValueError(
            f"{name} must be positive definite; no numerical jitter is added"
        ) from error
    return matrix, factor


def gaussian_population_criterion(
    candidate_covariance: ArrayLike,
    true_covariance: ArrayLike,
) -> float:
    """Expected zero-mean Gaussian negative log-likelihood for one vector.

    The returned value includes the ``n log(2*pi)`` constant.  Both covariance
    matrices must be symmetric positive definite.
    """

    candidate, factor = _covariance_cholesky(
        candidate_covariance, name="candidate_covariance"
    )
    truth, _ = _covariance_cholesky(true_covariance, name="true_covariance")
    if candidate.shape != truth.shape:
        raise ValueError("candidate_covariance and true_covariance must have the same shape")
    log_determinant = 2.0 * np.log(np.diag(factor)).sum()
    trace_term = np.trace(cho_solve((factor, True), truth, check_finite=False))
    dimension = candidate.shape[0]
    return float(
        0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + trace_term)
    )


def gaussian_kl_divergence(
    true_covariance: ArrayLike,
    candidate_covariance: ArrayLike,
) -> float:
    """KL divergence from ``N(0, true_covariance)`` to the candidate model."""

    candidate, candidate_factor = _covariance_cholesky(
        candidate_covariance, name="candidate_covariance"
    )
    truth, truth_factor = _covariance_cholesky(true_covariance, name="true_covariance")
    if candidate.shape != truth.shape:
        raise ValueError("candidate_covariance and true_covariance must have the same shape")
    trace_term = np.trace(cho_solve((candidate_factor, True), truth, check_finite=False))
    candidate_log_determinant = 2.0 * np.log(np.diag(candidate_factor)).sum()
    truth_log_determinant = 2.0 * np.log(np.diag(truth_factor)).sum()
    divergence = 0.5 * (
        trace_term - candidate.shape[0] + candidate_log_determinant - truth_log_determinant
    )
    # The exact value is nonnegative; suppress only a roundoff-sized negative.
    if divergence < 0.0 and divergence >= -1e-10:
        divergence = 0.0
    return float(divergence)


def gaussian_sample_nll(
    samples: ArrayLike,
    covariance: ArrayLike,
    *,
    average: bool = False,
) -> float:
    """Zero-mean Gaussian negative log-likelihood for one or more samples.

    A one-dimensional input is one observation vector.  A two-dimensional input
    has independent observation vectors in its rows.  Set ``average=True`` to
    return the mean per vector instead of the sum.
    """

    matrix, factor = _covariance_cholesky(covariance, name="covariance")
    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != matrix.shape[0]:
        raise ValueError(
            "samples must have shape (dimension,) or (n_samples, dimension) matching covariance"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values")
    whitened = solve_triangular(factor, values.T, lower=True, check_finite=False)
    quadratic = float(np.square(whitened).sum())
    log_determinant = 2.0 * np.log(np.diag(factor)).sum()
    dimension = matrix.shape[0]
    n_samples = values.shape[0]
    objective = 0.5 * (
        n_samples * (dimension * np.log(2.0 * np.pi) + log_determinant) + quadratic
    )
    if average:
        objective /= n_samples
    return float(objective)


def _validate_fit_inputs(
    covariance_builder: CovarianceBuilder,
    log_decay_bounds: tuple[float, float],
    xatol: float,
    maxiter: int,
) -> tuple[float, float, float, int]:
    if not callable(covariance_builder):
        raise TypeError("covariance_builder must be callable")
    if len(log_decay_bounds) != 2:
        raise ValueError("log_decay_bounds must contain exactly two values")
    lower, upper = (float(value) for value in log_decay_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("log_decay_bounds must be finite and strictly increasing")
    xatol = _positive_scalar(xatol, name="xatol")
    if isinstance(maxiter, bool) or int(maxiter) != maxiter or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    return lower, upper, xatol, int(maxiter)


def _fit_log_decay(
    objective_at_decay: Callable[[float], float],
    log_decay_bounds: tuple[float, float],
    *,
    xatol: float,
    maxiter: int,
) -> LogDecayFit:
    lower, upper = log_decay_bounds

    def objective(log_decay: float) -> float:
        value = float(objective_at_decay(float(np.exp(log_decay))))
        if not np.isfinite(value):
            raise ValueError("the decay objective returned a nonfinite value")
        return value

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    log_decay = float(result.x)
    boundary_tolerance = max(10.0 * xatol, 1e-7 * (upper - lower))
    return LogDecayFit(
        decay=float(np.exp(log_decay)),
        log_decay=log_decay,
        objective=float(result.fun),
        success=bool(result.success),
        status=int(getattr(result, "status", 0 if result.success else 1)),
        message=str(result.message),
        nfev=int(result.nfev),
        nit=int(getattr(result, "nit", -1)),
        log_decay_bounds=(lower, upper),
        at_lower_bound=log_decay - lower <= boundary_tolerance,
        at_upper_bound=upper - log_decay <= boundary_tolerance,
    )


def fit_population_log_decay(
    true_covariance: ArrayLike,
    covariance_builder: CovarianceBuilder,
    log_decay_bounds: tuple[float, float],
    *,
    xatol: float = 1e-8,
    maxiter: int = 500,
) -> LogDecayFit:
    """Fit decay by minimizing the finite-design population criterion.

    Optimization is bounded in log-decay, while ``covariance_builder`` receives
    the positive decay on its original scale.
    """

    truth, _ = _covariance_cholesky(true_covariance, name="true_covariance")
    lower, upper, xatol, maxiter = _validate_fit_inputs(
        covariance_builder, log_decay_bounds, xatol, maxiter
    )
    return _fit_log_decay(
        lambda decay: gaussian_population_criterion(covariance_builder(decay), truth),
        (lower, upper),
        xatol=xatol,
        maxiter=maxiter,
    )


def fit_sample_log_decay(
    samples: ArrayLike,
    covariance_builder: CovarianceBuilder,
    log_decay_bounds: tuple[float, float],
    *,
    xatol: float = 1e-8,
    maxiter: int = 500,
) -> LogDecayFit:
    """Fit decay by minimizing sample Gaussian NLL over bounded log-decay."""

    values = np.asarray(samples, dtype=float)
    if values.ndim not in (1, 2) or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a nonempty finite vector or matrix")
    lower, upper, xatol, maxiter = _validate_fit_inputs(
        covariance_builder, log_decay_bounds, xatol, maxiter
    )
    return _fit_log_decay(
        lambda decay: gaussian_sample_nll(
            values, covariance_builder(decay), average=values.ndim == 2
        ),
        (lower, upper),
        xatol=xatol,
        maxiter=maxiter,
    )


__all__ = [
    "LogDecayFit",
    "exact_smoothed_covariance",
    "fit_population_log_decay",
    "fit_sample_log_decay",
    "gaussian_kl_divergence",
    "gaussian_population_criterion",
    "gaussian_sample_nll",
    "matern_correlation",
    "matern_covariance",
    "naive_point_covariance",
]
