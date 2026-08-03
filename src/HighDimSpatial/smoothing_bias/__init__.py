"""Theory and estimators for covariance inference after spatial smoothing."""

from HighDimSpatial.smoothing_bias.continuous import (
    continuous_matern_pair_target,
    epanechnikov_difference_radial_moment,
)
from HighDimSpatial.smoothing_bias.design import (
    epanechnikov_smoothing_matrix,
    regular_grid_1d,
    regular_grid_2d,
    select_rectangular_centers,
)
from HighDimSpatial.smoothing_bias.estimators import (
    corrected_two_lag_estimate,
    naive_pair_estimate,
)
from HighDimSpatial.smoothing_bias.kl import (
    exact_smoothed_covariance,
    fit_population_log_decay,
    fit_sample_log_decay,
    matern_covariance,
    naive_point_covariance,
)
from HighDimSpatial.smoothing_bias.theory import (
    epanechnikov_far_lag_factor,
    epanechnikov_variance_factor,
    naive_exponential_pseudo_target,
    naive_separable_axis_pseudo_target,
    smoothed_exponential_covariance,
)

__all__ = [
    "corrected_two_lag_estimate",
    "continuous_matern_pair_target",
    "epanechnikov_difference_radial_moment",
    "epanechnikov_far_lag_factor",
    "epanechnikov_smoothing_matrix",
    "epanechnikov_variance_factor",
    "exact_smoothed_covariance",
    "fit_population_log_decay",
    "fit_sample_log_decay",
    "matern_covariance",
    "naive_exponential_pseudo_target",
    "naive_pair_estimate",
    "naive_point_covariance",
    "naive_separable_axis_pseudo_target",
    "regular_grid_1d",
    "regular_grid_2d",
    "select_rectangular_centers",
    "smoothed_exponential_covariance",
]
