import numpy as np
import pytest

from HighDimSpatial.smoothing_bias.design import (
    epanechnikov_smoothing_matrix,
    jitter_design,
    regular_grid_1d,
    regular_grid_2d,
    select_rectangular_centers,
    validate_smoothing_matrix,
)


def test_regular_grids_have_expected_coordinates():
    one_dimensional = regular_grid_1d(4, spacing=0.5, origin=-1.0)
    np.testing.assert_allclose(one_dimensional[:, 0], [-1.0, -0.5, 0.0, 0.5])

    two_dimensional = regular_grid_2d(
        (2, 3), spacing=(2.0, 0.5), origin=(-1.0, 4.0)
    )
    np.testing.assert_allclose(
        two_dimensional,
        [
            [-1.0, 4.0],
            [-1.0, 4.5],
            [-1.0, 5.0],
            [1.0, 4.0],
            [1.0, 4.5],
            [1.0, 5.0],
        ],
    )


def test_rectangular_centers_respect_stride_and_physical_trim():
    grid = regular_grid_2d((5, 6), spacing=(1.0, 2.0))
    centers = select_rectangular_centers(grid, stride=(2, 2), boundary_trim=(1.0, 2.0))
    np.testing.assert_allclose(
        centers,
        [[2.0, 4.0], [2.0, 8.0]],
    )


def test_rectangular_centers_reject_noncartesian_input():
    incomplete = regular_grid_2d((2, 2))[:-1]
    with pytest.raises(ValueError, match="full Cartesian"):
        select_rectangular_centers(incomplete)


def test_zero_bandwidth_is_exact_rectangular_selector():
    inputs = regular_grid_1d(7, spacing=0.25)
    centers = inputs[[1, 4, 6]]
    smoothing = epanechnikov_smoothing_matrix(inputs, centers, 0.0)
    expected = np.zeros((3, 7))
    expected[np.arange(3), [1, 4, 6]] = 1.0
    np.testing.assert_array_equal(smoothing, expected)


@pytest.mark.parametrize("kernel", ["radial", "product"])
def test_epanechnikov_rows_are_normalized_and_have_compact_support(kernel):
    inputs = regular_grid_2d((7, 7), spacing=0.25, origin=(-0.75, -0.75))
    centers = np.array([[0.0, 0.0], [0.25, -0.25]])
    bandwidth = np.array([0.6, 0.4])
    smoothing = epanechnikov_smoothing_matrix(
        inputs, centers, bandwidth, kernel=kernel
    )
    validate_smoothing_matrix(smoothing)
    np.testing.assert_allclose(smoothing @ np.ones(inputs.shape[0]), 1.0)

    scaled = (inputs[None, :, :] - centers[:, None, :]) / bandwidth
    if kernel == "radial":
        outside = np.sum(scaled**2, axis=2) >= 1.0
    else:
        outside = np.any(np.abs(scaled) >= 1.0, axis=2)
    assert np.all(smoothing[outside] == 0.0)


def test_product_and_radial_kernels_differ_at_square_corners():
    inputs = np.array([[0.0, 0.0], [0.75, 0.75]])
    center = np.array([[0.0, 0.0]])
    radial = epanechnikov_smoothing_matrix(inputs, center, 1.0, kernel="radial")
    product_kernel = epanechnikov_smoothing_matrix(
        inputs, center, 1.0, kernel="product"
    )
    assert radial[0, 1] == 0.0
    assert product_kernel[0, 1] > 0.0


def test_zero_bandwidth_coordinate_enforces_exact_match():
    inputs = regular_grid_2d((3, 5), spacing=(1.0, 0.5))
    center = np.array([[1.0, 1.0]])
    smoothing = epanechnikov_smoothing_matrix(
        inputs, center, bandwidth=(0.0, 0.8), kernel="product"
    )
    assert np.all(smoothing[0, inputs[:, 0] != 1.0] == 0.0)
    np.testing.assert_allclose(smoothing.sum(), 1.0)


def test_jitter_is_seeded_and_coordinatewise_bounded():
    grid = regular_grid_2d((4, 5), spacing=(1.0, 2.0))
    limits = np.array([0.2, 0.4])
    first = jitter_design(grid, limits, np.random.default_rng(713))
    second = jitter_design(grid, limits, np.random.default_rng(713))
    np.testing.assert_array_equal(first, second)
    assert np.all(np.abs(first - grid) <= limits + 1e-15)
    assert not np.array_equal(first, grid)


def test_jitter_can_preserve_original_bounding_box():
    grid = regular_grid_1d(8)
    jittered = jitter_design(
        grid, 0.45, np.random.default_rng(19), preserve_bounds=True
    )
    assert jittered.min() >= grid.min()
    assert jittered.max() <= grid.max()


def test_smoothing_matrix_fails_when_center_has_no_supported_input():
    inputs = regular_grid_1d(4)
    with pytest.raises(ValueError, match="no positive kernel support"):
        epanechnikov_smoothing_matrix(inputs, np.array([[10.0]]), 0.25)
