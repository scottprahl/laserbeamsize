# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

"""Tests for functions in background.py."""

import inspect

import numpy as np
import laserbeamsize as lbs
import laserbeamsize.background as bg


# subtract_background_image
def test_basic_subtraction():
    """Test basic subtraction."""
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=float)
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)

    result = lbs.subtract_background_image(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_negative_subtraction():
    """Test negative subtraction."""
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = np.array([[10, 15, 20], [15, 20, 25]], dtype=float)
    expected = np.array([[-5, -5, -5], [-5, -5, -5]], dtype=float)

    result = lbs.subtract_background_image(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_subtraction_type_float():
    """Test subtraction type float."""
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=np.uint8)

    result = lbs.subtract_background_image(original, background)
    assert result.dtype == float


# subtract_constant
def test_basic_subtract_constant():
    """Test basic subtract constant."""
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = 5
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)

    result = lbs.subtract_constant(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_negative_subtract_constant_iso_false():
    """Test negative subtract constant iso false."""
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[0, 0, 5], [0, 5, 10]], dtype=float)

    result = lbs.subtract_constant(original, background, iso_noise=False)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_negative_subtract_constant_iso_true():
    """Test negative subtract constant iso true."""
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[-5, 0, 5], [0, 5, 10]], dtype=float)

    result = lbs.subtract_constant(original, background, iso_noise=True)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_subtract_constant_type_float():
    """Test subtract constant type float."""
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = 5

    result = lbs.subtract_constant(original, background)
    assert result.dtype == np.float64


# corner_background
def test_corner_known_mean_stdev():
    """Test corner known mean stdev."""
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    corner_mean, corner_stdev = lbs.corner_background(image, 0.25)
    # considering the corners: 1, 4, 13, 16
    expected_mean = np.mean([1, 4, 13, 16])
    expected_stdev = np.std([1, 4, 13, 16])
    assert np.isclose(corner_mean, expected_mean)
    assert np.isclose(corner_stdev, expected_stdev)


def test_corner_zero_corner_fraction():
    """Test corner zero corner fraction."""
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    corner_mean, corner_stdev = lbs.corner_background(image, 0)
    assert corner_mean == 0
    assert corner_stdev == 0


def test_corner_varying_corner_fraction():
    """Test corner varying corner fraction."""
    image = np.ones((100, 100))  # uniform image
    corner_mean, corner_stdev = lbs.corner_background(image, 0.05)
    assert corner_mean == 1
    assert corner_stdev == 0


def test_corner_uniform_image():
    """Test corner uniform image."""
    image = np.ones((100, 100))
    corner_mean, corner_stdev = lbs.corner_background(image, 0.05)
    assert corner_mean == 1
    assert corner_stdev == 0


def test_corner_image_data_types():
    """Test corner image data types."""
    image_float = np.ones((100, 100), dtype=float)
    image_int = np.ones((100, 100), dtype=int)
    corner_mean_float, corner_stdev_float = lbs.corner_background(image_float, 0.05)
    corner_mean_int, corner_stdev_int = lbs.corner_background(image_int, 0.05)
    assert corner_mean_float == corner_mean_int == 1
    assert corner_stdev_float == corner_stdev_int == 0


def test_corner_test_image():
    """Test corner test image."""
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi)
    corner_mean, corner_stdev = lbs.corner_background(image)
    assert corner_mean == 0
    assert corner_stdev == 0


def test_corner_test_image_with_noise():
    """Test corner test image with noise."""
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    noise = 20
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, noise=noise)
    corner_mean, corner_stdev = lbs.corner_background(image)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


# iso_background
def test_iso_known_mean_stdev():
    """Test iso known mean stdev."""
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    corner_mean, corner_stdev = lbs.iso_background(image, 0.25)
    # considering the corners: 1, 4, 13, 16
    expected_mean = np.mean(image)
    expected_stdev = np.std(image)
    assert np.isclose(corner_mean, expected_mean)
    assert np.isclose(corner_stdev, expected_stdev)


def test_iso_zero_corner_fraction():
    """Test iso zero corner fraction."""
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    try:
        lbs.iso_background(image, 0)
        assert False, "Expected ValueError for corner_fraction <= 0"
    except ValueError:
        pass
    try:
        lbs.iso_background(image, 0.3)
        assert False, "Expected ValueError for corner_fraction > 0.25"
    except ValueError:
        pass


def test_iso_test_noise_only_image():
    """Test iso test noise only image."""
    noise = 20
    image = np.random.poisson(noise, size=(400, 400))
    corner_mean, corner_stdev = lbs.iso_background(image)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


def test_iso_test_image_with_noise():
    """Test iso test image with noise."""
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    noise = 20
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, noise=noise)
    corner_mean, corner_stdev = lbs.iso_background(image)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


def test_iso_varying_corner_fraction():
    """Test iso varying corner fraction."""
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    noise = 20
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, noise=noise)
    corner_mean, corner_stdev = lbs.iso_background(image, 0.05)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


def test_iso_uniform_image():
    """Test iso uniform image."""
    image = np.ones((100, 100))
    corner_mean, corner_stdev = lbs.iso_background(image)
    assert corner_mean == 1
    assert corner_stdev == 0


def test_iso_image_data_types():
    """Test iso image data types."""
    image_float = np.ones((100, 100), dtype=float)
    image_int = np.ones((100, 100), dtype=int)
    corner_mean_float, corner_stdev_float = lbs.iso_background(image_float, 0.05)
    corner_mean_int, corner_stdev_int = lbs.iso_background(image_int, 0.05)
    assert corner_mean_float == corner_mean_int == 1
    assert corner_stdev_float == corner_stdev_int == 0


def test_subtract_tilted_background_no_variable_shadowing():
    """subtract_tilted_background must not name the perimeter array 'b' (shadows fit coefficient)."""
    src = inspect.getsource(bg.subtract_tilted_background)
    # The perimeter values array must not be named 'b' (shadows the plane-fit coefficient b)
    assert (
        "b = np.array(perimeter_values)" not in src
    ), "Perimeter array is still named 'b', which shadows the plane-fit coefficient"


def test_dead_code_functions_removed():
    """_mean_filter, _std_filter, and image_background2 are dead code and must be removed."""
    assert not hasattr(bg, "_mean_filter"), "_mean_filter should be removed (dead code)"
    assert not hasattr(bg, "_std_filter"), "_std_filter should be removed (dead code)"
    assert not hasattr(bg, "image_background2"), "image_background2 should be removed (dead code)"


def test_scipy_ndimage_not_imported_in_background():
    """background.py must not import scipy.ndimage after dead code removal."""
    src = inspect.getsource(bg)
    assert "import scipy.ndimage" not in src, "scipy.ndimage is imported but unused after dead-code removal"


def test_rotated_rect_mask_slow_removed():
    """rotated_rect_mask_slow is an unused duplicate and must be removed."""
    assert not hasattr(bg, "rotated_rect_mask_slow"), "rotated_rect_mask_slow should be removed (dead code)"


def test_rotated_rect_mask_docstring_no_mask_diameters():
    """rotated_rect_mask docstring must not reference a mask_diameters parameter that doesn't exist."""
    doc = bg.rotated_rect_mask.__doc__ or ""
    assert "mask_diameters" not in doc, (
        "rotated_rect_mask docstring still references 'mask_diameters', "
        "which is not a parameter of this function"
    )


def test_corner_background_docstring_lists_both_return_values():
    """corner_background docstring must document both return values: mean and stdev."""
    doc = bg.corner_background.__doc__ or ""
    assert "stdev" in doc or "std" in doc, (
        "corner_background docstring only lists corner_mean but the function returns (mean, stdev)"
    )
