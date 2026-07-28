# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

"""Tests for functions in background.py."""

import inspect
from typing import Any

import numpy as np
import pytest
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


def test_subtract_background_image_rejects_non_arrays():
    """Both inputs must be NumPy arrays."""
    invalid_original: Any = [[1, 2]]
    valid_background = np.array([[1, 2]])
    with pytest.raises(TypeError, match="must be numpy arrays"):
        lbs.subtract_background_image(invalid_original, valid_background)
    with pytest.raises(TypeError, match="must be numpy arrays"):
        lbs.subtract_background_image(valid_background, invalid_original)


def test_subtract_background_image_rejects_non_2d_arrays():
    """Both inputs must be two-dimensional."""
    one_dimensional = np.array([1, 2])
    two_dimensional = np.array([[1, 2]])
    with pytest.raises(ValueError, match="must be two-dimensional"):
        lbs.subtract_background_image(one_dimensional, two_dimensional)


def test_subtract_background_image_rejects_unequal_shapes():
    """Inputs must have identical shapes."""
    with pytest.raises(ValueError, match="must have equal shapes"):
        lbs.subtract_background_image(np.ones((2, 2)), np.ones((3, 2)))


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


def test_corner_mask_uses_unmasked_bounding_box():
    """Masked images use the unmasked region when locating corners."""
    image = np.ma.masked_all((6, 7))
    image[1:5, 2:6] = 0

    mask = lbs.corner_mask(image, corner_fraction=0.5)

    expected = np.zeros(image.shape, dtype=bool)
    expected[1:3, 2:4] = True
    expected[1:3, 4:6] = True
    expected[3:5, 2:4] = True
    expected[3:5, 4:6] = True
    assert np.array_equal(mask, expected)


def test_corner_mask_fully_masked_image_is_empty():
    """A fully masked image has no corner pixels."""
    image = np.ma.masked_all((4, 5))
    assert not np.any(lbs.corner_mask(image, corner_fraction=0.25))


def test_perimeter_mask_fully_masked_image_is_empty():
    """A fully masked image has no perimeter pixels."""
    image = np.ma.masked_all((4, 5))
    assert not np.any(lbs.perimeter_mask(image, corner_fraction=0.25))


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


def test_iso_background_mask_excludes_bright_and_masked_pixels(monkeypatch):
    """The ISO mask selects only unmasked pixels below the threshold."""
    image = np.ma.array([[1.0, 4.0], [2.0, 3.0]], mask=[[False, False], [True, False]])
    monkeypatch.setattr(bg, "corner_background", lambda *_args, **_kwargs: (2.0, 0.5))

    mask = lbs.iso_background_mask(image, nT=2)

    assert np.array_equal(np.ma.filled(mask, False), np.array([[True, False], [False, False]]))


def test_iso_background_raises_when_threshold_selects_no_pixels(monkeypatch):
    """A threshold below every image value produces a clear error."""
    image = np.zeros((2, 2))
    monkeypatch.setattr(bg, "corner_background", lambda *_args, **_kwargs: (-1.0, 0.0))

    with pytest.raises(ValueError, match="No values in image"):
        lbs.iso_background(image)


def test_subtract_iso_background_can_zero_noise(monkeypatch):
    """Disabling ISO noise clips values below the noise threshold."""
    image = np.array([[1.0, 4.0]])
    monkeypatch.setattr(bg, "iso_background", lambda *_args, **_kwargs: (2.0, 1.0))

    result = lbs.subtract_iso_background(image, nT=1, iso_noise=False)

    assert np.array_equal(result, np.array([[0.0, 2.0]]))


def test_subtract_corner_background_can_zero_noise(monkeypatch):
    """Corner subtraction clips values below the requested threshold."""
    image = np.array([[1.0, 4.0]])
    monkeypatch.setattr(bg, "corner_background", lambda *_args, **_kwargs: (2.0, 1.0))

    result = lbs.subtract_corner_background(image, nT=1, iso_noise=False)

    assert np.array_equal(result, np.array([[0.0, 2.0]]))


def test_subtract_tilted_background_removes_fitted_plane():
    """A fitted planar background leaves the perimeter standard deviation."""
    yy, xx = np.mgrid[:10, :12]
    image = 2 * yy + 3 * xx + 5
    perimeter = lbs.perimeter_mask(image, corner_fraction=0.2)
    expected = np.std(image[perimeter])

    result = lbs.subtract_tilted_background(image, corner_fraction=0.2)

    assert np.allclose(result, expected)


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
        "rotated_rect_mask docstring still references 'mask_diameters', " "which is not a parameter of this function"
    )


def test_corner_background_docstring_lists_both_return_values():
    """corner_background Returns section must list both corner_mean and corner_stdev."""
    doc = bg.corner_background.__doc__ or ""
    returns_block = doc.rsplit("Returns:", maxsplit=1)[-1]
    assert "corner_stdev" in returns_block, (
        "corner_background Returns section does not list corner_stdev; "
        "the function returns (mean, stdev) but only corner_mean is documented"
    )


def test_corner_background_zero_fraction_documented():
    """corner_background docstring must mention the corner_fraction=0 special-case behaviour."""
    doc = bg.corner_background.__doc__ or ""
    assert (
        "corner_fraction=0" in doc or "corner_fraction == 0" in doc
    ), "corner_background docstring does not document the corner_fraction=0 special case"


def test_elliptical_mask_raises_on_zero_diameter():
    """elliptical_mask must raise ValueError when d_major or d_minor is zero."""
    image = np.zeros((50, 50))
    with pytest.raises(ValueError):
        lbs.elliptical_mask(image, 25, 25, 0, 10, 0)
    with pytest.raises(ValueError):
        lbs.elliptical_mask(image, 25, 25, 10, 0, 0)
