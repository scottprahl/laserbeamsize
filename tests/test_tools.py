"""Tests for functions in image_tools.py."""

import numpy as np
import pytest

from laserbeamsize.image_tools import (
    rotate_image,
    rotate_points,
    values_along_line,
    major_axis_arrays,
    minor_axis_arrays,
    axes_arrays,
    create_test_image,
    crop_image_to_integration_rect,
)


# rotate_points
def test_rotate_points_0_degrees():
    """Test rotate points 0 degrees."""
    x, y = rotate_points(1, 0, 0, 0, 0)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)


def test_rotate_points_90_degrees():
    """Test rotate points 90 degrees."""
    x, y = rotate_points(1, 0, 0, 0, np.pi / 2)
    assert np.isclose(x, 0, atol=1e-8)
    assert np.isclose(y, -1, atol=1e-8)


def test_rotate_points_180_degrees():
    """Test rotate points 180 degrees."""
    x, y = rotate_points(1, 0, 0, 0, np.pi)
    assert np.isclose(x, -1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)


def test_rotate_points_360_degrees():
    """Test rotate points 360 degrees."""
    x, y = rotate_points(1, 0, 0, 0, 2 * np.pi)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)


# values_along_line
def test_values_along_line():
    """Test values along line."""
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = values_along_line(image, 0, 0, 1, 1)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678, 0.70710678]))


def test_values_along_line_vertical():
    """Test values along line vertical."""
    image = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    x, y, z, s = values_along_line(image, 0, 0, 0, 3)
    assert np.all(x == np.array([0, 0, 0, 0]))
    assert np.all(y == np.array([0, 1, 2, 3]))
    assert np.all(z == np.array([0, 2, 4, 6]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))


def test_values_along_line_horizontal():
    """Test values along line horizontal."""
    image = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    x, y, z, s = values_along_line(image, 0, 0, 3, 0)
    assert np.all(x == np.array([0, 1, 2, 3]))
    assert np.all(y == np.array([0, 0, 0, 0]))
    assert np.all(z == np.array([0, 1, 2, 3]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))


def test_values_along_line_diagonal_small():
    """Test values along line diagonal small."""
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = values_along_line(image, 0, 0, 1, 1)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678, 0.70710678]))


# major_axis_arrays
def test_major_axis_arrays_horizontal_major():
    """Test major axis arrays horizontal major."""
    image = np.ones((5, 5))
    x, y, z, s = major_axis_arrays(image, 2, 2, 4, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_major_axis_arrays_vertical_major():
    """Test major axis arrays vertical major."""
    image = np.ones((5, 5))
    x, y, z, s = major_axis_arrays(image, 2, 2, 4, -np.pi / 2)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_major_axis_arrays_large_diameter():
    """Test major axis arrays large diameter."""
    image = np.ones((5, 5))
    x, y, z, s = major_axis_arrays(image, 2, 2, 10, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_major_axis_arrays_rotated():
    """Test major axis arrays rotated."""
    image = np.ones((7, 7))
    x, y, z, s = major_axis_arrays(image, 3, 3, 6, np.pi / 4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 1)
    assert np.isclose(x[-1], 5)
    assert np.isclose(y[0], 5)
    assert np.isclose(y[-1], 1)
    assert np.isclose(s[0], -3)
    assert np.isclose(s[-1], 3)


# minor_axis_arrays
def test_minor_axis_arrays_horizontal():
    """Test minor axis arrays horizontal."""
    image = np.ones((5, 5))
    x, y, z, s = minor_axis_arrays(image, 2, 2, 4, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_minor_axis_arrays_vertical():
    """Test minor axis arrays vertical."""
    image = np.ones((5, 5))
    x, y, z, s = minor_axis_arrays(image, 2, 2, 4, -np.pi / 2)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_minor_axis_arrays_large_diameter():
    """Test minor axis arrays large diameter."""
    image = np.ones((5, 5))
    x, y, z, s = minor_axis_arrays(image, 2, 2, 10, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_minor_axis_arrays_rotated():
    """Test minor axis arrays rotated."""
    image = np.ones((5, 5))
    x, y, z, _ = minor_axis_arrays(image, 2, 2, 10, np.pi / 4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)


# rotate_image
def test_no_rotation():
    """Test no rotation."""
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = rotate_image(original, 1, 1, 0)
    assert np.array_equal(original, result)


def test_full_rotation():
    """Test full rotation."""
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = rotate_image(original, 1, 1, 2 * np.pi)
    assert np.array_equal(original, result)


def test_half_rotation():
    """Test half rotation."""
    original = np.array([[200, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    expected = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 200]], dtype=np.uint8)
    result = rotate_image(original, 1, 1, np.pi)
    assert np.array_equal(expected, result)


def test_quarter_rotation():
    """Test quarter rotation."""
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    expected = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
    result = rotate_image(original, 1, 1, np.pi / 2)
    assert np.array_equal(expected, result)


def test_rotate_and_crop():
    """Test rotate and crop."""
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = rotate_image(original, 1, 1, np.pi / 4)
    assert original.shape == result.shape


# create test image
def test_create_test_image_dimensions():
    """Test create test image dimensions."""
    img = create_test_image(10, 10, 5, 5, 5, 5, 0)
    assert img.shape == (10, 10), f"Expected (10, 10), but got {img.shape}"


def test_invalid_max_value():
    """Test invalid max value."""
    try:
        create_test_image(10, 10, 5, 5, 5, 5, 0, max_value=66000)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_invalid_h():
    """Test invalid h."""
    try:
        create_test_image(-10, 10, 5, 5, 5, 5, 0)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_invalid_v():
    """Test invalid v."""
    try:
        create_test_image(10, 0, 5, 5, 5, 5, 0)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_invalid_phi():
    """Test invalid phi."""
    try:
        create_test_image(10, 10, 5, 5, 5, 5, 10 * np.pi)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_noise_addition():
    """Test noise addition."""
    img = create_test_image(10, 10, 5, 5, 5, 5, 0, noise=10)
    without_noise = create_test_image(10, 10, 5, 5, 5, 5, 0, noise=0)
    assert not np.array_equal(img, without_noise), "Noise not added properly"


def test_dtype_returned():
    """Test dtype returned."""
    img = create_test_image(10, 10, 5, 5, 5, 5, 0, max_value=255)
    assert img.dtype == np.uint8, f"Expected dtype uint8 but got {img.dtype}"

    img = create_test_image(10, 10, 5, 5, 5, 5, 0, max_value=65534)
    assert img.dtype == np.uint16, f"Expected dtype uint16 but got {img.dtype}"


def test_axes_arrays_with_d_minor_returns_four_numeric_arrays():
    """axes_arrays with valid d_minor must return 4 finite numeric arrays."""
    result = axes_arrays(50, 50, 100, 60, 0)
    xp1, yp1, xp2, yp2 = result
    assert np.all(np.isfinite(xp1))
    assert np.all(np.isfinite(yp1))
    assert np.all(np.isfinite(xp2))
    assert np.all(np.isfinite(yp2))


def test_axes_arrays_with_none_d_minor_returns_none_for_minor_axis():
    """axes_arrays with d_minor=None must return None (not array) for minor axis."""
    xp1, yp1, xp2, yp2 = axes_arrays(50, 50, 100, None, 0)
    assert xp2 is None
    assert yp2 is None
    assert np.all(np.isfinite(xp1))
    assert np.all(np.isfinite(yp1))


def test_axes_arrays_with_none_d_minor_not_ndarray():
    """axes_arrays with d_minor=None must not return an ndarray with embedded None values."""
    result = axes_arrays(50, 50, 100, None, 0)
    # Result must be unpackable into 4 Python objects, not a numpy array of objects
    _, _, xp2, _ = result
    assert not isinstance(xp2, np.ndarray), "xp2 should be None, not an ndarray containing None"


def test_crop_image_to_integration_rect_normal():
    """crop_image_to_integration_rect returns a valid cropped image for a normal beam."""
    image = create_test_image(200, 200, 100, 100, 60, 40, 0)
    cropped, new_xc, _ = crop_image_to_integration_rect(image, 100, 100, 60, 40, 0)
    assert cropped is not None
    assert new_xc is not None
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0


def test_crop_image_to_integration_rect_none_d_minor_returns_original():
    """crop_image_to_integration_rect with d_minor=None returns the original image unchanged."""
    image = create_test_image(200, 200, 100, 100, 60, 40, 0)
    cropped, new_xc, _ = crop_image_to_integration_rect(image, 100, 100, 60, None, 0)
    assert np.array_equal(cropped, image)
    assert new_xc == 100


def test_crop_image_to_integration_rect_raises_not_returns_none():
    """crop_image_to_integration_rect must raise ValueError, not silently return None."""
    image = create_test_image(200, 200, 100, 100, 60, 40, 0)
    # A crop request with diameters so small the result is <3 pixels should raise
    with pytest.raises(ValueError):
        crop_image_to_integration_rect(image, 100, 100, 1, 1, 0, mask_diameters=1)


def test_create_test_image_docstring_no_typo():
    """create_test_image docstring must not contain the typo 'magnitued'."""
    doc = create_test_image.__doc__ or ""
    assert "magnitued" not in doc, (
        "create_test_image docstring has typo 'magnitued' — should be 'magnitude'"
    )


def test_create_test_image_docstring_lists_ntype_values():
    """create_test_image docstring must document valid ntype values."""
    doc = create_test_image.__doc__ or ""
    assert "poisson" in doc, (
        "create_test_image docstring does not list valid ntype values (e.g. 'poisson')"
    )


# Run the tests
if __name__ == "__main__":
    test_rotate_points_0_degrees()
    test_rotate_points_90_degrees()
    test_rotate_points_180_degrees()
    test_rotate_points_360_degrees()
    test_values_along_line()
    test_values_along_line_vertical()
    test_values_along_line_horizontal()
    test_values_along_line_diagonal_small()
    test_major_axis_arrays_horizontal_major()
    test_major_axis_arrays_vertical_major()
    test_major_axis_arrays_large_diameter()
    test_major_axis_arrays_rotated()
    test_minor_axis_arrays_horizontal()
    test_minor_axis_arrays_vertical()
    test_minor_axis_arrays_large_diameter()
    test_minor_axis_arrays_rotated()
    test_no_rotation()
    test_full_rotation()
    test_half_rotation()
    test_quarter_rotation()
    test_rotate_and_crop()
    test_create_test_image_dimensions()
    test_invalid_max_value()
    test_invalid_h()
    test_invalid_v()
    test_invalid_phi()
    test_noise_addition()
    test_dtype_returned()

    print("All tests passed!")
