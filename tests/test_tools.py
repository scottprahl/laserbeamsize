"""Tests for functions in image_tools.py."""

import numpy as np

from laserbeamsize.image_tools import (
    rotate_image,
    rotate_points,
    values_along_line,
    major_axis_arrays,
    minor_axis_arrays,
    create_test_image,
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
