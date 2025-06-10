"""Tests for functions in image_tools.py."""

import numpy as np
import laserbeamsize as lbs


# rotate_points
def test_rotate_points_0_degrees():
    x, y = lbs.image_tools.rotate_points(1, 0, 0, 0, 0)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)


def test_rotate_points_90_degrees():
    x, y = lbs.image_tools.rotate_points(1, 0, 0, 0, np.pi / 2)
    assert np.isclose(x, 0, atol=1e-8)
    assert np.isclose(y, -1, atol=1e-8)


def test_rotate_points_180_degrees():
    x, y = lbs.image_tools.rotate_points(1, 0, 0, 0, np.pi)
    assert np.isclose(x, -1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)


def test_rotate_points_360_degrees():
    x, y = lbs.image_tools.rotate_points(1, 0, 0, 0, 2 * np.pi)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)


# values_along_line
def test_values_along_line():
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = lbs.image_tools.values_along_line(image, 0, 0, 1, 1)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678, 0.70710678]))


def test_values_along_line_vertical():
    image = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    x, y, z, s = lbs.image_tools.values_along_line(image, 0, 0, 0, 3)
    assert np.all(x == np.array([0, 0, 0, 0]))
    assert np.all(y == np.array([0, 1, 2, 3]))
    assert np.all(z == np.array([0, 2, 4, 6]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))


def test_values_along_line_horizontal():
    image = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    x, y, z, s = lbs.image_tools.values_along_line(image, 0, 0, 3, 0)
    assert np.all(x == np.array([0, 1, 2, 3]))
    assert np.all(y == np.array([0, 0, 0, 0]))
    assert np.all(z == np.array([0, 1, 2, 3]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))


def test_values_along_line_diagonal_small():
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = lbs.image_tools.values_along_line(image, 0, 0, 1, 1)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678, 0.70710678]))


# major_axis_arrays
def test_major_axis_arrays_horizontal_major():
    image = np.ones((5, 5))
    x, y, z, s = lbs.major_axis_arrays(image, 2, 2, 4, 3, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_major_axis_arrays_vertical_major():
    image = np.ones((5, 5))
    x, y, z, s = lbs.major_axis_arrays(image, 2, 2, 3, 4, -np.pi / 2)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_major_axis_arrays_large_diameter():
    image = np.ones((5, 5))
    x, y, z, s = lbs.major_axis_arrays(image, 2, 2, 10, 2, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_major_axis_arrays_rotated():
    image = np.ones((7, 7))
    x, y, z, s = lbs.major_axis_arrays(image, 3, 3, 3, 2, np.pi / 4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 1)
    assert np.isclose(x[-1], 5)
    assert np.isclose(y[0], 5)
    assert np.isclose(y[-1], 1)
    assert np.isclose(s[0], -3)
    assert np.isclose(s[-1], 3)


# minor_axis_arrays
def test_minor_axis_arrays_horizontal():
    image = np.ones((5, 5))
    x, y, z, s = lbs.minor_axis_arrays(image, 2, 2, 4, 3, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_minor_axis_arrays_vertical():
    image = np.ones((5, 5))
    x, y, z, s = lbs.minor_axis_arrays(image, 2, 2, 3, 4, -np.pi / 2)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_minor_axis_arrays_large_diameter():
    image = np.ones((5, 5))
    x, y, z, s = lbs.minor_axis_arrays(image, 2, 2, 10, 2, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


def test_minor_axis_arrays_rotated():
    image = np.ones((5, 5))
    x, y, z, s = lbs.minor_axis_arrays(image, 2, 2, 10, 2, np.pi / 4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)


# rotate_image
def test_no_rotation():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = lbs.rotate_image(original, 1, 1, 0)
    assert np.array_equal(original, result)


def test_full_rotation():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = lbs.rotate_image(original, 1, 1, 2 * np.pi)
    assert np.array_equal(original, result)


def test_half_rotation():
    original = np.array([[200, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    expected = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 200]], dtype=np.uint8)
    result = lbs.rotate_image(original, 1, 1, np.pi)
    assert np.array_equal(expected, result)


def test_quarter_rotation():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    expected = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
    result = lbs.rotate_image(original, 1, 1, np.pi / 2)
    assert np.array_equal(expected, result)


def test_rotate_and_crop():
    original = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = lbs.rotate_image(original, 1, 1, np.pi / 4)
    assert original.shape == result.shape


# create test image
def test_create_test_image_dimensions():
    img = lbs.create_test_image(10, 10, 5, 5, 5, 5, 0)
    assert img.shape == (10, 10), f"Expected (10, 10), but got {img.shape}"


def test_invalid_max_value():
    try:
        lbs.create_test_image(10, 10, 5, 5, 5, 5, 0, max_value=66000)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_invalid_h():
    try:
        lbs.create_test_image(-10, 10, 5, 5, 5, 5, 0)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_invalid_v():
    try:
        lbs.create_test_image(10, 0, 5, 5, 5, 5, 0)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_invalid_phi():
    try:
        lbs.create_test_image(10, 10, 5, 5, 5, 5, 10 * np.pi)
        assert False, "Expected ValueError but got none"
    except ValueError:
        pass


def test_noise_addition():
    img = lbs.create_test_image(10, 10, 5, 5, 5, 5, 0, noise=10)
    without_noise = lbs.create_test_image(10, 10, 5, 5, 5, 5, 0, noise=0)
    assert not np.array_equal(img, without_noise), "Noise not added properly"


def test_dtype_returned():
    img = lbs.create_test_image(10, 10, 5, 5, 5, 5, 0, max_value=255)
    assert img.dtype == np.uint8, f"Expected dtype uint8 but got {img.dtype}"

    img = lbs.create_test_image(10, 10, 5, 5, 5, 5, 0, max_value=65534)
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
