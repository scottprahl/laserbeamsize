# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

"""Tests for functions in background.py."""

import numpy as np
import laserbeamsize as lbs


# subtract_background_image
def test_basic_subtraction():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=float)
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)

    result = lbs.subtract_background_image(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_negative_subtraction():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = np.array([[10, 15, 20], [15, 20, 25]], dtype=float)
    expected = np.array([[-5, -5, -5], [-5, -5, -5]], dtype=float)

    result = lbs.subtract_background_image(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_subtraction_type_float():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=np.uint8)

    result = lbs.subtract_background_image(original, background)
    assert result.dtype == float


# subtract_constant
def test_basic_subtract_constant():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = 5
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)

    result = lbs.subtract_constant(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_negative_subtract_constant_iso_false():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[0, 0, 5], [0, 5, 10]], dtype=float)

    result = lbs.subtract_constant(original, background, iso_noise=False)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_negative_subtract_constant_iso_true():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[-5, 0, 5], [0, 5, 10]], dtype=float)

    result = lbs.subtract_constant(original, background, iso_noise=True)
    assert np.all(np.isclose(result, expected, atol=1e-5))


def test_subtract_constant_type_float():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = 5

    result = lbs.subtract_constant(original, background)
    assert result.dtype == np.float64


# corner_background
def test_corner_known_mean_stdev():
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    corner_mean, corner_stdev = lbs.corner_background(image, 0.25)
    # considering the corners: 1, 4, 13, 16
    expected_mean = np.mean([1, 4, 13, 16])
    expected_stdev = np.std([1, 4, 13, 16])
    assert np.isclose(corner_mean, expected_mean)
    assert np.isclose(corner_stdev, expected_stdev)


def test_corner_zero_corner_fraction():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    corner_mean, corner_stdev = lbs.corner_background(image, 0)
    assert corner_mean == 0
    assert corner_stdev == 0


def test_corner_varying_corner_fraction():
    image = np.ones((100, 100))  # uniform image
    corner_mean, corner_stdev = lbs.corner_background(image, 0.05)
    assert corner_mean == 1
    assert corner_stdev == 0


def test_corner_uniform_image():
    image = np.ones((100, 100))
    corner_mean, corner_stdev = lbs.corner_background(image, 0.05)
    assert corner_mean == 1
    assert corner_stdev == 0


def test_corner_image_data_types():
    image_float = np.ones((100, 100), dtype=float)
    image_int = np.ones((100, 100), dtype=int)
    corner_mean_float, corner_stdev_float = lbs.corner_background(image_float, 0.05)
    corner_mean_int, corner_stdev_int = lbs.corner_background(image_int, 0.05)
    assert corner_mean_float == corner_mean_int == 1
    assert corner_stdev_float == corner_stdev_int == 0


def test_corner_test_image():
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi)
    corner_mean, corner_stdev = lbs.corner_background(image)
    assert corner_mean == 0
    assert corner_stdev == 0


def test_corner_test_image_with_noise():
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    noise = 20
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, noise=noise)
    corner_mean, corner_stdev = lbs.corner_background(image)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


# iso_background
def test_iso_known_mean_stdev():
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    corner_mean, corner_stdev = lbs.iso_background(image, 0.25)
    # considering the corners: 1, 4, 13, 16
    expected_mean = np.mean(image)
    expected_stdev = np.std(image)
    assert np.isclose(corner_mean, expected_mean)
    assert np.isclose(corner_stdev, expected_stdev)


def test_iso_zero_corner_fraction():
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
    noise = 20
    image = np.random.poisson(noise, size=(400, 400))
    corner_mean, corner_stdev = lbs.iso_background(image)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


def test_iso_test_image_with_noise():
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    noise = 20
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, noise=noise)
    corner_mean, corner_stdev = lbs.iso_background(image)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


def test_iso_varying_corner_fraction():
    h, v, xc, yc, dx, dy, phi = 400, 400, 200, 200, 50, 100, 0
    noise = 20
    image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, noise=noise)
    corner_mean, corner_stdev = lbs.iso_background(image, 0.05)
    assert np.isclose(corner_mean, noise, rtol=0.1)
    assert np.isclose(corner_stdev, np.sqrt(noise), rtol=0.1)


def test_iso_uniform_image():
    image = np.ones((100, 100))
    corner_mean, corner_stdev = lbs.iso_background(image)
    assert corner_mean == 1
    assert corner_stdev == 0


def test_iso_image_data_types():
    image_float = np.ones((100, 100), dtype=float)
    image_int = np.ones((100, 100), dtype=int)
    corner_mean_float, corner_stdev_float = lbs.iso_background(image_float, 0.05)
    corner_mean_int, corner_stdev_int = lbs.iso_background(image_int, 0.05)
    assert corner_mean_float == corner_mean_int == 1
    assert corner_stdev_float == corner_stdev_int == 0
