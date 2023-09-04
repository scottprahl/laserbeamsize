# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

"""
Tests for functions in background.py.
"""

import numpy as np
import laserbeamsize


########### subtract_background_image

def test_basic_subtraction():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=float)
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)

    result = laserbeamsize.subtract_background_image(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtraction_iso_false():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = np.array([[10, 15, 20], [15, 20, 25]], dtype=float)
    expected = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)

    result = laserbeamsize.subtract_background_image(original, background, iso_noise=False)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtraction_iso_true():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = np.array([[10, 15, 20], [15, 20, 25]], dtype=float)
    expected = np.array([[-5, -5, -5], [-5, -5, -5]], dtype=float)

    result = laserbeamsize.subtract_background_image(original, background, iso_noise=True)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_subtraction_type_float():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = np.array([[5, 5, 5], [5, 5, 5]], dtype=np.uint8)

    result = laserbeamsize.subtract_background_image(original, background, iso_noise=False)
    assert result.dtype == float

########### subtract_constant

def test_basic_subtract_constant():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=float)
    background = 5
    expected = np.array([[5, 10, 15], [25, 30, 35]], dtype=float)

    result = laserbeamsize.subtract_constant(original, background)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtract_constant_iso_false():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[0, 0, 5], [0, 5, 10]], dtype=float)

    result = laserbeamsize.subtract_constant(original, background, iso_noise=False)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_negative_subtract_constant_iso_true():
    original = np.array([[5, 10, 15], [10, 15, 20]], dtype=float)
    background = 10
    expected = np.array([[-5, 0, 5], [0, 5, 10]], dtype=float)

    result = laserbeamsize.subtract_constant(original, background, iso_noise=True)
    assert np.all(np.isclose(result, expected, atol=1e-5))

def test_subtract_constant_type_float():
    original = np.array([[10, 15, 20], [30, 35, 40]], dtype=np.uint8)
    background = 5

    result = laserbeamsize.subtract_constant(original, background)
    assert result.dtype == np.float64
