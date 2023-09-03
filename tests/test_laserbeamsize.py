import numpy as np
import pytest
import laserbeamsize

########## rotate_points 

def test_rotate_points_0_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, 0)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)

def test_rotate_points_90_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, np.pi/2)
    assert np.isclose(x, 0, atol=1e-8)
    assert np.isclose(y, -1, atol=1e-8)

def test_rotate_points_180_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, np.pi)
    assert np.isclose(x, -1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)

def test_rotate_points_360_degrees():
    x, y = laserbeamsize.laserbeamsize.rotate_points(1, 0, 0, 0, 2*np.pi)
    assert np.isclose(x, 1, atol=1e-8)
    assert np.isclose(y, 0, atol=1e-8)

########## values_along_line 

def test_values_along_line():
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 1, 1, 2)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678,  0.70710678]))

def test_values_along_line_vertical():
    image = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 0, 3, 4)
    assert np.all(x == np.array([0, 0, 0, 0]))
    assert np.all(y == np.array([0, 1, 2, 3]))
    assert np.all(z == np.array([0, 2, 4, 6]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))

def test_values_along_line_horizontal():
    image = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 3, 0, 4)
    assert np.all(x == np.array([0, 1, 2, 3]))
    assert np.all(y == np.array([0, 0, 0, 0]))
    assert np.all(z == np.array([0, 1, 2, 3]))
    assert np.allclose(s, np.array([-1.5, -0.5, 0.5, 1.5]))

def test_values_along_line_diagonal_small():
    image = np.array([[0, 1], [2, 3]])
    x, y, z, s = laserbeamsize.laserbeamsize.values_along_line(image, 0, 0, 1, 1, 2)
    assert np.all(x == np.array([0, 1]))
    assert np.all(y == np.array([0, 1]))
    assert np.all(z == np.array([0, 3]))
    assert np.allclose(s, np.array([-0.70710678,  0.70710678]))

############## major_axis_arrays
def test_major_axis_arrays_horizontal_major():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 4, 3, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_major_axis_arrays_vertical_major():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 3, 4, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_major_axis_arrays_large_diameter():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 10, 2, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_major_axis_arrays_rotated():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.major_axis_arrays(image, 2, 2, 4, 2, np.pi/4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 3)
    assert np.isclose(y[0], 3)
    assert np.isclose(y[-1], 0)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

############## minor_axis_arrays
def test_minor_axis_arrays_horizontal():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 4, 3, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_minor_axis_arrays_vertical():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 3, 4, 0)
    assert np.all(y == 2)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_minor_axis_arrays_large_diameter():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 10, 2, 0)
    assert np.all(x == 2)
    assert np.all(z == 1)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 4)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)

def test_minor_axis_arrays_rotated():
    image = np.ones((5, 5))
    x, y, z, s = laserbeamsize.minor_axis_arrays(image, 2, 2, 4, 2, np.pi/4)
    assert np.all(z == 1)
    assert np.isclose(x[0], 0)
    assert np.isclose(x[-1], 3)
    assert np.isclose(y[0], 0)
    assert np.isclose(y[-1], 3)
    assert np.isclose(s[0], -2)
    assert np.isclose(s[-1], 2)


