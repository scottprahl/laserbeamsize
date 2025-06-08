"""Tests for functions in analysis.py."""

import matplotlib.pyplot as plt
import numpy as np
import laserbeamsize as lbs

interactive = False
h, v = 400, 400  # Image dimensions


def run_test(
    xc,
    yc,
    dx,
    dy,
    phi,
    expected_dx=None,
    expected_dy=None,
    expected_phi=None,
    max_value=255,
):
    if expected_dx is None:
        expected_dx = dx
    if expected_dy is None:
        expected_dy = dy
    if expected_phi is None:
        expected_phi = phi

    test_img = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, max_value=max_value)
    result_xc, result_yc, result_dx, result_dy, result_phi = lbs.basic_beam_size(test_img)
    erp = np.degrees(expected_phi)
    rp = np.degrees(result_phi)

    if interactive:
        plt.title("result=%.1f° expected=%.1f°" % (rp, erp))
        plt.imshow(test_img)
        x, y = lbs.image_tools.ellipse_arrays(result_xc, result_yc, result_dx, result_dy, result_phi)
        plt.plot(x, y)
        plt.colorbar()
        plt.show()

    assert np.isclose(result_xc, xc, rtol=0.03), f"Expected xc = {xc}, but got {result_xc}"
    assert np.isclose(result_yc, yc, rtol=0.03), f"Expected yc = {yc}, but got {result_yc}"
    assert np.isclose(result_dx, expected_dx, rtol=0.03), f"Expected dx = {expected_dx}, but got {result_dx}"
    assert np.isclose(result_dy, expected_dy, rtol=0.03), f"Expected dy = {expected_dy}, but got {result_dy}"
    assert np.isclose(rp, erp, rtol=0.03), f"Expected phi around {erp}°, but got {rp}°"


def test_horizontal_ellipse():
    run_test(200, 200, 100, 50, 0)


def test_vertical_ellipse_rotated():
    run_test(200, 200, 100, 50, np.pi / 6)


def test_vertical_ellipse_negative_rotation():
    run_test(200, 200, 100, 50, -np.pi / 6)


def test_ellipse_120_degree_rotation():
    run_test(
        200,
        200,
        100,
        50,
        2 * np.pi / 3,
        expected_dx=100,
        expected_dy=50,
        expected_phi=2 * np.pi / 3 - np.pi,
    )


def test_vertical_ellipse_no_rotation():
    run_test(200, 200, 50, 100, 0, expected_dx=100, expected_dy=50, expected_phi=np.pi / 2)


def test_horizontal_ellipse_off_center():
    run_test(150, 300, 50, 100, np.pi / 6, expected_dx=100, expected_dy=50, expected_phi=-np.pi / 3)


def test_horizontal_ellipse_4048():
    run_test(200, 200, 100, 50, 0, max_value=4047)


def test_vertical_ellipse_rotated_4048():
    run_test(200, 200, 100, 50, np.pi / 6, max_value=4047)


# Running the tests
if __name__ == "__main__":
    test_horizontal_ellipse()
    test_vertical_ellipse_rotated()
    test_vertical_ellipse_negative_rotation()
    test_ellipse_120_degree_rotation()
    test_vertical_ellipse_no_rotation()
    test_horizontal_ellipse_off_center()
    print("All tests passed!")
