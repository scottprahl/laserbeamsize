"""Tests for functions in analysis.py."""

import matplotlib.pyplot as plt
import numpy as np
import laserbeamsize as lbs

interactive = False

h, v = 400, 400  # Image dimensions
xc, yc, dx, dy, phi = 200, 200, 100, 50, 0
image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi)


def test_mask_diameters():
    try:
        lbs.beam_size(image, mask_diameters=0)
        assert False, "Expected ValueError for mask_diameters <= 0"
    except ValueError:
        pass
    try:
        lbs.beam_size(image, mask_diameters=6)
        assert False, "Expected ValueError for mask_diameters > 5"
    except ValueError:
        pass


def test_corner_fraction():
    try:
        lbs.beam_size(image, corner_fraction=-0.1)
        assert False, "Expected ValueError for corner_fraction < 0"
    except ValueError:
        pass

    try:
        lbs.beam_size(image, corner_fraction=0.6)
        assert False, "Expected ValueError for corner_fraction > 0.25"
    except ValueError:
        pass


def test_nT_values():
    try:
        lbs.beam_size(image, nT=1)
        assert False, "Expected ValueError for nT <= 2"
    except ValueError:
        pass

    try:
        lbs.beam_size(image, nT=5)
        assert False, "Expected ValueError for nT >= 4"
    except ValueError:
        pass


def test_max_iter():
    try:
        lbs.beam_size(image, max_iter=-1)
        assert False, "Expected ValueError for max_iter < 0"
    except ValueError:
        pass

    try:
        lbs.beam_size(image, max_iter=3.5)
        assert False, "Expected ValueError for non-integer max_iter"
    except ValueError:
        pass


def test_phi_values():
    try:
        lbs.beam_size(image, phi_fixed=-10)
        assert False, "Expected ValueError for -2pi <= phi <= 2pi"
    except ValueError:
        pass

    try:
        lbs.beam_size(image, phi_fixed=10)
        assert False, "Expected ValueError for -2pi <= phi <= 2pi"
    except ValueError:
        pass


def wrap_phi(phi):
    # Wrap to (-π/2, π/2]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    if phi <= -np.pi / 2:
        phi += np.pi
    elif phi > np.pi / 2:
        phi -= np.pi

    return phi


def run_test(h, v, xc, yc, dx, dy, phi, max_value=255):

    test_img = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, max_value=max_value)
    result_xc, result_yc, result_dx, result_dy, result_phi = lbs.beam_size(test_img)
    rp = np.degrees(result_phi)
    erp = np.degrees(phi)

    if interactive:
        plt.title("result=%.1f° expected=%.1f°" % (rp, erp))
        plt.imshow(test_img)
        x, y = lbs.image_tools.ellipse_arrays(result_xc, result_yc, result_dx, result_dy, result_phi)
        plt.plot(x, y)
        plt.colorbar()
        plt.show()

    if dy > dx:
        dd = dx
        dx = dy
        dy = dd
        result_phi = wrap_phi(result_phi + np.pi / 2)

    rp = np.degrees(result_phi)

    assert np.isclose(result_xc, xc, rtol=0.03), f"Expected xc around {xc}, but got {result_xc}"
    assert np.isclose(result_yc, yc, rtol=0.03), f"Expected yc around {yc}, but got {result_yc}"
    assert np.isclose(result_dx, dx, rtol=0.03), f"Expected dx around {dx}, but got {result_dx}"
    assert np.isclose(result_dy, dy, rtol=0.03), f"Expected dy around {dy}, but got {result_dy}"
    assert np.isclose(rp, erp, rtol=0.03), f"Expected phi around {erp}°, but got {rp}°"


def test_horizontal_ellipse():
    run_test(400, 400, 200, 200, 100, 50, 0)


def test_vertical_ellipse_rotated():
    run_test(400, 400, 200, 200, 100, 50, np.pi / 6)


def test_vertical_ellipse_negative_rotation():
    run_test(400, 400, 200, 200, 100, 50, -np.pi / 6)


def test_vertical_ellipse_no_rotation():
    run_test(400, 400, 200, 200, 50, 100, 0)


def test_horizontal_ellipse_off_center():
    run_test(400, 400, 150, 300, 50, 100, np.pi / 6)


def test_horizontal_ellipse_4048():
    run_test(400, 400, 200, 200, 100, 50, 0, max_value=4047)


def test_vertical_ellipse_rotated_4048():
    run_test(400, 400, 200, 200, 100, 50, np.pi / 6, max_value=4047)


# Running the tests
if __name__ == "__main__":
    test_mask_diameters()
    test_corner_fraction()
    test_nT_values()
    test_max_iter()
    test_phi_values()
    test_horizontal_ellipse()
    test_vertical_ellipse_rotated()
    test_vertical_ellipse_negative_rotation()
    test_vertical_ellipse_no_rotation()
    test_horizontal_ellipse_off_center()
    test_horizontal_ellipse_4048()
    test_vertical_ellipse_rotated_4048()
    print("All tests passed!")
