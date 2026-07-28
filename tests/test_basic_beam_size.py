"""Tests for functions in analysis.py."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import laserbeamsize as lbs
import laserbeamsize.analysis as ana
from laserbeamsize.analysis import wrap_phi, _validate_inputs

interactive = False
h, v = 400, 400  # Image dimensions


def run_test(
    xc,
    yc,
    d_major,
    d_minor,
    phi,
    expect_major=None,
    expect_minor=None,
    expect_phi=None,
    max_value=255,
):
    """Helper for run test."""
    if expect_major is None:
        expect_major = d_major
    if expect_minor is None:
        expect_minor = d_minor
    if expect_phi is None:
        expect_phi = phi

    test_img = lbs.image_tools.create_test_image(h, v, xc, yc, d_major, d_minor, phi, max_value=max_value)
    result_xc, result_yc, result_dx, result_dy, result_phi = lbs.basic_beam_size(test_img)
    erp = np.degrees(expect_phi)
    rp = np.degrees(result_phi)

    if interactive:
        plt.title("result=%.1f° expected=%.1f°" % (rp, erp))
        plt.imshow(test_img)
        assert result_dy is not None
        x, y = lbs.image_tools.ellipse_arrays(result_xc, result_yc, result_dx, result_dy, result_phi)
        plt.plot(x, y)
        plt.colorbar()
        plt.show()

    assert np.isclose(result_xc, xc, rtol=0.03), f"Expected xc = {xc}, but got {result_xc}"
    assert np.isclose(result_yc, yc, rtol=0.03), f"Expected yc = {yc}, but got {result_yc}"
    assert np.isclose(result_dx, expect_major, rtol=0.03), f"Expected d_major = {expect_major}, but got {result_dx}"
    assert np.isclose(result_dy, expect_minor, rtol=0.03), f"Expected d_minor = {expect_minor}, but got {result_dy}"
    assert np.isclose(rp, erp, rtol=0.03), f"Expected phi around {erp}°, but got {rp}°"


def test_horizontal_ellipse():
    """Test horizontal ellipse."""
    run_test(200, 200, 100, 50, 0)


def test_vertical_ellipse_rotated():
    """Test vertical ellipse rotated."""
    run_test(200, 200, 100, 50, np.pi / 6)


def test_vertical_ellipse_negative_rotation():
    """Test vertical ellipse negative rotation."""
    run_test(200, 200, 100, 50, -np.pi / 6)


def test_ellipse_120_degree_rotation():
    """Test ellipse 120 degree rotation."""
    run_test(
        200,
        200,
        100,
        50,
        2 * np.pi / 3,
        expect_major=100,
        expect_minor=50,
        expect_phi=2 * np.pi / 3 - np.pi,
    )


def test_vertical_ellipse_no_rotation():
    """Test vertical ellipse no rotation."""
    run_test(200, 200, 50, 100, 0, expect_major=100, expect_minor=50, expect_phi=np.pi / 2)


def test_horizontal_ellipse_off_center():
    """Test horizontal ellipse off center."""
    run_test(150, 300, 50, 100, np.pi / 6, expect_major=100, expect_minor=50, expect_phi=-np.pi / 3)


def test_horizontal_ellipse_4048():
    """Test horizontal ellipse 4048."""
    run_test(200, 200, 100, 50, 0, max_value=4047)


def test_vertical_ellipse_rotated_4048():
    """Test vertical ellipse rotated 4048."""
    run_test(200, 200, 100, 50, np.pi / 6, max_value=4047)


def test_phi_fixed_validation_accepts_angle_within_pi():
    """_validate_inputs should accept phi_fixed within (-π, π]."""
    image = np.ones((10, 10))
    _validate_inputs(image, phi_fixed=np.pi / 4)
    _validate_inputs(image, phi_fixed=-np.pi / 2)


def test_phi_fixed_validation_rejects_angle_beyond_pi():
    """_validate_inputs should reject phi_fixed > π (likely entered in degrees)."""
    image = np.ones((10, 10))
    with pytest.raises(ValueError, match="radians"):
        _validate_inputs(image, phi_fixed=3.5)  # 3.5 rad > π


def test_wrap_phi_at_half_pi():
    """wrap_phi should return π/2 for π/2 input."""
    assert np.isclose(wrap_phi(np.pi / 2), np.pi / 2)


def test_wrap_phi_at_minus_half_pi_maps_to_plus_half_pi():
    """wrap_phi should map -π/2 to π/2 (both represent same ellipse orientation)."""
    assert np.isclose(wrap_phi(-np.pi / 2), np.pi / 2)


def test_wrap_phi_stays_in_range():
    """wrap_phi should always produce values in (-π/2, π/2]."""
    for phi_deg in range(-360, 360, 5):
        phi = np.radians(phi_deg)
        result = wrap_phi(phi)
        assert -np.pi / 2 < result <= np.pi / 2, f"wrap_phi({phi:.3f}) = {result:.3f} out of (-π/2, π/2]"


def test_basic_beam_size_naive_removed():
    """basic_beam_size_naive should be removed as dead code."""
    assert not hasattr(ana, "basic_beam_size_naive"), "basic_beam_size_naive should be removed (dead code)"


def test_beam_size_d_minor_none_does_not_crash_convergence():
    """beam_size should not crash when d_minor becomes None during iteration."""
    # A degenerate image (all-zero except a line) can yield d_minor=None
    image = np.zeros((50, 50), dtype=np.uint8)
    image[25, :] = 100  # horizontal line — d_minor collapses
    # Should complete without TypeError from abs(None - value)
    _, _, d_major, _, _ = lbs.beam_size(image)
    assert d_major >= 0


# Running the tests
if __name__ == "__main__":
    test_horizontal_ellipse()
    test_vertical_ellipse_rotated()
    test_vertical_ellipse_negative_rotation()
    test_ellipse_120_degree_rotation()
    test_vertical_ellipse_no_rotation()
    test_horizontal_ellipse_off_center()
    print("All tests passed!")
