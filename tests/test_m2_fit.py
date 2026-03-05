"""Tests for M2 fitting/report generation."""

import numpy as np
import laserbeamsize as lbs


def _synthetic_beam_data():
    """Return noiseless hyperbolic-diameter data for both principal axes."""
    z = np.linspace(-10e-3, 10e-3, 21)

    d0x, z0x, thetax = 220e-6, 0.8e-3, 18e-3
    d0y, z0y, thetay = 180e-6, -0.6e-3, 14e-3

    dx = np.sqrt(d0x**2 + (thetax * (z - z0x)) ** 2)
    dy = np.sqrt(d0y**2 + (thetay * (z - z0y)) ** 2)
    return z, dx, dy


def test_m2_report_with_focus_and_minor_has_two_sections():
    """Report with a focusing lens includes focused and original-beam sections."""
    z, dx, dy = _synthetic_beam_data()
    report = lbs.M2_report(z, dx, 1064e-9, d_minor=dy, f=100e-3)

    assert "Beam Propagation Ratio of the focused beam" in report
    assert "Beam Propagation Ratio of the laser beam" in report
    assert "Beam parameter product of the focused beam" in report
    assert "Beam parameter product of the laser beam" in report


def test_m2_report_with_focus_and_minor_uses_iso_artificial_to_original():
    """Original-beam axis values are produced from ISO artificial-waist conversion."""
    z, dx, dy = _synthetic_beam_data()
    lambda0 = 1064e-9
    f = 100e-3
    report = lbs.M2_report(z, dx, lambda0, d_minor=dy, f=f)

    px, ex, _ = lbs.M2_fit(z, dx, lambda0)
    py, ey, _ = lbs.M2_fit(z, dy, lambda0)
    opx, oex = lbs.artificial_to_original(px, ex, f)
    opy, oey = lbs.artificial_to_original(py, ey, f)

    expected_d0x = "       d0x = %.0f ± %.0f µm" % (opx[0] * 1e6, oex[0] * 1e6)
    expected_d0y = "       d0y = %.0f ± %.0f µm" % (opy[0] * 1e6, oey[0] * 1e6)
    expected_z0x = "       z0x = %.0f ± %.0f mm" % (opx[1] * 1e3, oex[1] * 1e3)
    expected_theta_x = "   theta_x = %.2f ± %.2f milliradians" % (opx[2] * 1e3, oex[2] * 1e3)

    assert expected_d0x in report
    assert expected_d0y in report
    assert expected_z0x in report
    assert expected_theta_x in report
