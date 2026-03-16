"""Tests for M2 fitting/report generation."""

import inspect
import io
import sys
import warnings
import numpy as np
import laserbeamsize as lbs
import laserbeamsize.m2_fit as m2
from laserbeamsize.m2_fit import basic_beam_fit, max_index_in_focal_zone, min_index_in_outer_zone


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


def _simple_beam_data():
    """Return a simple noiseless hyperbolic beam dataset."""
    z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770]) * 1e-3
    r = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397]) * 1e-6
    return z, 2 * r


def test_basic_beam_fit_no_nan_when_d_less_than_d0():
    """basic_beam_fit with fixed d0 must not return NaN even when d < d0 for some points."""
    z, d = _simple_beam_data()
    # Use a d0 larger than some measurements to trigger the sqrt-of-negative path
    d0_fixed = d.max() * 1.1
    params, errors = basic_beam_fit(z, d, 632.8e-9, z0=z[np.argmin(d)], d0=d0_fixed)
    assert not any(np.isnan(p) for p in params), f"NaN in params: {params}"
    assert not any(np.isnan(e) for e in errors), f"NaN in errors: {errors}"


def test_m2_fit_strict_warns_not_prints():
    """M2_fit with strict=True and bad data distribution should warn, not print."""
    # Only 4 points, too few for ISO 11146 strict mode — should trigger the warning path
    z = np.array([0.1, 0.2, 0.3, 0.4])
    d = np.array([500e-6, 480e-6, 460e-6, 450e-6])

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lbs.M2_fit(z, d, 632.8e-9, strict=True)
    finally:
        sys.stdout = old_stdout

    printed = captured.getvalue()
    assert printed == "", f"M2_fit printed to stdout instead of using warnings.warn: {printed!r}"


def test_dead_zr_calculation_removed():
    """The dead first zR assignment (immediately overwritten) must be removed."""
    src = inspect.getsource(m2.basic_beam_fit)
    # Count occurrences of "zR = np.pi * d0**2 / (4 * lambda0)" (without M2)
    plain_zr_lines = [ln for ln in src.splitlines() if "zR = np.pi * d0**2 / (4 * lambda0)" in ln and "M2" not in ln]
    assert len(plain_zr_lines) == 0, f"Dead zR assignment still present: {plain_zr_lines}"


def test_max_index_in_focal_zone_uses_numpy():
    """max_index_in_focal_zone should use numpy, not a Python loop."""
    src = inspect.getsource(max_index_in_focal_zone)
    assert "for " not in src, "max_index_in_focal_zone still uses a Python loop"


def test_min_index_in_outer_zone_uses_numpy():
    """min_index_in_outer_zone should use numpy, not a Python loop."""
    src = inspect.getsource(min_index_in_outer_zone)
    assert "for " not in src, "min_index_in_outer_zone still uses a Python loop"


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


def test_m2_fit_strict_none_index_does_not_crash():
    """M2_fit zone-trimming must not crash with TypeError when a zone index is None."""
    # Many focal-zone points, few outer points — exhausts the outer zone during trimming
    z = np.array([
        -0.0001, -0.00005, 0.0, 0.00005,
        -0.0002, 0.0001, 0.0002, -0.00015,
        -1.0, -0.8, 0.8, 1.0,
    ])
    d0_val, theta = 100e-6, 15e-3
    d = np.sqrt(d0_val**2 + (theta * z) ** 2)
    try:
        lbs.M2_fit(z, d, 632.8e-9, strict=True)
    except TypeError as exc:
        raise AssertionError(
            f"M2_fit raised TypeError due to None zone index: {exc}"
        ) from exc
