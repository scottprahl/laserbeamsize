"""Tests for M2 fitting/report generation."""

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


def test_basic_beam_fit_with_fixed_waist_diameter_finds_location():
    """Fixing d0 still fits the waist location and divergence."""
    z, dx, _ = _synthetic_beam_data()

    params, errors = basic_beam_fit(z, dx, 1064e-9, d0=220e-6)

    assert np.isclose(params[0], 220e-6)
    assert np.isclose(params[1], 0.8e-3)
    assert np.isclose(params[2], 18e-3)
    assert errors[0] == 0


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


def test_zone_index_helpers_select_expected_measurements():
    """Zone helpers choose the requested extrema and handle empty zones."""
    z = np.array([0.4, 0.1, 0.3, 0.2])
    zone = np.array([1, 1, 2, 2])

    assert max_index_in_focal_zone(z, zone) == 0
    assert min_index_in_outer_zone(z, zone) == 3
    assert max_index_in_focal_zone(z, np.zeros(4)) is None
    assert min_index_in_outer_zone(z, np.zeros(4)) is None


def _strict_zone_data(n_focal, n_outer):
    """Create exact beam data with controlled focal and outer zone counts."""
    focal = np.array([-0.008, -0.005, -0.002, 0.002, 0.005, 0.008])[:n_focal]
    outer = np.array([-0.050, -0.040, -0.030, -0.025, 0.025, 0.030, 0.040])[:n_outer]
    z = np.concatenate((focal, outer))
    d0 = 100e-6
    theta = 10e-3
    d = np.sqrt(d0**2 + (theta * z) ** 2)
    return z, d, d0


def test_m2_fit_stops_outer_trimming_if_no_index_is_available(monkeypatch):
    """Strict fitting remains usable if outer-zone selection cannot continue."""
    z, d, d0 = _strict_zone_data(5, 6)
    monkeypatch.setattr(m2, "min_index_in_outer_zone", lambda *_args: None)

    params, _, used = lbs.M2_fit(z, d, 632.8e-9, strict=True, z0=0, d0=d0)

    assert np.all(used)
    assert np.isclose(params[2], 10e-3)


def test_m2_fit_stops_focal_trimming_if_no_index_is_available(monkeypatch):
    """Strict fitting remains usable if focal-zone selection cannot continue."""
    z, d, d0 = _strict_zone_data(6, 5)
    monkeypatch.setattr(m2, "max_index_in_focal_zone", lambda *_args: None)

    params, _, used = lbs.M2_fit(z, d, 632.8e-9, strict=True, z0=0, d0=d0)

    assert np.all(used)
    assert np.isclose(params[2], 10e-3)


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


def test_m2_report_single_axis_includes_fitted_values():
    """A single-axis report formats the fitted propagation parameters."""
    z, d = _simple_beam_data()

    report = lbs.M2_report(z, d, 632.8e-9)

    assert report.startswith("Beam propagation parameters\n")
    assert "M^2 =" in report
    assert "d_0 =" in report
    assert "z_R =" in report
    assert "BPP =" in report


def test_m2_report_single_axis_with_lens_includes_original_beam():
    """A lens report describes both the focused and original beams."""
    z, d = _simple_beam_data()

    report = lbs.M2_report(z, d, 632.8e-9, f=100e-3)

    assert "Beam propagation parameters for the focused beam" in report
    assert "Beam propagation parameters for the laser beam" in report


def test_m2_report_two_axes_without_lens_has_one_summary():
    """Two-axis data without a lens returns only the fitted-beam summary."""
    z, dx, dy = _synthetic_beam_data()

    report = lbs.M2_report(z, dx, 1064e-9, d_minor=dy)

    assert "Beam propagation parameters derived from hyperbolic fit" in report
    assert "Beam Propagation Ratio\n" in report
    assert "of the focused beam" not in report
    assert "of the laser beam" not in report


def test_m2_fit_strict_none_index_does_not_crash():
    """M2_fit zone-trimming must not crash with TypeError when a zone index is None."""
    # Many focal-zone points, few outer points — exhausts the outer zone during trimming
    z = np.array(
        [
            0.00085,
            0.00092,
            0.00102,
            0.00110,
            0.00075,
            0.00115,
            0.00128,
            0.00082,
            -0.899,
            -0.699,
            0.851,
            1.101,
        ]
    )
    d0_val, theta, z0_val = 100e-6, 15e-3, 1e-3
    d = np.sqrt(d0_val**2 + (theta * (z - z0_val)) ** 2)
    try:
        lbs.M2_fit(z, d, 632.8e-9, strict=True)
    except TypeError as exc:
        raise AssertionError(f"M2_fit raised TypeError due to None zone index: {exc}") from exc


def test_m2_fit_strict_preserves_focal_zone_points_after_outer_trim():
    """Strict trimming should keep focal-zone points when extra outer points are removed."""
    z = np.array([-34, -29, -24, -20, -4, -1, 3, 6, 21, 26, 31, 36], dtype=float) * 1e-3
    d0_val, theta, z0_val = 100e-6, 10e-3, 1e-3
    d = np.sqrt(d0_val**2 + (theta * (z - z0_val)) ** 2)

    _, _, used = lbs.M2_fit(z, d, 632.8e-9, strict=True)

    focal = np.abs(z - z0_val) < 10e-3
    outer = np.abs(z - z0_val) >= 20e-3
    assert np.sum(used) == 10
    assert np.sum(used & focal) == 4
    assert np.sum(used & outer) == 6
