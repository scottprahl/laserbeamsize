"""Tests for M2 fitting/report generation."""

import io
import sys
import warnings
import numpy as np
import pytest
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


@pytest.mark.parametrize(
    ("fixed_z0", "fixed_d0", "expected_lower", "expected_upper"),
    [
        (None, None, [0, -np.inf, 0], [np.inf, np.inf, np.inf]),
        (None, 220e-6, [-np.inf, 0], [np.inf, np.inf]),
        (0.8e-3, None, [0, 0], [np.inf, np.inf]),
        (0.8e-3, 220e-6, 0, np.inf),
    ],
)
def test_basic_beam_fit_constrains_physical_parameters(monkeypatch, fixed_z0, fixed_d0, expected_lower, expected_upper):
    """Every fit branch constrains waist diameter and divergence to be nonnegative."""
    observed_bounds = []

    def fake_curve_fit(_function, _z, _d, *, p0, bounds, **_options):
        observed_bounds.append(bounds)
        return np.asarray(p0), np.eye(len(p0))

    monkeypatch.setattr(m2.scipy.optimize, "curve_fit", fake_curve_fit)
    z, d, _ = _synthetic_beam_data()

    params, _ = basic_beam_fit(z, d, 1064e-9, z0=fixed_z0, d0=fixed_d0)

    assert len(observed_bounds) == 1
    lower, upper = observed_bounds[0]
    assert np.allclose(lower, expected_lower)
    assert np.allclose(upper, expected_upper)
    assert params[0] >= 0
    assert params[2] >= 0
    assert params[3] >= 0
    assert params[4] >= 0


@pytest.mark.parametrize(
    ("lambda0", "d0", "message"),
    [
        (0, None, "lambda0 must be positive"),
        (-1064e-9, None, "lambda0 must be positive"),
        (1064e-9, -220e-6, "d0 must be nonnegative"),
    ],
)
def test_basic_beam_fit_rejects_nonphysical_fixed_inputs(lambda0, d0, message):
    """Caller-supplied physical parameters cannot make derived values negative."""
    z, diameters, _ = _synthetic_beam_data()

    with pytest.raises(ValueError, match=message):
        basic_beam_fit(z, diameters, lambda0, d0=d0)


def test_basic_beam_fit_propagates_covariance_to_m2_and_rayleigh_range(monkeypatch):
    """Derived uncertainties must include d0/Theta covariance without double counting."""
    fitted = np.array([2e-3, 0.1, 10e-3])
    covariance = np.array(
        [
            [4e-10, 0, 1e-9],
            [0, 9e-6, 0],
            [1e-9, 0, 1e-8],
        ]
    )
    monkeypatch.setattr(m2.scipy.optimize, "curve_fit", lambda *_args, **_kwargs: (fitted, covariance))

    z = np.array([-1.0, 0.0, 1.0])
    d = np.array([11e-3, 2e-3, 10e-3])
    lambda0 = 632.8e-9
    params, errors = basic_beam_fit(z, d, lambda0)

    m2_factor = np.pi / (4 * lambda0)
    m2_gradient = np.array([m2_factor * fitted[2], 0, m2_factor * fitted[0]])
    zR_gradient = np.array([1 / fitted[2], 0, -fitted[0] / fitted[2] ** 2])

    assert np.allclose(errors[:3], np.sqrt(np.diag(covariance)))
    assert np.isclose(errors[3], np.sqrt(m2_gradient @ covariance @ m2_gradient))
    assert np.isclose(errors[4], np.sqrt(zR_gradient @ covariance @ zR_gradient))
    assert np.isclose(params[4], fitted[0] / fitted[2])


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


def test_m2_report_propagates_two_axis_summary_errors(monkeypatch):
    """Combined two-axis values must use the derivatives of their reported means."""
    fits = [
        (
            np.array([200e-6, 4e-3, 6e-3, 8.0, 10e-3]),
            np.array([20e-6, 0.4e-3, 0.6e-3, 0.8, 1.0e-3]),
            np.ones(2, dtype=bool),
        ),
        (
            np.array([400e-6, 6e-3, 8e-3, 18.0, 12e-3]),
            np.array([40e-6, 0.6e-3, 0.8e-3, 1.8, 1.2e-3]),
            np.ones(2, dtype=bool),
        ),
    ]
    monkeypatch.setattr(m2, "M2_fit", lambda *_args, **_kwargs: fits.pop(0))

    report = m2.M2_report(np.array([0.0, 1.0]), np.ones(2), 632.8e-9, d_minor=np.ones(2))

    assert "        M2 = 12.00 ± 0.85" in report
    assert "        d0 = 300 ± 22 µm" in report
    assert "        zR = 11 ± 1 mm" in report
    assert "     theta = 7.00 ± 0.50 milliradians" in report


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


def test_m2_fit_strict_refits_unconstrained_waist_after_excluding_middle_zone():
    """Strict fitting must not freeze z0 at the preliminary all-data estimate."""
    focal = np.array([-0.008, -0.005, -0.002, 0.002, 0.005, 0.008])
    outer = np.array([-0.050, -0.040, -0.030, 0.030, 0.040, 0.050])
    middle = np.array([0.014, 0.016])
    z = np.concatenate((focal, outer, middle))

    d0_val = 100e-6
    theta = 10e-3
    d = np.sqrt(d0_val**2 + (theta * z) ** 2)
    d[:12] *= np.array([1.002, 0.999, 1.001, 1.000, 0.998, 1.002, 1.001, 0.999, 1.002, 0.998, 1.001, 1.000])
    d[-2:] *= 1.5

    preliminary, _ = basic_beam_fit(z, d, 632.8e-9)
    params, _, used = lbs.M2_fit(z, d, 632.8e-9, strict=True)
    expected, _ = basic_beam_fit(z[used], d[used], 632.8e-9)

    assert not np.any(used[-2:])
    assert not np.isclose(preliminary[1], 0, atol=1e-4)
    assert np.allclose(params, expected)
    assert np.isclose(params[1], 0, atol=1e-4)
    assert np.isclose(params[0], d0_val, rtol=0.01)
    assert np.isclose(params[2], theta, rtol=0.01)
