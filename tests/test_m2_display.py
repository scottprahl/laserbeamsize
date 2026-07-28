"""Tests for m2_display.py."""

# pylint: disable=wrong-import-position,protected-access
import inspect
from typing import cast
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import laserbeamsize as lbs
import laserbeamsize.m2_display as m2d


def _symmetric_z_data():
    """Beam data centered at z=0 so sum(z[unused]) can be zero even with unused points."""
    z = np.linspace(-10e-3, 10e-3, 12)
    d0, theta = 200e-6, 15e-3
    d = np.sqrt(d0**2 + (theta * z) ** 2)
    return z, d


def test_fit_plot_uses_np_any_not_sum_for_legend():
    """_fit_plot must use np.any(unused) rather than sum(z[unused]) > 0 for legend condition."""
    src = inspect.getsource(m2d._fit_plot)
    assert "sum(z[unused])" not in src, "_fit_plot still uses sum(z[unused]) > 0 — fails for symmetric z datasets"
    assert (
        "np.any(unused)" in src or "unused.any()" in src
    ), "_fit_plot must use np.any(unused) for the legend condition"


def test_m2_radius_plot_uses_np_any_not_sum_for_legend():
    """M2_radius_plot must use np.any(unused) rather than sum(z[unused]) > 0."""
    src = inspect.getsource(m2d.M2_radius_plot)
    assert "sum(z[unused])" not in src, "M2_radius_plot still uses sum(z[unused]) > 0 — fails for symmetric z datasets"
    assert (
        "np.any(unused)" in src or "unused.any()" in src
    ), "M2_radius_plot must use np.any(unused) for the legend condition"


def test_m2_radius_plot_tick_labels_no_variable_shadowing():
    """Tick-label list comprehension must not use 'z' as loop variable (shadows outer array)."""
    src = inspect.getsource(m2d.M2_radius_plot)
    # The tick label comprehension should use a variable other than 'z'
    assert "for z in ticks" not in src, "Tick label comprehension uses 'z' which shadows the outer z array"


def test_m2_radius_plot_asymptote_uses_half_angle_variable():
    """Asymptote calculation should use an explicit half_angle variable for clarity."""
    src = inspect.getsource(m2d.M2_radius_plot)
    assert "half_angle" in src, "Asymptote slope should use explicit 'half_angle = Theta / 2' variable"


def test_m2_radius_plot_completes_without_error():
    """M2_radius_plot must run without exception on standard beam data."""
    z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770]) * 1e-3
    d = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397]) * 1e-6 * 2
    plt.figure()
    lbs.M2_radius_plot(z, d, 632.8e-9)
    plt.close("all")


def test_m2_diameter_plot_completes_without_error():
    """M2_diameter_plot must run without exception on standard beam data."""
    z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770]) * 1e-3
    d = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397]) * 1e-6 * 2
    plt.figure()
    lbs.M2_diameter_plot(z, d, 632.8e-9)
    plt.close("all")


def test_m2_diameter_plot_minor_residual_spans_use_minor_fit_limits():
    """Minor residual ISO spans must be based on the minor-axis fit, not the major-axis fit."""
    z = np.linspace(-5e-3, 5e-3, 11)
    d_major = np.sqrt((120e-6) ** 2 + (18e-3 * (z - 0.8e-3)) ** 2)
    d_minor = np.sqrt((80e-6) ** 2 + (10e-3 * (z + 1.5e-3)) ** 2)

    plt.figure()
    lbs.M2_diameter_plot(z, d_major, 1064e-9, d_minor=d_minor)

    fig = plt.gcf()
    minor_residual_ax = fig.axes[3]
    params, _, _ = lbs.M2_fit(z, d_minor, 1064e-9)
    z0y = params[1]
    zR = params[4]
    zmin = min(np.min(z), z0y - 4 * zR)
    zmax = max(np.max(z), z0y + 4 * zR)

    left_span = cast(Rectangle, minor_residual_ax.patches[1])
    right_span = cast(Rectangle, minor_residual_ax.patches[2])

    assert np.isclose(left_span.get_x(), (z0y - 2 * zR) * 1e3)
    assert np.isclose(left_span.get_width(), (zmin - (z0y - 2 * zR)) * 1e3)
    assert np.isclose(right_span.get_x(), (z0y + 2 * zR) * 1e3)
    assert np.isclose(right_span.get_width(), (zmax - (z0y + 2 * zR)) * 1e3)

    plt.close("all")
