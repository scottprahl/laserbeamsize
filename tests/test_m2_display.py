"""Tests for m2_display.py."""

# matplotlib.pyplot must be imported after the Agg backend is selected, so the
# matplotlib imports are deliberately split across the backend call.
# pylint: disable=wrong-import-position,protected-access,ungrouped-imports
from typing import cast
import numpy as np
import matplotlib
import pytest

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


@pytest.fixture(autouse=True)
def close_figures():
    """Close Matplotlib figures after every display test."""
    yield
    plt.close("all")


def test_fit_plot_adds_legend_for_symmetric_unused_points(monkeypatch):
    """Excluded points get a legend even when their axial positions sum to zero."""
    z, d = _symmetric_z_data()
    used = np.ones(z.size, dtype=bool)
    used[[0, -1]] = False
    params = np.array([200e-6, 0, 15e-3, 1.2, 10e-3])
    errors = np.zeros(5)
    monkeypatch.setattr(m2d, "M2_fit", lambda *_args, **_kwargs: (params, errors, used))

    _, _, _, returned_used = m2d._fit_plot(z, d, 632.8e-9)

    legend = plt.gca().get_legend()
    assert np.array_equal(returned_used, used)
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"used", "unused"}


def test_m2_radius_plot_handles_dense_ticks_unphysical_fit_and_unused_points(monkeypatch):
    """Dense plots rotate tick labels and show both reference and unused-point legends."""
    z = np.linspace(-10e-3, 10e-3, 12)
    d0 = 200e-6
    theta = 8e-3
    d = np.sqrt(d0**2 + (theta * z) ** 2)
    used = np.ones(z.size, dtype=bool)
    used[[0, -1]] = False
    params = np.array([d0, 0, theta, 0.8, 1e-3])
    errors = np.zeros(5)
    monkeypatch.setattr(m2d, "M2_fit", lambda *_args, **_kwargs: (params, errors, used))

    lbs.M2_radius_plot(z, d, 632.8e-9)

    ax1, ax2 = plt.gcf().axes
    assert all(label.get_rotation() == 90 for label in ax1.get_xticklabels())
    assert all(label.get_rotation() == 90 for label in ax2.get_xticklabels())
    legend = ax1.get_legend()
    assert legend is not None
    assert "unused" in {text.get_text() for text in legend.get_texts()}
    assert "M²=1" in {line.get_label() for line in ax2.lines}
    x, y = ax1.lines[0].get_data()
    assert np.isclose((y[-1] - y[0]) / (x[-1] - x[0]), np.tan(theta / 2) * 1e3)


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


def test_m2_focus_plot_shows_beam_lens_waists_and_rayleigh_region():
    """The focus diagram contains both beam envelopes and its optical landmarks."""
    lbs.M2_focus_plot(w0=250e-6, lambda0=1064e-9, f=100e-3, z0=-150e-3, M2=1.2)

    axis = plt.gca()
    assert axis.get_xlabel() == "Axial Position Relative to Lens (mm)"
    assert axis.get_ylabel() == "Beam Radius (microns)"
    assert "$w_0$=250µm" in axis.get_title()
    assert len(axis.collections) == 2
    assert len(axis.patches) == 1
    assert len(axis.lines) == 4
    assert any(np.allclose(line.get_xdata(), [0, 0]) for line in axis.lines)
