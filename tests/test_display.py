"""Tests for display functions."""

# matplotlib.pyplot must be imported after the Agg backend is selected, so the
# matplotlib imports are deliberately split across the backend call.
# pylint: disable=wrong-import-position,protected-access,ungrouped-imports
import re
import numpy as np
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import laserbeamsize as lbs
import laserbeamsize.display as disp


@pytest.fixture(autouse=True)
def close_figures():
    """Close Matplotlib figures after every display test."""
    yield
    plt.close("all")


@pytest.mark.parametrize(
    ("d_major", "d_minor", "expected_ellipticity"),
    [(4.0, 2.0, 0.5), (2.0, 4.0, 0.5), (3.0, 3.0, 1.0)],
)
def test_beam_ellipticity_orders_diameters(d_major, d_minor, expected_ellipticity):
    """Ellipticity is the smaller-to-larger diameter ratio."""
    ellipticity, d_circular = lbs.beam_ellipticity(d_major, d_minor)

    assert ellipticity == expected_ellipticity
    assert np.isclose(d_circular, np.sqrt((d_major**2 + d_minor**2) / 2))


def test_plot_beam_diagram_draws_expected_annotations():
    """The reference diagram contains its axes, dimensions, and beam outlines."""
    lbs.plot_beam_diagram()

    axis = plt.gca()
    labels = {text.get_text() for text in axis.texts}
    assert {"x", "y", r"$\phi$", r"$d_{major}$", r"$d_{minor}$"} <= labels
    assert len(axis.lines) >= 4
    assert not axis.axison


def test_plot_beam_diagram_uses_a_single_axes():
    """Drawing must reuse the subplot axes.

    Calling plt.axes() instead of plt.gca() added a second axes on top of the
    one made by subplots(), and since axis("off") only applied to the new one,
    the empty original showed through as a ticked frame behind the diagram.
    """
    lbs.plot_beam_diagram()

    figure = plt.gcf()
    assert len(figure.axes) == 1, "expected one axes, got %d" % len(figure.axes)

    axis = figure.axes[0]
    assert len(axis.lines) >= 4, "the diagram was drawn on a different axes"
    assert not axis.axison, "a visible frame remains behind the diagram"
    assert axis.get_aspect() == 1.0


def test_plot_beam_diagram_scales_both_rectangle_dimensions(monkeypatch):
    """The integration rectangle preserves the beam's major/minor aspect ratio."""
    captured = {}

    def fake_rotated_rect_arrays(_xc, _yc, d_major, d_minor, _phi):
        captured["diameters"] = (d_major, d_minor)
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    monkeypatch.setattr(disp, "rotated_rect_arrays", fake_rotated_rect_arrays)

    lbs.plot_beam_diagram()

    assert captured["diameters"] == (150, 75)


def test_plot_image_analysis_uses_minor_axis_data_for_minor_plot(monkeypatch):
    """Minor-axis limits and label placement depend on minor-axis data."""
    monkeypatch.setattr(disp, "_prepare_beam_analysis", lambda *_args, **_kwargs: (3, 2, 2, 4, 2, 0))
    monkeypatch.setattr(disp, "subtract_iso_background", lambda image, **_kwargs: image.astype(float))
    monkeypatch.setattr(disp, "iso_background", lambda *_args, **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(
        disp,
        "major_axis_arrays",
        lambda *_args, **_kwargs: (
            np.array([-100.0, 100.0]),
            np.zeros(2),
            np.ones(2),
            np.array([-100.0, 100.0]),
        ),
    )
    monkeypatch.setattr(
        disp,
        "minor_axis_arrays",
        lambda *_args, **_kwargs: (
            np.array([-1.0, 1.0]),
            np.zeros(2),
            np.ones(2),
            np.array([-1.0, 1.0]),
        ),
    )

    lbs.plot_image_analysis(np.ones((5, 5)))

    minor_axis = next(axis for axis in plt.gcf().axes if axis.get_title() == "Minor Axis")
    minor_label = next(text for text in minor_axis.texts if text.get_text().startswith("$d_{minor}$"))
    assert np.allclose(minor_axis.get_xlim(), (-1, 1))
    assert minor_label.get_position()[0] == 0
    assert minor_label.get_verticalalignment() == "bottom"


def test_plot_image_analysis_completes_without_error():
    """plot_image_analysis should run without exception for a standard beam."""
    image = lbs.image_tools.create_test_image(200, 200, 100, 100, 60, 40, 0, noise=5)
    plt.figure()
    lbs.plot_image_analysis(image)
    plt.close("all")


def test_plot_image_analysis_title_uses_pixels_by_default():
    """Test plot image analysis title uses pixels by default."""
    h, v = 120, 140
    test_img = lbs.image_tools.create_test_image(h, v, 60, 70, 40, 30, 0.0)

    lbs.plot_image_analysis(test_img)

    fig = plt.gcf()
    titles = [ax.get_title() for ax in fig.axes]
    beam_titles = [title for title in titles if "$d_{major}$" in title]

    assert beam_titles, "Expected a beam size title in plot_image_analysis output"
    assert "px" in beam_titles[0]
    assert "µm" not in beam_titles[0]

    plt.close(fig)


def test_format_beam_title_z_docstring_says_meters():
    """_format_beam_title docstring must say z is in meters, not mm."""
    doc = disp._format_beam_title.__doc__ or ""
    assert "in mm" not in doc, (
        "_format_beam_title docstring still says z should be 'in mm'; "
        "it should say 'in meters' (the implementation multiplies by 1e3 to convert to mm for display)"
    )


def test_format_beam_title_numpy_nan_does_not_raise():
    """_format_beam_title must handle numpy NaN (np.float64) without raising."""
    nan64 = np.float64("nan")
    result = disp._format_beam_title(nan64, nan64, "mm")
    assert "fail" in result, f"Expected 'fail' in title for NaN inputs, got: {result!r}"


def test_format_beam_title_formats_mm_and_z_position():
    """Millimeter values retain decimals and z is converted from meters."""
    result = disp._format_beam_title(1.234, 0.567, "mm", z=0.25)

    assert result == "z=250mm, $d_{major}$=1.23mm, $d_{minor}$=0.57mm"


def test_setup_scale_and_labels_uses_physical_units():
    """A supplied pixel size controls scaling and labels."""
    scale, label, units = disp._setup_scale_and_labels(0.01, "mm")

    assert scale == 0.01
    assert label == "Distance from Center [mm]"
    assert units == "mm"


def test_crop_image_if_needed_supports_explicit_size():
    """An explicit crop size is centered on the fitted beam."""
    image = np.arange(100).reshape(10, 10)

    cropped, xc, yc = disp._crop_image_if_needed(image, 5, 5, 2, 2, 0, [4, 6], 1, 3)

    assert cropped.shape == (4, 6)
    assert (xc, yc) == (3, 2)


def test_crop_image_if_needed_keeps_image_when_crop_is_too_small():
    """An unusably small explicit crop falls back to the original image."""
    image = np.ones((10, 10))

    cropped, xc, yc = disp._crop_image_if_needed(image, 5, 5, 2, 2, 0, [2, 2], 1, 3)

    assert cropped is image
    assert (xc, yc) == (5, 5)


def test_crop_image_if_needed_supports_integration_crop():
    """True selects the fitted integration rectangle."""
    image = np.ones((20, 20))

    cropped, xc, yc = disp._crop_image_if_needed(image, 10, 10, 4, 2, 0, True, 1, 3)

    assert cropped.shape == (6, 12)
    assert np.isclose(xc, 6)
    assert np.isclose(yc, 3)


def test_plot_image_and_fit_returns_scaled_measurements_and_colorbar():
    """The public fit plot reports physical dimensions and can add a colorbar."""
    image = lbs.image_tools.create_test_image(100, 100, 50, 50, 30, 20, 0)

    xc, yc, d_major, d_minor, phi = lbs.plot_image_and_fit(
        image, pixel_size=0.01, units="mm", colorbar=True, corner_fraction=0.1
    )

    assert np.isclose(xc, 0.5, rtol=0.03)
    assert np.isclose(yc, 0.5, rtol=0.03)
    assert np.isclose(d_major, 0.3, rtol=0.05)
    assert np.isclose(d_minor, 0.2, rtol=0.05)
    assert np.isclose(phi, 0, atol=0.03)
    assert "mm" in plt.gca().get_title()
    assert len(plt.gcf().axes) == 2


def test_plot_image_and_fit_preserves_failed_minor_fit(monkeypatch):
    """A failed minor-axis fit is returned as None and shown without crashing."""
    monkeypatch.setattr(disp, "_prepare_beam_analysis", lambda *_args, **_kwargs: (3, 2, 2, 2, None, 0))
    monkeypatch.setattr(disp, "subtract_iso_background", lambda image, **_kwargs: image.astype(float))

    result = lbs.plot_image_and_fit(np.ones((5, 5)))

    assert result == (2, 2, 2, None, 0)
    assert "fail" in plt.gca().get_title()


def test_plot_image_montage_handles_layout_titles_and_colorbar_selection(monkeypatch):
    """Montage layout applies z titles, label suppression, and one colorbar request."""
    calls = []

    def fake_plot_image_and_fit(_image, **kwargs):
        calls.append(kwargs)
        return 1, 2, 4, 2, 0

    monkeypatch.setattr(disp, "plot_image_and_fit", fake_plot_image_and_fit)
    images = [np.ones((4, 4)) for _ in range(3)]

    d_major, d_minor = lbs.plot_image_montage(images, z=np.array([0.1, 0.2, 0.3]), cols=2, vmax=10, crop=[4, 4])

    assert np.array_equal(d_major, np.array([4, 4, 4]))
    assert np.array_equal(d_minor, np.array([2, 2, 2]))
    assert [call["colorbar"] for call in calls] == [False, True, False]
    assert calls[0]["units"] == "px"
    assert any("z=100mm" in axis.get_title() for axis in plt.gcf().axes)
    assert not plt.gcf().axes[-1].axison


def test_plot_image_montage_formats_titles_without_z(monkeypatch):
    """Montages without axial positions use diameter-only titles."""
    monkeypatch.setattr(disp, "plot_image_and_fit", lambda *_args, **_kwargs: (1, 2, 4, 2, 0))

    lbs.plot_image_montage([np.ones((4, 4))], cols=1, pixel_size=0.5, units="mm")

    title = plt.gca().get_title()
    assert "z=" not in title
    assert "mm" in title


def test_plot_image_analysis_uses_inside_labels_for_small_mask():
    """Small integration masks place diameter labels above the arrows."""
    image = lbs.image_tools.create_test_image(100, 100, 50, 50, 30, 20, 0)

    lbs.plot_image_analysis(image, mask_diameters=1, corner_fraction=0.1)

    major_axis = next(axis for axis in plt.gcf().axes if axis.get_title() == "Major Axis")
    minor_axis = next(axis for axis in plt.gcf().axes if axis.get_title() == "Minor Axis")
    assert any(text.get_text().startswith("$d_{major}$") for text in major_axis.texts)
    assert any(text.get_text().startswith("$d_{minor}$") for text in minor_axis.texts)


def test_plot_image_analysis_displays_failed_minor_fit(monkeypatch):
    """A missing minor diameter produces a clear failure panel."""
    monkeypatch.setattr(disp, "_prepare_beam_analysis", lambda *_args, **_kwargs: (3, 2, 2, 2, None, 0))
    monkeypatch.setattr(disp, "subtract_iso_background", lambda image, **_kwargs: image.astype(float))
    monkeypatch.setattr(disp, "iso_background", lambda *_args, **_kwargs: (0.0, 0.0))

    lbs.plot_image_analysis(np.ones((5, 5)))

    assert any(text.get_text() == "Fit failed." for axis in plt.gcf().axes for text in axis.texts)


def test_init_docstring_does_not_reference_m2_module():
    """__init__.py docstring must not mention laserbeamsize.m2 (the module was split)."""
    doc = lbs.__doc__ or ""
    assert "laserbeamsize.m2`" not in doc, (
        "__init__.py docstring still references `laserbeamsize.m2` which no longer exists; "
        "update to reference `laserbeamsize.m2_fit` and `laserbeamsize.m2_display`"
    )


def test_prepare_beam_analysis_docstring_matches_return_count():
    """_prepare_beam_analysis docstring must list exactly 6 return values."""
    doc = disp._prepare_beam_analysis.__doc__ or ""
    match = re.search(r"tuple:\s*\(([^)]+)\)", doc)
    assert match is not None, "_prepare_beam_analysis docstring has no 'tuple: (...)' return description"
    items = [x.strip() for x in match.group(1).split(",")]
    assert len(items) == 6, (
        f"_prepare_beam_analysis docstring lists {len(items)} return values " f"but the function returns 6: {items}"
    )


def test_set_zero_to_lightgray_maps_zero_within_range():
    """Zero maps to light gray when it lies within the data range."""
    cmap = disp.set_zero_to_lightgray("viridis", -1.0, 1.0)
    assert np.allclose(cmap.colors[128], [0.827, 0.827, 0.827, 1.0])


def test_set_zero_to_lightgray_handles_zero_at_top_of_range():
    """set_zero_to_lightgray must not raise when zero is the top of the data range."""
    cmap = disp.set_zero_to_lightgray("viridis", -1.0, 0.0)
    assert np.allclose(cmap.colors[-1], [0.827, 0.827, 0.827, 1.0])
