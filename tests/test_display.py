"""Tests for display functions."""

# pylint: disable=wrong-import-position,protected-access
import re
import inspect
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import laserbeamsize as lbs
import laserbeamsize.display as disp


def test_plot_beam_diagram_rect_minor_uses_d_minor():
    """plot_beam_diagram must use d_minor (not d_major) for the minor rectangle dimension."""
    src = inspect.getsource(disp.plot_beam_diagram)
    assert "rect_minor = d_minor" in src, "plot_beam_diagram still uses d_major for rect_minor (copy-paste bug)"


def test_plot_image_analysis_minor_label_uses_minor_variables():
    """Minor-axis label condition must compare r_minor_s against s_minor_px, not major axis variables."""
    src = inspect.getsource(disp.plot_image_analysis)
    # Find the block for the minor axis (after 'if d_minor_px is not None:' in subplot 2,2,4)
    # The condition guarding minor axis label placement must reference minor variables
    assert (
        "r_major_s < max(s_major_px)" not in src.rsplit("Minor Axis", maxsplit=1)[-1]
    ), "plot_image_analysis minor-axis label still uses r_major_s/s_major_px (copy-paste bug)"


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
    import numpy as np  # pylint: disable=import-outside-toplevel
    nan64 = np.float64("nan")
    result = disp._format_beam_title(nan64, nan64, "mm")
    assert "fail" in result, f"Expected 'fail' in title for NaN inputs, got: {result!r}"


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
    match = re.search(r'tuple:\s*\(([^)]+)\)', doc)
    assert match is not None, "_prepare_beam_analysis docstring has no 'tuple: (...)' return description"
    items = [x.strip() for x in match.group(1).split(',')]
    assert len(items) == 6, (
        f"_prepare_beam_analysis docstring lists {len(items)} return values "
        f"but the function returns 6: {items}"
    )


def test_set_zero_to_lightgray_no_commented_out_color():
    """set_zero_to_lightgray must not contain a commented-out alternative color."""
    src = inspect.getsource(disp.set_zero_to_lightgray)
    assert "[0.7, 0.7, 0.7, 1.0]" not in src, (
        "set_zero_to_lightgray still has a commented-out alternative color [0.7, 0.7, 0.7, 1.0]; "
        "remove it"
    )
