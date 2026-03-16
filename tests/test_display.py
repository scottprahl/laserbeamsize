"""Tests for display functions."""

# pylint: disable=wrong-import-position
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
