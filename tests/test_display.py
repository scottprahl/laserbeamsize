"""Tests for display functions."""

import matplotlib.pyplot as plt
import laserbeamsize as lbs


def test_plot_image_analysis_title_uses_pixels_by_default():
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
