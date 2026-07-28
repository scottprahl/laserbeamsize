"""Tests for beam-size computations with fixed phi."""

import numpy as np
import laserbeamsize as lbs


def create_image(phi=np.pi / 6):
    """Helper for create image."""
    return lbs.image_tools.create_test_image(400, 400, 200, 200, 80, 40, phi)


def test_basic_beam_size_phi_returned():
    """Test basic beam size phi returned."""
    image = create_image()
    result = lbs.basic_beam_size(image, phi_fixed=0)
    assert np.isclose(result[4], 0)


def test_beam_size_phi_returned():
    """Test beam size phi returned."""
    image = create_image()
    result = lbs.beam_size(image, phi_fixed=-np.pi / 3)
    assert np.isclose(result[4], -np.pi / 3)


def _run_case(image, phi_arg, expected_xc, expected_yc, expect_major, expect_minor, expected_phi):
    """Helper for run case."""
    result = lbs.beam_size(image, phi_fixed=phi_arg)
    #     if phi_arg is None:
    #         print("phi fixed is none")
    #     else:
    #         print("phi fixed = %.0f°" % np.degrees(phi_arg))
    #     print("xc %.1f  %.1f expected" % (result[0], expected_xc))
    #     print("yc %.1f  %.1f expected" % (result[1], expected_yc))
    #     print("dj %.1f  %.1f expected" % (result[2], expect_major))
    #     print("dn %.1f  %.1f expected" % (result[3], expect_minor))
    #     print()
    assert np.isclose(result[0], expected_xc, rtol=0.01)
    assert np.isclose(result[1], expected_yc, rtol=0.01)
    assert np.isclose(result[2], expect_major, rtol=0.05)
    assert np.isclose(result[3], expect_minor, rtol=0.05)
    if phi_arg is None:
        assert np.isclose(result[4], expected_phi, atol=1e-2)
    else:
        assert np.isclose(result[4], phi_arg)


def _equivalent_diameter(major, minor):
    """Return the equivalent circular diameter for a forced 45-degree offset fit."""
    return np.sqrt((major**2 + minor**2) / 2)


def test_fixed_45_examples():
    # 45°
    """Test fixed 45 examples."""
    beam1 = lbs.create_test_image(h=600, v=600, xc_px=300, yc_px=300, d_major=150, d_minor=100, phi=np.pi / 4)
    w = np.sqrt((150**2 + 100**2) / 2)

    cases1 = [
        (None, 300, 300, 150, 100, np.pi / 4),
        (np.pi / 4, 300, 300, 150, 100, np.pi / 4),
        (0, 300, 300, w, w, 0),
        (3 * np.pi / 4, 300, 300, 150, 100, np.pi / 2),
        (-np.pi / 4, 300, 300, 150, 100, -np.pi / 4),
    ]
    for phi_arg, xc, yc, d_major, d_minor, phi in cases1:
        _run_case(beam1, phi_arg, xc, yc, d_major, d_minor, phi)


def test_fixed_phi_45_degree_offsets_produce_equivalent_diameter():
    """Forcing a fit 45 degrees away from the principal axes should produce equal diameters."""
    cases = [
        (lbs.create_test_image(h=600, v=600, xc_px=250, yc_px=250, d_major=150, d_minor=100, phi=np.pi / 4), np.pi / 4),
        (lbs.create_test_image(h=600, v=600, xc_px=250, yc_px=350, d_major=150, d_minor=100, phi=-np.pi / 6), -np.pi / 6),
    ]

    for beam, phi_true in cases:
        _, _, aligned_major, aligned_minor, _ = lbs.beam_size(beam, phi_fixed=phi_true)
        expected = _equivalent_diameter(aligned_major, aligned_minor)

        for phi_arg in (phi_true + np.pi / 4, phi_true - np.pi / 4):
            _, _, fitted_major, fitted_minor, fitted_phi = lbs.beam_size(beam, phi_fixed=phi_arg)

            assert fitted_minor is not None
            assert np.isclose(fitted_phi, phi_arg)
            assert np.isclose(fitted_major, fitted_minor, rtol=1e-4)
            assert np.isclose(fitted_major, expected, rtol=1e-4)
            assert np.isclose(fitted_minor, expected, rtol=1e-4)


def test_fixed_30_examples():
    # -30°
    """Test fixed 30 examples."""
    beam2 = lbs.create_test_image(h=600, v=600, xc_px=250, yc_px=350, d_major=150, d_minor=100, phi=-np.pi / 6)
    w = np.sqrt((150**2 + 100**2) / 2)

    cases2 = [
        (None, 250, 350, 150, 100, -np.pi / 6),
        (-np.pi / 6, 250, 350, 150, 100, -np.pi / 6),
        (-np.pi / 6 + np.pi / 4, 250, 350, w, w, -np.pi / 6 + np.pi / 4),
        (-np.pi / 6 - np.pi / 4, 250, 350, w, w, -np.pi / 6 - np.pi / 4),
        (np.pi / 3, 250, 350, 150, 100, np.pi / 3),
    ]
    for phi_arg, xc, yc, d_major, d_minor, phi in cases2:
        _run_case(beam2, phi_arg, xc, yc, d_major, d_minor, phi)
