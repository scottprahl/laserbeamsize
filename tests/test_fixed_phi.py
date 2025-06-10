import numpy as np
import laserbeamsize as lbs


def create_image(phi=np.pi / 6):
    return lbs.image_tools.create_test_image(400, 400, 200, 200, 80, 40, phi)


def test_basic_beam_size_phi_returned():
    image = create_image()
    result = lbs.basic_beam_size(image, phi_fixed=0)
    assert np.isclose(result[4], 0)


def test_beam_size_phi_returned():
    image = create_image()
    result = lbs.beam_size(image, phi_fixed=-np.pi / 3)
    assert np.isclose(result[4], -np.pi / 3)


def _run_case(image, phi_arg, expected_xc, expected_yc, expect_major, expect_minor, expected_phi):
    result = lbs.beam_size(image, phi_fixed=phi_arg)
    assert np.isclose(result[0], expected_xc, rtol=0.01)
    assert np.isclose(result[1], expected_yc, rtol=0.01)
    assert np.isclose(result[2], expect_major, rtol=0.05)
    assert np.isclose(result[3], expect_minor, rtol=0.05)
    if phi_arg is None:
        assert np.isclose(result[4], expected_phi, atol=1e-2)
    else:
        assert np.isclose(result[4], phi_arg)


def test_fixed_45_examples():
    # 45°
    beam1 = lbs.create_test_image(h=600, v=600, xc=300, yc=300, d_major=150, d_minor=100, phi=np.pi / 4)
    w = np.sqrt((150**2 + 100**2) / 2)

    cases1 = [
        (None, 300, 300, 150, 100, np.pi / 4),
        (np.pi / 4, 300, 300, 150, 100, np.pi / 4),
        (0, 300, 300, w, w, 0),
        (3 * np.pi / 4, 300, 300, 100, 150, np.pi / 2),
        (-np.pi / 4, 300, 300, 100, 150, -np.pi / 4),
    ]
    for phi_arg, xc, yc, d_major, d_minor, phi in cases1:
        _run_case(beam1, phi_arg, xc, yc, d_major, d_minor, phi)


def test_fixed_30_examples():
    # -30°
    beam2 = lbs.create_test_image(h=600, v=600, xc=250, yc=350, d_major=150, d_minor=100, phi=-np.pi / 6)
    w = np.sqrt((150**2 + 100**2) / 2)

    cases2 = [
        (None, 250, 350, 150, 100, -np.pi / 6),
        (-np.pi / 6, 250, 350, 150, 100, -np.pi / 6),
        (-np.pi / 6 + np.pi / 4, 250, 350, w, w, -np.pi / 6 + np.pi / 4),
        (-np.pi / 6 - np.pi / 4, 250, 350, w, w, -np.pi / 6 - np.pi / 4),
        (np.pi / 3, 250, 350, 100, 150, np.pi / 3),
    ]
    for phi_arg, xc, yc, d_major, d_minor, phi in cases2:
        _run_case(beam2, phi_arg, xc, yc, d_major, d_minor, phi)
