import numpy as np
import laserbeamsize as lbs


def create_image(phi=np.pi / 6):
    return lbs.image_tools.create_test_image(400, 400, 200, 200, 80, 40, phi)


def test_basic_beam_size_phi_returned():
    image = create_image()
    result = lbs.basic_beam_size(image, phi=0)
    assert np.isclose(result[4], 0)


def test_basic_beam_size_naive_phi_returned():
    image = create_image().astype(float)
    result = lbs.analysis.basic_beam_size_naive(image, phi=np.pi / 4)
    assert np.isclose(result[4], np.pi / 4)


def test_beam_size_phi_returned():
    image = create_image()
    result = lbs.beam_size(image, phi=-np.pi / 3)
    assert np.isclose(result[4], -np.pi / 3)
