"""
Tests for functions in analysis.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import laserbeamsize as lbs

interactive = True


def run_test(h, v, xc, yc, dx, dy, phi, noise=0, ntype='poisson', max_value=255):

    test_img = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, 
                                                 noise=noise,
                                                 ntype=ntype,
                                                 max_value=max_value)

    result_xc, result_yc, result_dx, result_dy, result_phi = lbs.beam_size(test_img)
    erp = np.degrees(phi)
    rp = np.degrees(result_phi)
    
    if interactive:
        plt.title('noise=%.1f pixels, type=%s' % (noise, ntype))
        plt.imshow(test_img)
        x, y = lbs.image_tools.ellipse_arrays(result_xc, result_yc, result_dx, result_dy, result_phi)
        plt.colorbar()
        plt.plot(x, y)
        plt.show()
    
    assert np.isclose(result_xc, xc, rtol=0.03), f"Expected xc around {xc}, but got {result_xc}"
    assert np.isclose(result_yc, yc, rtol=0.03), f"Expected yc around {yc}, but got {result_yc}"
    assert np.isclose(result_dx, dx, rtol=0.05), f"Expected dx around {dx}, but got {result_dx}"
    assert np.isclose(result_dy, dy, rtol=0.05), f"Expected dy around {dy}, but got {result_dy}"
    assert np.isclose(rp, erp, rtol=0.03), f"Expected phi around {phi}°, but got {result_phi}°"


def test_constant_noise_0():
    noise=0
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='constant')


def test_constant_noise_20():
    noise=20
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='constant')


def test_constant_noise_50():
    noise=50
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='constant')


def test_constant_noise_50a():
    noise=50
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='constant', max_value=4047)


def test_poisson_noise_0():
    noise=0
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='poisson')


def test_poisson_noise_5():
    noise=5
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='poisson')


def test_poisson_noise_20():
    noise=20
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='poisson')


def test_poisson_noise_20a():
    noise=20
    run_test(400, 400, 200, 200, 100, 50, 0, noise, ntype='poisson', max_value=4047)


# Running the tests
if __name__ == "__main__":
    test_horizontal_ellipse()
    test_vertical_ellipse_rotated()
    test_vertical_ellipse_negative_rotation()
    test_ellipse_120_degree_rotation()
    test_vertical_ellipse_no_rotation()
    test_horizontal_ellipse_off_center()
    print("All tests passed!")
