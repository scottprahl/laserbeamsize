"""
Tests for functions in analysis.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import laserbeamsize as lbs

interactive = True

h, v = 400, 400  # Image dimensions
xc, yc, dx, dy, phi = 200, 200, 100, 50, 0
image = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, max_value=4047)



def run_test(h, v, xc, yc, dx, dy, phi, noise, constant=False, max_value=255):
    if constant:
        test_img = image + noise
    else:
        test_img = lbs.image_tools.create_test_image(h, v, xc, yc, dx, dy, phi, noise=noise, max_value=max_value)
    result_xc, result_yc, result_dx, result_dy, result_phi = lbs.beam_size(test_img)
    erp = np.degrees(phi)
    rp = np.degrees(result_phi)
    
    if interactive:
        plt.title('result=%.1f° expected=%.1f°' % (rp, erp))
        plt.imshow(test_img)
        x, y = lbs.image_tools.ellipse_arrays(result_xc, result_yc, result_dx, result_dy, result_phi)
        plt.colorbar()
        plt.plot(x, y)
        plt.show()
    
    assert np.isclose(result_xc, xc, rtol=0.03), f"Expected xc around {xc}, but got {result_xc}"
    assert np.isclose(result_yc, yc, rtol=0.03), f"Expected yc around {yc}, but got {result_yc}"
    assert np.isclose(result_dx, dx, rtol=0.03), f"Expected dx around {dx}, but got {result_dx}"
    assert np.isclose(result_dy, dy, rtol=0.03), f"Expected dy around {dy}, but got {result_dy}"
    assert np.isclose(rp, erp, rtol=0.03), f"Expected phi around {phi}°, but got {result_phi}°"


def test_constant_noise_0():
    noise=0
    run_test(400, 400, 200, 200, 100, 50, 0, noise, constant=True)

def test_constant_noise_20():
    noise=20
    run_test(400, 400, 200, 200, 100, 50, 0, noise, constant=True)


def test_constant_noise_20a():
    noise=20
    run_test(400, 400, 200, 200, 100, 50, 0, noise, constant=True, max_value=4047)


def test_constant_noise_50():
    noise=50
    run_test(400, 400, 200, 200, 100, 50, 0, noise, constant=True)

def test_1_noise():
    noise=0
    run_test(400, 400, 200, 200, 100, 50, 0, noise)


def test_1_noise():
    noise=1
    run_test(400, 400, 200, 200, 100, 50, 0, noise)


def test_20_noise():
    noise=20
    run_test(400, 400, 200, 200, 100, 50, 0, noise)


def test_50_noise():
    noise=50
    run_test(400, 400, 200, 200, 100, 50, 0, noise)


def test_vertical_ellipse_no_rotation():
    run_test(400, 400, 200, 200, 50, 100, 0)


def test_horizontal_ellipse_off_center():
    run_test(400, 400, 150, 300, 50, 100, np.pi / 6)


# Running the tests
if __name__ == "__main__":
    test_horizontal_ellipse()
    test_vertical_ellipse_rotated()
    test_vertical_ellipse_negative_rotation()
    test_ellipse_120_degree_rotation()
    test_vertical_ellipse_no_rotation()
    test_horizontal_ellipse_off_center()
    print("All tests passed!")
