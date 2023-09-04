"""
A basic test case.
"""

import numpy as np
import laserbeamsize as lbs

def test_beam_size():
    h, v = 400, 400  # Image dimensions

    # Test Case 1: Horizontal Ellipse
    xc, yc, dx, dy, phi = 200, 200, 100, 50, 0
    test_img = create_test_image(h, v, xc, yc, dx, dy, phi)
    result_xc, result_yc, result_dx, result_dy, result_phi = lbs.beam_size(test_img)
    assert abs(result_xc - xc) < 5, f"Expected xc around {xc}, but got {result_xc}"
    assert abs(result_yc - yc) < 5, f"Expected yc around {yc}, but got {result_yc}"
    assert abs(result_dx - dx) < 5, f"Expected dx around {dx}, but got {result_dx}"
    assert abs(result_dy - dy) < 5, f"Expected dy around {dy}, but got {result_dy}"

    # Test Case 2: Vertical Ellipse with a rotation
    xc, yc, dx, dy, phi = 200, 200, 50, 100, np.pi / 4
    test_img = create_test_image(h, v, xc, yc, dx, dy, phi)
    result_xc, result_yc, result_dx, result_dy, result_phi = lbs.beam_size(test_img)
    assert abs(result_xc - xc) < 5, f"Expected xc around {xc}, but got {result_xc}"
    assert abs(result_yc - yc) < 5, f"Expected yc around {yc}, but got {result_yc}"
    assert abs(result_dx - dx) < 5, f"Expected dx around {dx}, but got {result_dx}"
    assert abs(result_dy - dy) < 5, f"Expected dy around {dy}, but got {result_dy}"

    # You can continue with more test cases...

    print("All tests passed!")


# Running the tests
if __name__ == "__main__":
    test_beam_size()
