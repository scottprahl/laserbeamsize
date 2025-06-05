import numpy as np
import laserbeamsize as lbs


# Rayleigh distance
def test_z_rayleigh_positive_values():
    w0, lambda0 = 0.001, 1e-6
    result = lbs.z_rayleigh(w0, lambda0)
    assert np.isclose(result, np.pi)


def test_z_rayleigh_with_m2():
    w0, lambda0, M2 = 0.001, 1e-6, 2
    result = lbs.z_rayleigh(w0, lambda0, M2)
    assert np.isclose(result, np.pi / 2)


# beam radius
def test_beam_radius_standard_conditions():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.002, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_z_equals_z0():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.001, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_large_z():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.01, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_negative_z():
    w0, lambda0, z, z0 = 0.001, 1e-6, -0.001, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_m2_factor():
    w0, lambda0, z, z0, M2 = 0.001, 1e-6, 0.002, 0.001, 2
    result = lbs.beam_radius(w0, lambda0, z, z0, M2)
    assert np.isclose(result, 0.001)


def test_beam_radius_varied_wavelength():
    w0, lambda0, z, z0 = 0.001, 2e-6, 0.002, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


# magnification
def test_magnification_standard_conditions():
    w0, lambda0, s, f = 0.001, 1e-6, -0.01, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.006366165)


def test_magnification_positive_s():
    w0, lambda0, s, f = 0.001, 1e-6, 0.01, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.006365907)


def test_magnification_negative_f():
    w0, lambda0, s, f = 0.001, 1e-6, 0.01, -0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, -0.006366165)


def test_magnification_s_equals_negative_f():
    w0, lambda0, s, f = 0.001, 1e-6, -0.02, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.0063661977)


def test_magnification_m2_factor():
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.01, 0.02, 2
    result = lbs.magnification(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.012732137)


def test_magnification_varied_wavelength():
    w0, lambda0, s, f = 0.001, 2e-6, -0.01, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.012732137)


# curvature
def test_curvature_standard_conditions():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.002, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, 9869.6054)


def test_curvature_z_equals_z0():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.001, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    # radius of curvature should be infinite at the beam waist
    assert np.isclose(result, np.inf)


def test_curvature_large_z():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.01, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, 1096.6317)


def test_curvature_negative_z():
    w0, lambda0, z, z0 = 0.001, 1e-6, -0.001, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, -4934.8042)


def test_curvature_m2_factor():
    w0, lambda0, z, z0, M2 = 0.001, 1e-6, 0.002, 0.001, 2
    result = lbs.curvature(w0, lambda0, z, z0, M2)
    assert np.isclose(result, 2467.4021)


def test_curvature_varied_wavelength():
    w0, lambda0, z, z0 = 0.001, 2e-6, 0.002, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, 2467.4021)


# divergence
def test_divergence_standard_conditions():
    w0, lambda0 = 0.001, 1e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.00063662)


def test_divergence_small_w0():
    w0, lambda0 = 0.0001, 1e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.0063662)


def test_divergence_large_w0():
    w0, lambda0 = 0.01, 1e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.00006366)


def test_divergence_varied_lambda0():
    w0, lambda0 = 0.001, 2e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.00127323)


def test_divergence_m2_factor():
    w0, lambda0, M2 = 0.001, 1e-6, 2
    result = lbs.divergence(w0, lambda0, M2)
    assert np.isclose(result, 0.00127323)


# Gouy phase
def test_gouy_phase_standard_conditions():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.002, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, -0.00031831)


def test_gouy_phase_at_beam_waist():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.001, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, 0)


def test_gouy_phase_past_rayleigh_distance():
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.01, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, -0.00286478)


def test_gouy_phase_with_negative_z():
    w0, lambda0, z, z0 = 0.001, 1e-6, -0.001, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, 0.00063662)


def test_gouy_phase_varied_wavelength():
    w0, lambda0, z, z0 = 0.001, 2e-6, 0.002, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, -0.00063662)


# focused diameter
def test_focused_diameter_standard_conditions():
    f, lambda0, d, M2 = 0.01, 1e-6, 0.002, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 6.3661977e-06)  # replace with the correct 8-digit accurate result


def test_focused_diameter_varied_focal_length():
    f, lambda0, d, M2 = 0.02, 1e-6, 0.002, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 1.2732395e-05)  # replace with the correct 8-digit accurate result


def test_focused_diameter_varied_wavelength():
    f, lambda0, d, M2 = 0.01, 2e-6, 0.002, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 1.2732395e-05)  # replace with the correct 8-digit accurate result


def test_focused_diameter_varied_aperture_diameter():
    f, lambda0, d, M2 = 0.01, 1e-6, 0.004, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(
        result,
        3.18309886e-06,
    )  # replace with the correct 8-digit accurate result


def test_focused_diameter_with_m2_factor():
    f, lambda0, d, M2 = 120e-3, 532e-9, 10e-3, 1.3
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 0.010567e-03)  # replace with the correct 8-digit accurate result


# beam parameter product
def test_beam_parameter_product_standard_conditions():
    Theta, d0, Theta_std, d0_std = 0.01, 0.002, 0.0001, 0.00002
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 5e-06)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 7.071067e-08)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_zero_std_conditions():
    Theta, d0, Theta_std, d0_std = 0.01, 0.002, 0, 0
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 5e-06)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 0.00000000)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_high_std_conditions():
    Theta, d0, Theta_std, d0_std = 0.01, 0.002, 0.005, 0.001
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 5e-06)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 3.5355339e-06)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_varied_divergence_angle():
    Theta, d0, Theta_std, d0_std = 0.02, 0.002, 0.0001, 0.00002
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 1e-05)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 1.118034e-07)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_varied_waist_diameter():
    Theta, d0, Theta_std, d0_std = 0.01, 0.004, 0.0001, 0.00002
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 1e-05)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 1.118034e-07)  # replace with the correct 8-digit accurate result


# image distance
def test_image_distance_default():
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.05, 0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.05)


def test_image_distance_positive_s():
    w0, lambda0, s, f, M2 = 0.001, 1e-6, 0.02, 0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.049982277591)


def test_image_distance_negative_f():
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.05, -0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, -0.04997469)


def test_image_distance_with_M2_factor():
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.05, 0.05, 2
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.050)


def test_image_distance_varied_wavelength():
    w0, lambda0, s, f, M2 = 0.001, 2e-6, -0.05, 0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.05)
