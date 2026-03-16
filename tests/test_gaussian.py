"""Tests for Gaussian-beam helper calculations."""

import inspect

import numpy as np
import laserbeamsize as lbs
import laserbeamsize.gaussian as g


# Rayleigh distance
def test_z_rayleigh_positive_values():
    """Test z rayleigh positive values."""
    w0, lambda0 = 0.001, 1e-6
    result = lbs.z_rayleigh(w0, lambda0)
    assert np.isclose(result, np.pi)


def test_z_rayleigh_with_m2():
    """Test z rayleigh with m2."""
    w0, lambda0, M2 = 0.001, 1e-6, 2
    result = lbs.z_rayleigh(w0, lambda0, M2)
    assert np.isclose(result, np.pi / 2)


# beam radius
def test_beam_radius_standard_conditions():
    """Test beam radius standard conditions."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.002, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_z_equals_z0():
    """Test beam radius z equals z0."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.001, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_large_z():
    """Test beam radius large z."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.01, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_negative_z():
    """Test beam radius negative z."""
    w0, lambda0, z, z0 = 0.001, 1e-6, -0.001, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


def test_beam_radius_m2_factor():
    """Test beam radius m2 factor."""
    w0, lambda0, z, z0, M2 = 0.001, 1e-6, 0.002, 0.001, 2
    result = lbs.beam_radius(w0, lambda0, z, z0, M2)
    assert np.isclose(result, 0.001)


def test_beam_radius_varied_wavelength():
    """Test beam radius varied wavelength."""
    w0, lambda0, z, z0 = 0.001, 2e-6, 0.002, 0.001
    result = lbs.beam_radius(w0, lambda0, z, z0)
    assert np.isclose(result, 0.001)


# magnification
def test_magnification_standard_conditions():
    """Test magnification standard conditions."""
    w0, lambda0, s, f = 0.001, 1e-6, -0.01, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.006366165)


def test_magnification_positive_s():
    """Test magnification positive s."""
    w0, lambda0, s, f = 0.001, 1e-6, 0.01, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.006365907)


def test_magnification_negative_f():
    """Test magnification negative f."""
    w0, lambda0, s, f = 0.001, 1e-6, 0.01, -0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, -0.006366165)


def test_magnification_s_equals_negative_f():
    """Test magnification s equals negative f."""
    w0, lambda0, s, f = 0.001, 1e-6, -0.02, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.0063661977)


def test_magnification_m2_factor():
    """Test magnification m2 factor."""
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.01, 0.02, 2
    result = lbs.magnification(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.012732137)


def test_magnification_varied_wavelength():
    """Test magnification varied wavelength."""
    w0, lambda0, s, f = 0.001, 2e-6, -0.01, 0.02
    result = lbs.magnification(w0, lambda0, s, f)
    assert np.isclose(result, 0.012732137)


# curvature
def test_curvature_standard_conditions():
    """Test curvature standard conditions."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.002, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, 9869.6054)


def test_curvature_z_equals_z0():
    """Test curvature z equals z0."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.001, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    # radius of curvature should be infinite at the beam waist
    assert np.isclose(result, np.inf)


def test_curvature_large_z():
    """Test curvature large z."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.01, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, 1096.6317)


def test_curvature_negative_z():
    """Test curvature negative z."""
    w0, lambda0, z, z0 = 0.001, 1e-6, -0.001, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, -4934.8042)


def test_curvature_m2_factor():
    """Test curvature m2 factor."""
    w0, lambda0, z, z0, M2 = 0.001, 1e-6, 0.002, 0.001, 2
    result = lbs.curvature(w0, lambda0, z, z0, M2)
    assert np.isclose(result, 2467.4021)


def test_curvature_varied_wavelength():
    """Test curvature varied wavelength."""
    w0, lambda0, z, z0 = 0.001, 2e-6, 0.002, 0.001
    result = lbs.curvature(w0, lambda0, z, z0)
    assert np.isclose(result, 2467.4021)


# divergence
def test_divergence_standard_conditions():
    """Test divergence standard conditions."""
    w0, lambda0 = 0.001, 1e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.00063662)


def test_divergence_small_w0():
    """Test divergence small w0."""
    w0, lambda0 = 0.0001, 1e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.0063662)


def test_divergence_large_w0():
    """Test divergence large w0."""
    w0, lambda0 = 0.01, 1e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.00006366)


def test_divergence_varied_lambda0():
    """Test divergence varied lambda0."""
    w0, lambda0 = 0.001, 2e-6
    result = lbs.divergence(w0, lambda0)
    assert np.isclose(result, 0.00127323)


def test_divergence_m2_factor():
    """Test divergence m2 factor."""
    w0, lambda0, M2 = 0.001, 1e-6, 2
    result = lbs.divergence(w0, lambda0, M2)
    assert np.isclose(result, 0.00127323)


# Gouy phase
def test_gouy_phase_standard_conditions():
    """Test gouy phase standard conditions."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.002, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, -0.00031831)


def test_gouy_phase_at_beam_waist():
    """Test gouy phase at beam waist."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.001, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, 0)


def test_gouy_phase_past_rayleigh_distance():
    """Test gouy phase past rayleigh distance."""
    w0, lambda0, z, z0 = 0.001, 1e-6, 0.01, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, -0.00286478)


def test_gouy_phase_with_negative_z():
    """Test gouy phase with negative z."""
    w0, lambda0, z, z0 = 0.001, 1e-6, -0.001, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, 0.00063662)


def test_gouy_phase_varied_wavelength():
    """Test gouy phase varied wavelength."""
    w0, lambda0, z, z0 = 0.001, 2e-6, 0.002, 0.001
    result = lbs.gouy_phase(w0, lambda0, z, z0)
    assert np.isclose(result, -0.00063662)


# focused diameter
def test_focused_diameter_standard_conditions():
    """Test focused diameter standard conditions."""
    f, lambda0, d, M2 = 0.01, 1e-6, 0.002, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 6.3661977e-06)  # replace with the correct 8-digit accurate result


def test_focused_diameter_varied_focal_length():
    """Test focused diameter varied focal length."""
    f, lambda0, d, M2 = 0.02, 1e-6, 0.002, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 1.2732395e-05)  # replace with the correct 8-digit accurate result


def test_focused_diameter_varied_wavelength():
    """Test focused diameter varied wavelength."""
    f, lambda0, d, M2 = 0.01, 2e-6, 0.002, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 1.2732395e-05)  # replace with the correct 8-digit accurate result


def test_focused_diameter_varied_aperture_diameter():
    """Test focused diameter varied aperture diameter."""
    f, lambda0, d, M2 = 0.01, 1e-6, 0.004, 1
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(
        result,
        3.18309886e-06,
    )  # replace with the correct 8-digit accurate result


def test_focused_diameter_with_m2_factor():
    """Test focused diameter with m2 factor."""
    f, lambda0, d, M2 = 120e-3, 532e-9, 10e-3, 1.3
    result = lbs.focused_diameter(f, lambda0, d, M2)
    assert np.isclose(result, 0.010567e-03)  # replace with the correct 8-digit accurate result


# beam parameter product
def test_beam_parameter_product_standard_conditions():
    """Test beam parameter product standard conditions."""
    Theta, d0, Theta_std, d0_std = 0.01, 0.002, 0.0001, 0.00002
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 5e-06)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 7.071067e-08)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_zero_std_conditions():
    """Test beam parameter product zero std conditions."""
    Theta, d0, Theta_std, d0_std = 0.01, 0.002, 0, 0
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 5e-06)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 0.00000000)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_high_std_conditions():
    """Test beam parameter product high std conditions."""
    Theta, d0, Theta_std, d0_std = 0.01, 0.002, 0.005, 0.001
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 5e-06)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 3.5355339e-06)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_varied_divergence_angle():
    """Test beam parameter product varied divergence angle."""
    Theta, d0, Theta_std, d0_std = 0.02, 0.002, 0.0001, 0.00002
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 1e-05)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 1.118034e-07)  # replace with the correct 8-digit accurate result


def test_beam_parameter_product_varied_waist_diameter():
    """Test beam parameter product varied waist diameter."""
    Theta, d0, Theta_std, d0_std = 0.01, 0.004, 0.0001, 0.00002
    result, result_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    assert np.isclose(result, 1e-05)  # replace with the correct 8-digit accurate result
    assert np.isclose(result_std, 1.118034e-07)  # replace with the correct 8-digit accurate result


# image distance
def test_image_distance_default():
    """Test image distance default."""
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.05, 0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.05)


def test_image_distance_positive_s():
    """Test image distance positive s."""
    w0, lambda0, s, f, M2 = 0.001, 1e-6, 0.02, 0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.049982277591)


def test_image_distance_negative_f():
    """Test image distance negative f."""
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.05, -0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, -0.04997469)


def test_image_distance_with_M2_factor():
    """Test image distance with M2 factor."""
    w0, lambda0, s, f, M2 = 0.001, 1e-6, -0.05, 0.05, 2
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.050)


def test_image_distance_varied_wavelength():
    """Test image distance varied wavelength."""
    w0, lambda0, s, f, M2 = 0.001, 2e-6, -0.05, 0.05, 1
    result = lbs.image_distance(w0, lambda0, s, f, M2)
    assert np.isclose(result, 0.05)


def test_beam_parameter_product_zero_theta_returns_zero():
    """beam_parameter_product with Theta=0 should return zero BPP, not raise ZeroDivisionError."""
    BPP, BPP_std = lbs.beam_parameter_product(0, 0.002, 0, 0)
    assert np.isclose(BPP, 0)
    assert np.isclose(BPP_std, 0)


def test_beam_parameter_product_zero_d0_returns_zero():
    """beam_parameter_product with d0=0 should return zero BPP, not raise ZeroDivisionError."""
    BPP, BPP_std = lbs.beam_parameter_product(0.01, 0, 0, 0)
    assert np.isclose(BPP, 0)
    assert np.isclose(BPP_std, 0)


def test_artificial_to_original_no_placeholder():
    """artificial_to_original must not contain a placeholder comment."""
    src = inspect.getsource(g.artificial_to_original)
    assert "rest of your code" not in src, "artificial_to_original still has placeholder comment"


def test_divergence_docstring_notes_small_angle():
    """divergence() docstring should note the small-angle approximation used."""
    doc = g.divergence.__doc__ or ""
    assert (
        "small" in doc.lower() or "approximat" in doc.lower()
    ), "divergence() docstring should mention the small-angle approximation"


def test_beam_parameter_product_near_zero_theta_does_not_divide_by_zero():
    """beam_parameter_product with near-zero Theta must zero out BPP_std, not overflow."""
    # A subnormal float: not exactly zero but np.isclose(x, 0) should be True
    almost_zero = 1e-310
    BPP, BPP_std = lbs.beam_parameter_product(almost_zero, 0.002, 1e-310, 2e-5)
    assert np.isfinite(BPP), f"BPP is not finite: {BPP}"
    assert BPP_std == 0, (
        f"beam_parameter_product did not short-circuit for near-zero Theta={almost_zero}; "
        f"got BPP_std={BPP_std} (expected 0)"
    )
