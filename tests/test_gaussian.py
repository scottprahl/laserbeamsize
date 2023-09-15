import numpy as np
import laserbeamsize as lbs

def test_z_rayleigh_positive_values():
    w0, lambda0 = 0.001, 1e-6
    result = lbs.z_rayleigh(w0, lambda0)
    assert np.isclose(result, np.pi)

def test_z_rayleigh_with_m2():
    w0, lambda0, M2 = 0.001, 1e-6, 2
    result = lbs.z_rayleigh(w0, lambda0, M2)
    assert np.isclose(result, np.pi / 2)

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
