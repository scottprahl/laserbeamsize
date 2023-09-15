# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=consider-using-f-string)
# pylint: disable=too-many-lines
"""
A module for calculating properties of a Gaussian laser beam.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

This module contains a collection of functions to calculate various properties
of a Gaussian laser beam, based on its physical parameters such as waist radius,
wavelength, and beam propagation factor (M²). These functions are essential
in laser physics and optics to analyze and predict the behavior of laser beams
under different circumstances.

Functions:
    z_rayleigh(w0, lambda0, M2=1): Calculates the Rayleigh range of the beam.

    beam_radius(w0, lambda0, z, z0=0, M2=1): Determines the beam radius at a specific
    axial location.

    magnification(w0, lambda0, s, f, M2=1): Computes the magnification of the beam given
    certain parameters.

    curvature(w0, lambda0, z, z0=0, M2=1): Calculates the radius of curvature of the beam
    at a specified axial position.

    divergence(w0, lambda0, M2=1): Finds the full angle of beam divergence.

    gouy_phase(w0, lambda0, z, z0=0): Determines the Gouy phase of the beam at a
    particular axial location.

    focused_diameter(f, lambda0, d, M2=1): Calculates the diameter of a
    diffraction-limited focused beam.

    beam_parameter_product(Theta, d0, Theta_std=0, d0_std=0): Computes the beam parameter
    product and its standard deviation.

    image_distance(w0, lambda0, s, f, M2=1): Finds the location of the new beam waist
    after passing through a lens.

All functions take floating-point numbers as inputs for the physical parameters of the
beam, and return a floating-point number as the output, representing the calculated
property. The unit for distances is meters, and the unit for angles is radians.

Example:
    >>> import laserbeamsize as lbs
    >>> w0 = 0.001
    >>> lambda0 = 0.65e-6
    >>> z = lbs.z_rayleigh(w0, lambda0)
    >>> print(f"Rayleigh range: {z} meters")
    Rayleigh range: 4.811823616524697 meters

"""

# (your existing function definitions)

import numpy as np

__all__ = ('z_rayleigh',
           'beam_radius',
           'magnification',
           'image_distance',
           'curvature',
           'divergence',
           'gouy_phase',
           'focused_diameter',
           'beam_parameter_product',
           'artificial_to_original',
           )


def z_rayleigh(w0, lambda0, M2=1):
    """
    Return the Rayleigh distance for a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
    Returns:
        z: axial distance from focus that irradiance has dropped 50% [m]
    """
    return np.pi * w0**2 / lambda0 / M2


def beam_radius(w0, lambda0, z, z0=0, M2=1):
    """
    Return the beam radius at an axial location of a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        z: axial location of desired beam radius [m]
        z0: axial location of beam waist [m]
        M2: beam propagation factor [-]
    Returns:
        r: beam radius at axial position [m]
    """
    zz = (z - z0) / z_rayleigh(w0, lambda0, M2)
    return w0 * np.sqrt(1 + zz**2)


def magnification(w0, lambda0, s, f, M2=1):
    """
    Return the magnification of a Gaussian beam.

    If the beam waist is before the lens, then the distance s
    will be negative, i.e. if it is at the front focus of the lens (s=-f).

    The new beam waist will be `m * w0` and the new Rayleigh
    distance will be `m**2 * zR`

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        s: distance of beam waist to lens [m]
        f: focal distance of lens [m]
        M2: beam propagation factor [-]

    Returns:
        m: magnification [-]
    """
    zR2 = z_rayleigh(w0, lambda0, M2)**2
    return f / np.sqrt((s + f)**2 + zR2)


def curvature(w0, lambda0, z, z0=0, M2=1):
    """
    Calculate the radius of curvature of a Gaussian beam.

    The curvature will be a maximum at the Rayleigh distance and
    it will be infinite at the beam waist.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        z   axial position along beam  [m]
        z0  axial position of the beam waist  [m]
        M2: beam propagation factor [-]
    Returns:
        R: radius of curvature of field at z [m]
    """
    zR2 = z_rayleigh(w0, lambda0, M2)**2
    return (z - z0) + zR2 / (z - z0)


def divergence(w0, lambda0, M2=1):
    """
    Calculate the full angle of divergence of a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        M2: beam propagation factor [-]
    Returns:
        theta: divergence of beam [radians]
    """
    return 2 * w0 / z_rayleigh(w0, lambda0, M2)


def gouy_phase(w0, lambda0, z, z0=0):
    """
    Calculate the Gouy phase of a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        z: axial position along beam  [m]
        z0: axial position of beam waist  [m]
    Returns:
        phase: Gouy phase at axial position [radians]
    """
    zR = z_rayleigh(w0, lambda0)
    return -np.arctan2(z - z0, zR)


def focused_diameter(f, lambda0, d, M2=1):
    """
    Diameter of diffraction-limited focused beam.

    see eq 6b from Roundy, "Current Technology of Beam Profile Measurements"
    in Laser Beam Shaping: Theory and Techniques by Dickey, 2000

    Args:
        f: focal length of lens [m]
        lambda0: wavelength of light [m]
        d: diameter of limiting aperture [m]
        M2: beam propagation factor [-]
    Returns:
        d: diffraction-limited beam diameter [m]
    """
    return 4 * M2**2 * lambda0 * f / (np.pi * d)


def beam_parameter_product(Theta, d0, Theta_std=0, d0_std=0):
    """
    Find the beam parameter product (BPP).

    Better beam quality is associated with the lower BPP values. The best
    (smallest) BPP is λ / π and corresponds to a diffraction-limited Gaussian beam.

    Args:
        Theta: full beam divergence angle [radians]
        d0: beam waist diameter [m]
        Theta_std: std. dev. of full beam divergence angle [radians]
        d0_std: std. dev. of beam waist diameter [m]
    Returns:
        BPP: Beam parameter product [m * radian]
        BPP_std: standard deviation of beam parameter product [m * radian]
    """
    BPP = Theta * d0 / 4
    BPP_std = BPP * np.sqrt((Theta_std / Theta)**2 + (d0_std / d0)**2)
    return BPP, BPP_std


def image_distance(w0, lambda0, s, f, M2=1):
    """
    Return the image location of a Gaussian beam.

    The default case is when the beam waist is located at
    the front focus of the lens (s=-f).

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        s: distance of beam waist to lens [m]
        f: focal distance of lens [m]
        M2: beam propagation factor [-]
    Returns:
        z: location of new beam waist [m]
    """
    zR2 = z_rayleigh(w0, lambda0, M2)**2
    return f * (s * f + s * s + zR2) / ((f + s)**2 + zR2)


def artificial_to_original(params, errors, f, hiatus=0):
    """
    Convert artificial beam parameters to original beam parameters.

    ISO 11146-1 section 9 equations are used to retrieve the original beam
    parameters from parameters measured for an artificial waist
    created by focusing the beam with a lens.

    M2 does not change.

    Ideally, the waist position would be relative to the rear principal
    plane of the lens and the original beam waist position would be corrected
    by the hiatus between the principal planes of the lens.

    The beam parameters are in an array `[d0,z0,Theta,M2,zR]` ::

        d0: beam waist diameter [m]
        z0: axial location of beam waist [m]
        Theta: full beam divergence angle [radians]
        M2: beam propagation parameter [-]
        zR: Rayleigh distance [m]

    The errors that are returned are not quite right at the moment.

    Args:
        params: array of artificial beam parameters
        errors: array with std dev of above parameters
        f: focal length of lens [m]
        hiatus: distance between principal planes of focusing lens [m]

    Returns:
        params: array of original beam parameters (without lens)
        errors: array of std deviations of above parameters
    """
    art_d0, art_z0, art_Theta, M2, art_zR = params
    art_d0_std, art_z0_std, art_Theta_std, M2_std, art_zR_std = errors

    x2 = art_z0 - f
    V = f / np.sqrt(art_zR**2 + x2**2)

    orig_d0 = V * art_d0
    orig_d0_std = V * art_d0_std

    orig_z0 = V**2 * x2 + f - hiatus
    orig_z0_std = V**2 * art_z0_std

    orig_zR = V**2 * art_zR
    orig_zR_std = V**2 * art_zR_std

    orig_Theta = art_Theta / V
    orig_Theta_std = art_Theta_std / V

    o_params = [orig_d0, orig_z0, orig_Theta, M2, orig_zR]
    o_errors = [orig_d0_std, orig_z0_std, orig_Theta_std, M2_std, orig_zR_std]
    return o_params, o_errors
