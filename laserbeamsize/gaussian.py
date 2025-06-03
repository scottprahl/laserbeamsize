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

import numpy as np

__all__ = (
    "z_rayleigh",
    "beam_radius",
    "magnification",
    "image_distance",
    "curvature",
    "divergence",
    "gouy_phase",
    "focused_diameter",
    "beam_parameter_product",
    "artificial_to_original",
)


def z_rayleigh(w0, lambda0, M2=1):
    """
    Return the Rayleigh distance for a Gaussian beam.

    The Rayleigh distance is important not only because it marks a specific
    point in the evolution of the beam radius but also because it fundamentally
    characterizes the propagation of Gaussian beams.

    The Rayleigh distance (zR) is the distance from the beam waist (the location
    where the beam radius is minimum) to the point where the beam radius has
    increased by a factor of √2, and consequently, the beam's cross-sectional
    area and the irradiance (power per unit area) has doubled and halved,
    respectively. At this point, the beam radius is √2 times the minimum beam
    radius (w0).

    The region within ±zR of the beam waist represents the depth of focus where
    the beam maintains a nearly constant radius. It fundamentally delineates the
    region where the beam can be considered to be "focused."

    The Rayleigh distance has a direct relationship with the numerical aperture
    (NA) and the focal spot size. Specifically, NA is inversely proportional to
    the Rayleigh distance, given by NA ≈ w0/zR. Moreover, the Rayleigh distance
    influences the focal spot size at the focal plane through the lens's focal
    length, defining the resolution limit in imaging systems and determining the
    efficiency of energy transfer in various optical setups.

    By measuring the beam radius at several points along the propagation
    direction and using the Rayleigh distance, one can locate the position of
    the beam waist precisely.  This is what the `laserbeamsize.m2` module does.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        M2: beam propagation factor [-]

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

    Here "magnification" refers to the scaling of the beam waist and the
    Rayleigh range as the beam propagates through a lens. Essentially, it tells
    you how the smallest cross-sectional area of the beam (beam waist) and the
    distance over which the beam expands before and after the waist (Rayleigh
    range) scale due to the lens.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        s: distance of beam waist to lens [m]
        f: focal distance of lens [m]
        M2: beam propagation factor [-]

    Returns:
        m: magnification [-]
    """
    zR2 = z_rayleigh(w0, lambda0, M2) ** 2
    return f / np.sqrt((s + f) ** 2 + zR2)


def curvature(w0, lambda0, z, z0=0, M2=1):
    """
    Calculate the radius of curvature of a Gaussian beam at a given axial position.

    The radius of curvature of a Gaussian beam refers to the curvature of the
    beam's wavefronts as it propagates in space. At the beam waist, where the
    beam is at its narrowest, the radius of curvature is infinite, indicating
    that the wavefronts are planar.

    As we move away from the beam waist, the radius of curvature starts to
    decrease, transitioning from infinite to a finite value. When we reach the
    Rayleigh distance `z = z_0 ± z_R`, the radius of curvature equals the
    Rayleigh distance. Past the Rayleigh distance, the curvature continues to
    decrease, eventually going to zero as `z` goes to infinity, meaning the
    beam becomes a plane wave at infinity. The point at which the
    radius of curvature is equal to the Rayleigh distanceis where we can say
    the curvature is "most pronounced". After this point, the beam begins to diverge,
    and its radius of curvature continually decreases, following the inverse of the axial
    distance.

    The beam propagation factor, M², plays a crucial role in determining
    the beam's divergence and, consequently, its radius of curvature. A beam
    with M² = 1 is an ideal Gaussian beam, exhibiting the minimum possible
    divergence for a given waist size. As M² increases above 1, the beam
    diverges more, indicating a decrease in beam quality.

    Args:
        w0:      minimum beam radius [m]
        lambda0: wavelength of light [m]
        z:       axial position along beam  [m]
        z0:      axial position of the beam waist  [m]
        M2:      beam propagation factor [-]

    Returns:
        R: radius of curvature of field at z [m]
    """
    if z == z0:
        return np.inf
    zR2 = z_rayleigh(w0, lambda0, M2) ** 2
    return (z - z0) + zR2 / (z - z0)


def divergence(w0, lambda0, M2=1):
    """
    Calculate the full angle of divergence of a Gaussian beam.

    The beam divergence refers to the increase in the beam diameter as the beam
    propagates away from the beam waist (minimum beam radius). In Gaussian beam
    optics, it's characterized by a half-angle at which the beam radius
    increases with the distance from the optical axis.

    The limiting case occurs when the beam propagation factor (M²) equals 1,
    representing a perfect Gaussian beam. In this case, the divergence is solely
    dependent on the wavelength of the light and the minimum beam radius,
    leading to the smallest possible divergence angle for the given beam
    characteristics.

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
    Calculate the Gouy phase shift of a Gaussian beam at a specific axial position.

    The Gouy phase shift is an additional phase shift experienced by Gaussian beams,
    besides the normal propagation phase change. This phenomenon is prominent near the
    beam waist. The function utilizes the arctangent function to calculate this phase shift
    at the specified axial position.

    The beam waist location z0 defaults to 0, implying that the calculation is done
    assuming the beam waist is at the origin.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        z: axial position where Gouy phase is calculated  [m]
        z0: axial position of beam waist  [m]

    Returns:
        phase: Gouy phase at the specified axial position [radians]
    """
    zR = z_rayleigh(w0, lambda0)
    return -np.arctan2(z - z0, zR)


def focused_diameter(f, lambda0, d, M2=1):
    """
    Calculate the diameter of a diffraction-limited focused beam.

    This function computes the diameter of a beam at its focus, assuming it is
    diffraction-limited, using the formula described in equation 6b from the
    paper by Roundy in "Current Technology of Beam Profile Measurements," found
    in "Laser Beam Shaping: Theory and Techniques" edited by Dickey in 2000.

    Args:
        f: focal length of lens [m]
        lambda0: wavelength of light [m]
        d: diameter of limiting aperture [m]
        M2: beam propagation factor [-]

    Returns:
        d: diffraction-limited beam diameter [m]
    """
    return 4 * M2 * lambda0 * f / (np.pi * d)


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
    BPP_std = BPP * np.sqrt((Theta_std / Theta) ** 2 + (d0_std / d0) ** 2)
    return BPP, BPP_std


def image_distance(w0, lambda0, s, f, M2=1):
    """
    Return the location of the new beam waist after passing through a lens.

    This gives the location where the beam will be most focused after passing through
    the lens. This is also known as the image location of a Gaussian beam. The calculation
    takes into account the beam's initial properties and the properties of the
    lens to determine where the new beam waist will form. By default,
    the function assumes that the beam waist is located at the front focus
    of the lens, corresponding to a distance s = -f.

    The beam propagation factor, M², affects the divergence angle and the Rayleigh
    range of the beam, which in turn affects the location of the new beam waist.
    A beam with M² > 1 will have a larger divergence angle and a shorter Rayleigh
    range compared to a diffraction-limited beam with M² = 1.

    Args:
        w0: The minimum radius of the initial beam, measured at the beam waist [m].
        lambda0: The wavelength of the light constituting the beam [m].
        s: The axial distance from the beam waist to the lens [m].
        f: The focal length of the lens [m].
        M2: The beam propagation factor [-]

    Returns:
        The axial distance from the lens to the location of the new beam waist [m]
    """
    zR2 = z_rayleigh(w0, lambda0, M2) ** 2
    return f * (s * f + s * s + zR2) / ((f + s) ** 2 + zR2)


def artificial_to_original(params, errors, f, hiatus=0):
    """
    Convert artificial beam parameters to original beam parameters.

    This function uses the equations provided in section 9 of ISO 11146-1 to
    convert beam parameters measured at an artificial waist, created by focusing
    the beam with a lens, back to the original beam parameters before the lens
    was applied.

    The beam quality factor (M²) remains unchanged during this conversion. It is
    ideal to have the waist position relative to the rear principal plane of the
    lens, and to correct the original beam waist position by the hiatus, which
    is the distance between the principal planes of the lens.

    This function currently returns errors that might not be accurate.

    Args:
        params: A list or array containing artificial beam parameters in the following order:
            - d0: beam waist diameter [m]
            - z0: axial location of beam waist [m]
            - Theta: full beam divergence angle [radians]
            - M2: beam propagation parameter [-]
            - zR: Rayleigh distance [m]

        errors (list or array): A list or array containing the standard deviations of
        the artificial beam parameters.

        f (float): The focal length of the lens used to create the artificial waist [m].

        hiatus (float, optional): The distance between the principal planes of the
        focusing lens [m]. Defaults to 0.

    Returns:
        tuple: A tuple containing two lists:
            - params: A list of the original beam parameters without the lens.
            - errors: A list of the standard deviations of the original parameters.
    """
    # ... (rest of your code)
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
