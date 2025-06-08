# pylint: disable=unbalanced-tuple-unpacking
"""
A module for finding M² values for a laser beam.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

This module contains the function `M2_fit()` designed to determine the beam parameters
and their respective errors based on a set of measurements as per ISO 11146-1 section 9,
utilizing a non-linear curve fitting method. The function calculates beam parameters such as:
- Beam waist diameter (d0) [m]
- Axial location of the beam waist (z0) [m]
- Full beam divergence angle (Theta) [radians]
- Beam propagation parameter (M2) [-]
- Rayleigh distance (zR) [m]

    >>> import numpy as np
    >>> import laserbeamsize as lbs
    >>>
    >>> lambda0 = 632.8e-9  # meters
    >>> z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770]) * 1e-3
    >>> r = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397]) * 1e-6
    >>> params, errors, _ = lbs.M2_fit(z, 2 * r, lambda0)

To facilitate interpretation of the results, there is also a `M2_report` function.
    Example::

    >>> import numpy as np
    >>> import laserbeamsize as lbs
    >>>
    >>> lambda0 = 632.8e-9  # meters
    >>> z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770]) * 1e-3
    >>> r = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397]) * 1e-6
    >>>
    >>> s = lbs.M2_report(z, 2 * r, lambda0)
    >>> print(s)

"""

import numpy as np
import scipy.optimize
import laserbeamsize as lbs

__all__ = (
    "M2_fit",
    "M2_report",
)


def _beam_fit_fn_1(z, d0, z0, Theta):
    """
    Fitting function d0, z0, and Theta.

    Args:
        z: position on optical axis [m]
        d0: beam waist diameter  [m]
        z0: axial location of beam waist [m]
        Theta: full beam divergence angle [radians]

    Returns:
        beam diameter
    """
    return np.sqrt(d0**2 + (Theta * (z - z0)) ** 2)


def _beam_fit_fn_2(z, d0, Theta):
    """
    Fitting function for d0 and Theta.

    The axial location of the beam waist is zero.

    Args:
        z: position on optical axis [m]
        d0: beam waist diameter  [m]
        Theta: full beam divergence angle [radians]

    Returns:
        beam diameter
    """
    return np.sqrt(d0**2 + (Theta * z) ** 2)


def _beam_fit_fn_3(z, z0, Theta):
    """
    Fitting function for z0 and Theta.

    The beam waist is assumed to be zero.

    Args:
        z: position on optical axis [m]
        z0: axial location of beam waist [m]
        Theta: full beam divergence angle [radians]

    Returns:
        beam diameter
    """
    return np.abs(Theta * (z - z0))


def _beam_fit_fn_4(z, Theta):
    """
    Fitting function for Theta.

    The beam waist and axial location are both assumed to be zero.

    Args:
        z: position on optical axis [m]
        Theta: full beam divergence angle [radians]

    Returns:
        beam diameter
    """
    return np.abs(Theta * z)


def basic_beam_fit(z, d, lambda0, z0=None, d0=None):
    """
    Return the hyperbolic fit to the supplied diameters.

    Follows ISO 11146-1 section 9 but `a`, `b`, and `c` have been
    replaced by beam parameters `d0`, `z0`, and Theta.  The equation
    for the beam diameter `d(z)` is

    d(z)**2 = d0**2 + Theta**2 * (z-z0)**2

    A non-linear curve fit is done to determine the beam parameters and the
    standard deviations of those parameters.  The beam parameters are returned
    in one array and the errors in a separate array::

        Theta: full beam divergence angle [radians]
        M2: beam propagation parameter [-]
        zR: Rayleigh distance [m]

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]
        z0: (optional) axial location of beam waist [m]
        d0: (optional) beam waist diameter [m]

    Returns:
        params: [d0, z0, Theta, M2, zR]
        errors: array with standard deviations of above values
    """
    # approximate answer
    i = np.argmin(d)
    d0_guess = d[i]
    z0_guess = z[i]

    # fit data using SciPy's curve_fit() algorithm
    if z0 is None:
        if d0 is None:
            i = np.argmax(abs(z - z0_guess))
            theta_guess = abs(d[i] / (z[i] - z0_guess))
            p0 = [d0_guess, z0_guess, theta_guess]
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_1, z, d, p0=p0)
            d0, z0, Theta = nlfit
            d0_std, z0_std, Theta_std = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]
        else:
            i = np.argmax(abs(z - z0_guess))
            theta_guess = abs(d[i] / (z[i] - z0_guess))
            p0 = [z0_guess, theta_guess]
            dd = np.sqrt(d**2 - d0**2)
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_3, z, dd, p0=p0)
            z0, Theta = nlfit
            z0_std, Theta_std = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]
            d0_std = 0
    else:
        i = np.argmax(abs(z - z0))
        theta_guess = abs(d[i] / (z[i] - z0))
        if d0 is None:
            p0 = [d0_guess, theta_guess]
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_2, z - z0, d, p0=p0)
            d0, Theta = nlfit
            d0_std, Theta_std = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]
            z0_std = 0
        else:
            p0 = [theta_guess]
            dd = np.sqrt(d**2 - d0**2)
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_4, z - z0, dd, p0=p0)
            Theta = nlfit[0]
            Theta_std = np.sqrt(nlpcov[0, 0])
            z0_std = 0
            d0_std = 0

    # divergence and Rayleigh range of Gaussian beam
    Theta0 = 4 * lambda0 / (np.pi * d0)
    zR = np.pi * d0**2 / (4 * lambda0)

    M2 = Theta / Theta0
    zR = np.pi * d0**2 / (4 * lambda0 * M2)

    M2_std = M2 * np.sqrt((Theta_std / Theta) ** 2 + (d0_std / d0) ** 2)
    zR_std = zR * np.sqrt((M2_std / M2) ** 2 + (2 * d0_std / d0) ** 2)

    params = [d0, z0, Theta, M2, zR]
    errors = [d0_std, z0_std, Theta_std, M2_std, zR_std]
    return params, errors


def max_index_in_focal_zone(z, zone):
    """Return index farthest from focus in inner zone."""
    _max = -1e32
    imax = None
    for i, zz in enumerate(z):
        if zone[i] == 1:
            if _max < zz:
                _max = zz
                imax = i
    return imax


def min_index_in_outer_zone(z, zone):
    """Return index of measurement closest to focus in outer zone."""
    _min = 1e32
    imin = None
    for i, zz in enumerate(z):
        if zone[i] == 2:
            if zz < _min:
                _min = zz
                imin = i
    return imin


def M2_fit(z, d, lambda0, strict=False, z0=None, d0=None):
    """
    Return the hyperbolic fit to the supplied diameters.

    Follows ISO 11146-1 section 9 but `a`, `b`, and `c` have been
    replaced by beam parameters `d0`, `z0`, and Theta.  The equation
    for the beam diameter `d(z)` is

    d(z)**2 = d0**2 + Theta**2 * (z - z0)**2

    A non-linear curve fit is done to determine the beam parameters and the
    standard deviations of those parameters.  The beam parameters are returned
    in one array and the errors in a separate array::

        d0: beam waist diameter [m]
        z0: axial location of beam waist [m]
        Theta: full beam divergence angle [radians]
        M2: beam propagation parameter [-]
        zR: Rayleigh distance [m]

    When `strict==True`, an estimate is made for the location of the beam focus
    and the Rayleigh distance. These values are then used to divide the
    measurements into three zones::

        * those within one Rayleigh distance of the focus,
        * those between 1 and 2 Rayleigh distances, and
        * those beyond two Rayleigh distances.

    values are used or unused depending on whether they comply with a strict
    reading of the ISO 11146-1 standard which requires::

        ... measurements at at least 10 different z positions shall be taken.
        Approximately half of the measurements shall be distributed within
        one Rayleigh length on either side of the beam waist, and approximately
        half of them shall be distributed beyond two Rayleigh lengths
        from the beam waist.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]
        strict: (optional) boolean for strict usage of ISO 11146
        z0: (optional) location of beam waist [m]
        d0: (optional) diameter of beam waist [m]

    Returns:
        params: [d0, z0, Theta, M2, zR]
        errors: [d0_std, z0_std, Theta_std, M2_std, zR_std]
        used: boolean array indicating if data point is used
    """
    used = np.full_like(z, True, dtype=bool)
    params, errors = basic_beam_fit(z, d, lambda0, z0=z0, d0=d0)
    if not strict:
        return params, errors, used

    z0 = params[1]
    zR = params[4]

    # identify zones (0=unused, 1=focal region, 2=outer region)
    zone = np.zeros_like(z)
    for i, zz in enumerate(z):
        if abs(zz - z0) <= 1.01 * zR:
            zone[i] = 1
        if 1.99 * zR <= abs(zz - z0):
            zone[i] = 2

    # count points in each zone
    n_focal = np.sum(zone == 1)
    n_outer = np.sum(zone == 2)

    if n_focal + n_outer < 10 or n_focal < 4 or n_outer < 4:
        print("Invalid distribution of measurements for ISO 11146")
        print("%d points within 1 Rayleigh distance" % n_focal)
        print("%d points greater than 2 Rayleigh distances" % n_outer)
        return params, errors, used

    # mark extra points in outer zone closest to focus as unused
    extra = n_outer - n_focal
    if n_focal == 4:
        extra = n_outer - 6
    for _ in range(extra):
        zone[min_index_in_outer_zone(abs(z - z0), zone)] = 0

    # mark extra points in focal zone farthest from focus as unused
    extra = n_outer - n_focal
    if n_outer == 4:
        extra = n_focal - 6
    for _ in range(n_focal - n_outer):
        zone[max_index_in_focal_zone(abs(z - z0), zone)] = 0

    # now find beam parameters with 50% focal and 50% outer zone values
    used = zone != 0
    dd = d[used]
    zz = z[used]
    params, errors = basic_beam_fit(zz, dd, lambda0, z0=z0, d0=d0)
    return params, errors, used


def M2_string(params, errors):
    """
    Return string describing a single set of beam measurements.

    Args:
        params: array of [d0, z0, Theta, M2, zR]
        errors: array of standard deviations of above

    Returns:
        s: formatted string suitable for printing.
    """
    d0, z0, Theta, M2, zR = params
    d0_std, z0_std, Theta_std, M2_std, zR_std = errors
    BPP, BPP_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)

    s = ""
    s += "       M^2 = %.2f ± %.2f\n" % (M2, M2_std)
    s += "\n"
    s += "       d_0 = %.0f ± %.0f µm\n" % (d0 * 1e6, d0_std * 1e6)
    s += "       w_0 = %.0f ± %.0f µm\n" % (d0 / 2 * 1e6, d0_std / 2 * 1e6)
    s += "\n"
    s += "       z_0 = %.0f ± %.0f mm\n" % (z0 * 1e3, z0_std * 1e3)
    s += "       z_R = %.0f ± %.0f mm\n" % (zR * 1e3, zR_std * 1e3)
    s += "\n"
    s += "     Theta = %.2f ± %.2f mrad\n" % (Theta * 1e3, Theta_std * 1e3)
    s += "\n"
    s += "       BPP = %.2f ± %.2f mm mrad\n" % (BPP * 1e6, BPP_std * 1e6)
    return s


def _M2_report(z, d, lambda0, f=None, strict=False, z0=None, d0=None):
    """
    Return string describing a single set of beam measurements.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]
        f: (optional) focal length of lens [m]
        strict: (optional) boolean for strict usage of ISO 11146
        z0: (optional) location of beam waist [m]
        d0: (optional) diameter of beam waist [m]

    Returns:
        s: formatted string suitable for printing.
    """
    params, errors, _ = M2_fit(z, d, lambda0, strict, z0=z0, d0=d0)

    if f is None:
        s = "Beam propagation parameters\n"
        s += M2_string(params, errors)
        return s

    s = "Beam propagation parameters for the focused beam\n"
    s += M2_string(params, errors)
    o_params, o_errors = lbs.artificial_to_original(params, errors, f)
    s += "\nBeam propagation parameters for the laser beam\n"
    s += M2_string(o_params, o_errors)
    return s


def M2_report(z, d_major, lambda0, d_minor=None, f=None, strict=False, z0=None, d0=None):
    """
    Return string describing a one or more sets of beam measurements.

    Args:
        z: array of axial position of beam measurements [m]
        d_major: array of major axis (diameters) [m]
        lambda0: wavelength of the laser [m]
        d_minor: (optional) array of beam diameters for minor axis [m]
        f: (optional) focal length of lens [m]
        strict: (optional) boolean for strict usage of ISO 11146
        z0: (optional) location of beam waist [m]
        d0: (optional) diameter of beam waist [m]

    Returns:
        s: formatted string suitable for printing.
    """
    if d_minor is None:
        s = _M2_report(z, d_major, lambda0, f=f, strict=strict, z0=z0, d0=d0)
        return s

    params, errors, _ = M2_fit(z, d_major, lambda0, strict=strict, z0=z0, d0=d0)
    d0x, z0x, Thetax, M2x, zRx = params
    d0x_std, z0x_std, Thetax_std, M2x_std, zRx_std = errors

    params, errors, _ = M2_fit(z, d_minor, lambda0, strict=strict, z0=z0, d0=d0)
    d0y, z0y, Thetay, M2y, zRy = params
    d0y_std, z0y_std, Thetay_std, M2y_std, zRy_std = errors

    z0 = (z0x + z0y) / 2
    z0_std = np.sqrt(z0x_std**2 + z0y_std**2)

    d0 = (d0x + d0y) / 2
    d0_std = np.sqrt(d0x_std**2 + d0y_std**2)

    zR = (zRx + zRy) / 2
    zR_std = np.sqrt(zRx_std**2 + zRy_std**2)

    Theta = (Thetax + Thetay) / 2
    Theta_std = np.sqrt(Thetax_std**2 + Thetay_std**2)

    M2 = np.sqrt(M2x * M2y)
    M2_std = np.sqrt(M2x_std**2 + M2y_std**2)

    BPP, BPP_std = lbs.beam_parameter_product(Theta, d0, Theta_std, d0_std)
    BPPx, BPPx_std = lbs.beam_parameter_product(Thetax, d0x, Thetax_std, d0x_std)
    BPPy, BPPy_std = lbs.beam_parameter_product(Thetay, d0y, Thetay_std, d0y_std)

    tag = ""
    if f is not None:
        tag = " of the focused beam"

    s = "Beam propagation parameters derived from hyperbolic fit\n"
    s += "Beam Propagation Ratio%s\n" % tag
    s += "        M2 = %.2f ± %.2f\n" % (M2, M2_std)
    s += "       M2x = %.2f ± %.2f\n" % (M2x, M2x_std)
    s += "       M2y = %.2f ± %.2f\n" % (M2y, M2y_std)

    s += "Beam waist diameter%s\n" % tag
    s += "        d0 = %.0f ± %.0f µm\n" % (d0 * 1e6, d0_std * 1e6)
    s += "       d0x = %.0f ± %.0f µm\n" % (d0x * 1e6, d0x_std * 1e6)
    s += "       d0y = %.0f ± %.0f µm\n" % (d0y * 1e6, d0y_std * 1e6)

    s += "Beam waist location%s\n" % tag
    s += "        z0 = %.0f ± %.0f mm\n" % (z0 * 1e3, z0_std * 1e3)
    s += "       z0x = %.0f ± %.0f mm\n" % (z0x * 1e3, z0x_std * 1e3)
    s += "       z0y = %.0f ± %.0f mm\n" % (z0y * 1e3, z0y_std * 1e3)

    s += "Rayleigh Length%s\n" % tag
    s += "        zR = %.0f ± %.0f mm\n" % (zR * 1e3, zR_std * 1e3)
    s += "       zRx = %.0f ± %.0f mm\n" % (zRx * 1e3, zRx_std * 1e3)
    s += "       zRy = %.0f ± %.0f mm\n" % (zRy * 1e3, zRy_std * 1e3)

    s += "Divergence Angle%s\n" % tag
    s += "     theta = %.2f ± %.2f milliradians\n" % (Theta * 1e3, Theta_std * 1e3)
    s += "   theta_x = %.2f ± %.2f milliradians\n" % (Thetax * 1e3, Thetax_std * 1e3)
    s += "   theta_y = %.2f ± %.2f milliradians\n" % (Thetay * 1e3, Thetay_std * 1e3)

    s += "Beam parameter product%s\n" % tag
    s += "       BPP = %.2f ± %.2f mm * mrad\n" % (BPP * 1e6, BPP_std * 1e6)
    s += "     BPP_x = %.2f ± %.2f mm * mrad\n" % (BPPx * 1e6, BPPx_std * 1e6)
    s += "     BPP_y = %.2f ± %.2f mm * mrad\n" % (BPPy * 1e6, BPPy_std * 1e6)
    if f is None:
        return s

    # needs to be completed
    x2 = z0x - f
    y2 = z0y - f
    r2 = z0 - f

    Vx = f / np.sqrt(zRx**2 + x2**2)
    Vy = f / np.sqrt(zRy**2 + y2**2)
    V = f / np.sqrt(zR**2 + r2**2)

    d0x *= Vx
    d0y *= Vy
    d0 *= V

    z0x = Vx**2 * x2 + f
    z0y = Vy**2 * y2 + f
    z0 = V**2 * r2 + f

    return s
