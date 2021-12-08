# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=unbalanced-tuple-unpacking

"""
A module for finding M² values for a laser beam.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

Finding the beam waist size, location, and M² for a beam is straightforward::

    import numpy as np
    import laserbeamsize as lbs

    lambda0 = 632.8e-9 # m
    z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770])
    r = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397])
    lbs.M2_report(z*1e-3, 2*r*1e-6, lambda0)

A graphic of the fit to diameters can be created by::

    lbs.M2_diameter_plot(z*1e-3, 2*r*1e-6, lambda0)
    plt.show()

A graphic of the radial fit can be created by::

    lbs.M2_radius_plot(z*1e-3, 2*r*1e-6, lambda0)
    plt.show()
"""

import scipy.optimize
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np

__all__ = ('z_rayleigh',
           'beam_radius',
           'magnification',
           'image_distance',
           'curvature',
           'divergence',
           'gouy_phase',
           'focused_diameter',
           'artificial_to_original',
           'M2_fit',
           'M2_report',
           'M2_diameter_plot',
           'M2_radius_plot',
           'M2_focus_plot'
           )

def z_rayleigh(w0, lambda0, M2=1):
    """
    Return the Rayleigh distance for a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
    Returns:
        distance where irradiance drops by 1/2 [m]
    """
    return np.pi * w0**2/lambda0/M2


def beam_radius(w0, lambda0, z, z0=0, M2=1):
    """
    Return the beam radius at an axial location.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        z: axial location of desired beam radius [m]
        z0: axial location of beam waist [m]
        M2: beam propagation factor [-]
    Returns:
        Beam radius [m]
    """
    zz = (z-z0)/z_rayleigh(w0, lambda0, M2)
    return w0*np.sqrt(1+zz**2)


def magnification(w0, lambda0, s, f, M2=1):
    """
    Return the magnification of a Gaussian beam.

    If the beam waist is before the lens, then the distance s
    will be negative, i.e. if it is at the front focus of the lens (s=-f).

    The new beam waist will be `m*w0` and the new Rayleigh
    distance will be `m**2 * zR`

    Args:
        f: focal distance of lens [m]
        zR: Rayleigh distance [m]
        s: distance of beam waist to lens [m]

    Returns:
        magnification m [-]
    """
    zR2 = z_rayleigh(w0, lambda0, M2)**2
    return f/np.sqrt((s+f)**2+zR2)


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
    returns
        radius of curvature of field at z          [m]
    """
    zR2 = z_rayleigh(w0, lambda0, M2)**2
    return (z - z0) + zR2/(z - z0)


def divergence(w0, lambda0, M2=1):
    """
    Calculate the full angle of divergence of a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        M2: beam propagation factor [-]
    returns
        divergence of beam [radians]
    """
    return 2*w0/z_rayleigh(w0, lambda0, M2)


def gouy_phase(w0, lambda0, z, z0=0):
    """
    Calculate the Gouy phase of a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        z: axial position along beam  [m]
        z0: axial position of beam waist  [m]
    returns
        Gouy phase                     [radians]
    """
    zR = z_rayleigh(w0, lambda0)
    return -np.arctan2(z-z0, zR)


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
        Beam diameter [m]
    """
    return 4 * M2**2 * lambda0 * f / (np.pi * d)


def image_distance(w0, lambda0, s, f, M2=1):
    """
    Return the image location of a Gaussian beam.

    The default case is when the beam waist is located at
    the front focus of the lens (s=-f).

    Args:
        s: distance of beam waist to lens [m]
        f: focal distance of lens [m]
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
        M2: beam propagation factor [-]

    Returns:
        location of new beam waist [m]
    """
    zR2 = z_rayleigh(w0, lambda0, M2)**2
    return f * (s*f + s*s + zR2)/((f+s)**2+ zR2)


def _abc_fit(z, d, lambda0):
    """
    Return beam parameters for beam diameter measurements.

    Follows ISO 11146-1 section 9 and uses the standard `polyfit` routine
    in `numpy` to find the coefficients `a`, `b`, and `c`.

    d(z)**2 = a + b*z + c*z**2

    These coefficients are used to determine the beam parameters using
    equations 25-29 from ISO 11146-1.

    Unfortunately, standard error propagation fails to accurately determine
    the standard deviations of these parameters.  Therefore the error calculation
    lines are commented out and only the beam parameters are returned.

    Args:
        z: axial position of beam measurement [m]
        d: beam diameter [m]
    Returns:
        d0: beam waist diameter [m]
        z0: axial location of beam waist [m]
        M2: beam propagation parameter [-]
        Theta: full beam divergence angle [radians]
        zR: Rayleigh distance [m]
    """
    nlfit, _nlpcov = np.polyfit(z, d**2, 2, cov=True)

    # unpack fitting parameters
    c, b, a = nlfit


    z0 = -b/(2*c)
    Theta = np.sqrt(c)
    disc = np.sqrt(4*a*c-b*b)/2
    M2 = np.pi/4/lambda0*disc
    d0 = disc / np.sqrt(c)
    zR = disc/c
    params = [d0, z0, Theta, M2, zR]

# unpack uncertainties in fitting parameters from diagonal of covariance matrix
#c_std, b_std, a_std = [np.sqrt(_nlpcov[j, j]) for j in range(nlfit.size)]
#z0_std = z0*np.sqrt(b_std**2/b**2 + c_std**2/c**2)
#d0_std = np.sqrt((4*c**2*a_std)**2 + (2*b*c*b_std)**2 + (b**2*c_std)**2) / (8*c**2*d0)
#Theta_std = c_std/2/np.sqrt(c)
#zR_std = np.sqrt(4*c**4*a_std**2 + b**2*c**2*b_std**2 + (b**2-2*a*c)**2*c_std**2)/(4*c**3) / zR
#M2_std = np.pi**2 * np.sqrt(4*c**2*a_std**2 + b**2*b_std**2 + 4*a**2*c_std**2)/(64*lambda0**2) / M2
#errors = [d0_std, z0_std, M2_std, Theta_std, zR_std]
    return params


def _beam_fit_fn_(z, d0, z0, Theta):
    """Fitting function for d0, z0, and Theta."""
    return d0**2 + (Theta*(z-z0))**2

def _beam_fit_fn_2(z, d0, Theta):
    """Fitting function for d0 and Theta."""
    return d0**2 + (Theta*z)**2

def _beam_fit_fn_3(z, z0, Theta):
    """Fitting function for z0 and Theta."""
    return (Theta*(z-z0))**2

def _beam_fit_fn_4(z, Theta):
    """Fitting function for just Theta."""
    return (Theta*z)**2

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

        d0: beam waist diameter [m]
        z0: axial location of beam waist [m]
        Theta: full beam divergence angle [radians]
        M2: beam propagation parameter [-]
        zR: Rayleigh distance [m]

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]

    Returns:
        params, errors
    """
    # approximate answer
    i = np.argmin(d)
    d0_guess = d[i]
    z0_guess = z[i]

    # fit data using SciPy's curve_fit() algorithm
    if z0 is None:
        if d0 is None:
            i = np.argmax(abs(z-z0_guess))
            theta_guess = abs(d[i]/(z[i]-z0_guess))
            p0 = [d0_guess, z0_guess, theta_guess]
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_, z, d**2, p0=p0)
            d0, z0, Theta = nlfit
            d0_std, z0_std, Theta_std = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]
        else:
            i = np.argmax(abs(z-z0_guess))
            theta_guess = abs(d[i]/(z[i]-z0_guess))
            p0 = [z0_guess, theta_guess]
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_3, z, d**2-d0**2, p0=p0)
            z0, Theta = nlfit
            z0_std, Theta_std = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]
            d0_std = 0
    else:
        i = np.argmax(abs(z-z0))
        theta_guess = abs(d[i]/(z[i]-z0))
        if d0 is None:
            p0 = [d0_guess, theta_guess]
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_2, z-z0, d**2, p0=p0)
            d0, Theta = nlfit
            d0_std, Theta_std = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]
            z0_std = 0
        else:
            p0 = [theta_guess]
            nlfit, nlpcov = scipy.optimize.curve_fit(_beam_fit_fn_4, z-z0, d**2-d0**2, p0=p0)
            Theta = nlfit[0]
            Theta_std = np.sqrt(nlpcov[0, 0])
            z0_std = 0
            d0_std = 0

    # divergence and Rayleigh range of Gaussian beam
    Theta0 = 4 * lambda0 / (np.pi * d0)
    zR = np.pi * d0**2 / (4 * lambda0)

    M2 = Theta/Theta0
    zR = np.pi * d0**2 / (4 * lambda0 * M2)

    M2_std = M2 * np.sqrt((Theta_std/Theta)**2 + (d0_std/d0)**2)
    zR_std = zR * np.sqrt((M2_std/M2)**2 + (2*d0_std/d0)**2)

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

    d(z)**2 = d0**2 + Theta**2 * (z-z0)**2

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
        if abs(zz-z0) <= 1.01*zR:
            zone[i] = 1
        if 1.99*zR <= abs(zz-z0):
            zone[i] = 2

    # count points in each zone
    n_focal = np.sum(zone == 1)
    n_outer = np.sum(zone == 2)

    if n_focal+n_outer < 10 or n_focal < 4 or n_outer < 4:
        print("Invalid distribution of measurements for ISO 11146")
        print("%d points within 1 Rayleigh distance" % n_focal)
        print("%d points greater than 2 Rayleigh distances" % n_outer)
        return params, errors, used

    # mark extra points in outer zone closest to focus as unused
    extra = n_outer-n_focal
    if n_focal == 4:
        extra = n_outer - 6
    for _ in range(extra):
        zone[min_index_in_outer_zone(abs(z-z0), zone)] = 0

    # mark extra points in focal zone farthest from focus as unused
    extra = n_outer-n_focal
    if n_outer == 4:
        extra = n_focal - 6
    for _ in range(n_focal-n_outer):
        zone[max_index_in_focal_zone(abs(z-z0), zone)] = 0

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
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]

    Returns:
        Formatted string suitable for printing.
    """
    d0, z0, Theta, M2, zR = params
    d0_std, z0_std, Theta_std, M2_std, zR_std = errors
    s = ''
    s += "       M^2 = %.2f ± %.2f\n" % (M2, M2_std)
    s += "\n"
    s += "       d_0 = %.0f ± %.0f µm\n" % (d0*1e6, d0_std*1e6)
    s += "       w_0 = %.0f ± %.0f µm\n" % (d0/2*1e6, d0_std/2*1e6)
    s += "\n"
    s += "       z_0 = %.0f ± %.0f mm\n" % (z0*1e3, z0_std*1e3)
    s += "       z_R = %.0f ± %.0f mm\n" % (zR*1e3, zR_std*1e3)
    s += "\n"
    s += "     Theta = %.2f ± %.2f mrad\n" % (Theta*1e3, Theta_std*1e3)
    return s


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

    d0: artificial beam waist diameter [m]
    z0: artificial beam waist position relative to lens surface [m]
    Theta: full beam divergence angle for artificial beam [radians]
    M2: beam propagation parameter [-]
    zR: Rayleigh distance for artificial beam [m]

    The errors that are returned are not quite right at the moment.

    Args:
        params: [d0, z0, Theta, M2, zR]
        errors: array with std dev of above parameters
        f: focal length of lens [m]
        hiatus: distance between principal planes of focusing lens [m]

    Returns:
        original beam parameters and errors.
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

    orig_Theta = art_Theta/V
    orig_Theta_std = art_Theta_std/V

    o_params = [orig_d0, orig_z0, orig_Theta, M2, orig_zR]
    o_errors = [orig_d0_std, orig_z0_std, orig_Theta_std, M2_std, orig_zR_std]
    return o_params, o_errors


def _M2_report(z, d, lambda0, f=None, strict=False, z0=None, d0=None):
    """
    Return string describing a single set of beam measurements.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]

    Returns:
        Formatted string suitable for printing.
    """
    params, errors, _ = M2_fit(z, d, lambda0, strict, z0=z0, d0=d0)

    if f is None:
        s = "Beam propagation parameters\n"
        s += M2_string(params, errors)
        return s

    s = "Beam propagation parameters for the focused beam\n"
    s += M2_string(params, errors)
    o_params, o_errors = artificial_to_original(params, errors, f)
    s += "\nBeam propagation parameters for the laser beam\n"
    s += M2_string(o_params, o_errors)
    return s


def M2_report(z, dx, lambda0, dy=None, f=None, strict=False, z0=None, d0=None):
    """
    Return string describing a one or sets of beam measurements.

    Args:
        z: array of axial position of beam measurements [m]
        dx: array of beam diameters for semi-major axis [m]
        dy: array of beam diameters for semi-minor axis [m]
        lambda0: wavelength of the laser [m]

    Returns:
        Formatted string suitable for printing.
    """
    if dy is None:
        s = _M2_report(z, dx, lambda0, f=f, strict=strict, z0=z0, d0=d0)
        return s

    params, errors, _ = M2_fit(z, dx, lambda0, strict=strict, z0=z0, d0=d0)
    d0x, z0x, Thetax, M2x, zRx = params
    d0x_std, z0x_std, Thetax_std, M2x_std, zRx_std = errors

    params, errors, _ = M2_fit(z, dy, lambda0, strict=strict, z0=z0, d0=d0)
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

    tag = ''
    if f is not None:
        tag = " of the focused beam"

    s = "Beam propagation parameters derived from hyperbolic fit\n"
    s += "Beam Propagation Ratio%s\n"  %tag
    s += "        M2 = %.2f ± %.2f\n" % (M2, M2_std)
    s += "       M2x = %.2f ± %.2f\n" % (M2x, M2x_std)
    s += "       M2y = %.2f ± %.2f\n" % (M2y, M2y_std)

    s += "Beam waist diameter%s\n"  %tag
    s += "        d0 = %.0f ± %.0f µm\n" % (d0*1e6, d0_std*1e6)
    s += "       d0x = %.0f ± %.0f µm\n" % (d0x*1e6, d0x_std*1e6)
    s += "       d0y = %.0f ± %.0f µm\n" % (d0y*1e6, d0y_std*1e6)

    s += "Beam waist location%s\n"  %tag
    s += "        z0 = %.0f ± %.0f mm\n" % (z0*1e3, z0_std*1e3)
    s += "       z0x = %.0f ± %.0f mm\n" % (z0x*1e3, z0x_std*1e3)
    s += "       z0y = %.0f ± %.0f mm\n" % (z0y*1e3, z0y_std*1e3)

    s += "Rayleigh Length%s\n"  %tag
    s += "        zR = %.0f ± %.0f mm\n" % (zR*1e3, zR_std*1e3)
    s += "       zRx = %.0f ± %.0f mm\n" % (zRx*1e3, zRx_std*1e3)
    s += "       zRy = %.0f ± %.0f mm\n" % (zRy*1e3, zRy_std*1e3)

    s += "Divergence Angle%s\n"  %tag
    s += "     theta = %.2f ± %.2f milliradians\n" % (Theta*1e3, Theta_std*1e3)
    s += "   theta_x = %.2f ± %.2f milliradians\n" % (Thetax*1e3, Thetax_std*1e3)
    s += "   theta_y = %.2f ± %.2f milliradians\n" % (Thetay*1e3, Thetay_std*1e3)

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

def _fit_plot(z, d, lambda0, strict=False, z0=None, d0=None):
    """
    Plot beam diameters and ISO 11146 fit.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]

    Returns:
        residuals, z0, zR
    """
    params, errors, used = M2_fit(z, d, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    d0, z0, Theta, M2, zR = params
    d0_std, z0_std, Theta_std, M2_std, zR_std = errors

    # fitted line
    zmin = min(np.min(z), z0-4*zR)
    zmax = max(np.max(z), z0+4*zR)
#    plt.xlim(zmin, zmax)
    z_fit = np.linspace(zmin, zmax)
#    d_fit = np.sqrt(d0**2 + (Theta*(z_fit-z0))**2)
#    plt.plot(z_fit*1e3, d_fit*1e6, ':k')
    d_fit_lo = np.sqrt((d0-d0_std)**2 + ((Theta-Theta_std)*(z_fit-z0))**2)
    d_fit_hi = np.sqrt((d0+d0_std)**2 + ((Theta+Theta_std)*(z_fit-z0))**2)
    plt.fill_between(z_fit*1e3, d_fit_lo*1e6, d_fit_hi*1e6, color='red', alpha=0.5)

    # show perfect gaussian caustic when unphysical M2 arises
    if M2 < 1:
        Theta00 = 4 * lambda0 / (np.pi * d0)
        d_00 = np.sqrt(d0**2 + (Theta00*(z_fit-z0))**2)
        plt.plot(z_fit*1e3, d_00*1e6, ':k', lw=2, label="M²=1")
        plt.legend(loc="lower right")

    plt.fill_between(z_fit*1e3, d_fit_lo*1e6, d_fit_hi*1e6, color='red', alpha=0.5)
    # data points
    plt.plot(z[used]*1e3, d[used]*1e6, 'o', color='black', label='used')
    plt.plot(z[unused]*1e3, d[unused]*1e6, 'ok', mfc='none', label='unused')
    plt.xlabel('')
    plt.ylabel('')

    tax = plt.gca().transAxes
    plt.text(0.05, 0.30, '$M^2$ = %.2f±%.2f ' % (M2, M2_std), transform=tax)
    plt.text(0.05, 0.25, '$d_0$ = %.0f±%.0f µm' % (d0*1e6, d0_std*1e6), transform=tax)
    plt.text(0.05, 0.15, '$z_0$  = %.0f±%.0f mm' % (z0*1e3, z0_std*1e3), transform=tax)
    plt.text(0.05, 0.10, '$z_R$  = %.0f±%.0f mm' % (zR*1e3, zR_std*1e3), transform=tax)
    plt.text(0.05, 0.05, r'$\Theta$  = %.2f±%.2f mrad' % (Theta*1e3, Theta_std*1e3), transform=tax)

    plt.axvline(z0*1e3, color='black', lw=1)
    plt.axvspan((z0-zR)*1e3, (z0+zR)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0-2*zR)*1e3, (zmin)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0+2*zR)*1e3, (zmax)*1e3, color='cyan', alpha=0.3)

#    plt.axhline(d0*1e6, color='black', lw=1)
#    plt.axhspan((d0+d0_std)*1e6, (d0-d0_std)*1e6, color='red', alpha=0.1)
    plt.title(r'$d^2(z) = d_0^2 + \Theta^2 (z-z_0)^2$')
    if sum(z[unused]) > 0:
        plt.legend(loc='upper right')

    residuals = d - np.sqrt(d0**2 + (Theta*(z-z0))**2)
    return residuals, z0, zR, used


def _M2_diameter_plot(z, d, lambda0, strict=False, z0=None, d0=None):
    """
    Plot the fitted beam and the residuals.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]

    Returns:
        nothing
    """
    fig = plt.figure(1, figsize=(12, 8))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[6, 2])

    fig.add_subplot(gs[0])
    residualsx, z0, zR, used = _fit_plot(z, d, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    zmin = min(np.min(z), z0-4*zR)
    zmax = max(np.max(z), z0+4*zR)

    plt.ylabel('beam diameter (µm)')
    plt.ylim(0, 1.1*max(d)*1e6)

    fig.add_subplot(gs[1])
    plt.plot(z*1e3, residualsx*1e6, "ro")
    plt.plot(z[used]*1e3, residualsx[used]*1e6, 'ok', label='used')
    plt.plot(z[unused]*1e3, residualsx[unused]*1e6, 'ok', mfc='none', label='unused')

    plt.axhline(color="gray", zorder=-1)
    plt.xlabel('axial position $z$ (mm)')
    plt.ylabel('residuals (µm)')
    plt.axvspan((z0-zR)*1e3, (z0+zR)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0-2*zR)*1e3, (zmin)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0+2*zR)*1e3, (zmax)*1e3, color='cyan', alpha=0.3)


def M2_diameter_plot(z, dx, lambda0, dy=None, strict=False, z0=None, d0=None):
    """
    Plot the semi-major and semi-minor beam fits and residuals.

    Args:
        z: array of axial position of beam measurements [m]
        lambda0: wavelength of the laser [m]
        dx: array of beam diameters  [m]

    Returns:
        nothing
    """
    if dy is None:
        _M2_diameter_plot(z, dx, lambda0, strict=strict, z0=z0, d0=d0)
        return

    ymax = 1.1 * max(np.max(dx), np.max(dy)) * 1e6

    # Create figure window to plot data
    fig = plt.figure(1, figsize=(12, 8))
    gs = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[6, 2])

    # semi-major axis plot
    fig.add_subplot(gs[0, 0])
    residualsx, z0x, zR, used = _fit_plot(z, dx, lambda0, strict=strict, z0=z0, d0=d0)
    zmin = min(np.min(z), z0x-4*zR)
    zmax = max(np.max(z), z0x+4*zR)
    unused = np.logical_not(used)
    plt.ylabel('beam diameter (µm)')
    plt.title('Semi-major Axis Diameters')
    plt.ylim(0, ymax)

    # semi-major residuals
    fig.add_subplot(gs[1, 0])
    ax = plt.gca()
    plt.plot(z[used]*1e3, residualsx[used]*1e6, 'ok', label='used')
    plt.plot(z[unused]*1e3, residualsx[unused]*1e6, 'ok', mfc='none', label='unused')
    plt.axhline(color="gray", zorder=-1)
    plt.xlabel('axial position $z$ (mm)')
    plt.ylabel('residuals (µm)')
    plt.axvspan((z0x-zR)*1e3, (z0x+zR)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0x-2*zR)*1e3, (zmin)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0x+2*zR)*1e3, (zmax)*1e3, color='cyan', alpha=0.3)

    # semi-minor axis plot
    fig.add_subplot(gs[0, 1])
    residualsy, z0y, zR, used = _fit_plot(z, dy, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    plt.title('Semi-minor Axis Diameters')
    plt.ylim(0, ymax)

    ymax = max(np.max(residualsx), np.max(residualsy)) * 1e6
    ymin = min(np.min(residualsx), np.min(residualsy)) * 1e6
    ax.set_ylim(ymin, ymax)

    # semi-minor residuals
    fig.add_subplot(gs[1, 1])
    plt.plot(z[used]*1e3, residualsy[used]*1e6, 'ok', label='used')
    plt.plot(z[unused]*1e3, residualsy[unused]*1e6, 'ok', mfc='none', label='unused')
    plt.axhline(color="gray", zorder=-1)
    plt.xlabel('axial position $z$ (mm)')
    plt.ylabel('')
    plt.axvspan((z0y-zR)*1e3, (z0y+zR)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0y-2*zR)*1e3, (zmin)*1e3, color='cyan', alpha=0.3)
    plt.axvspan((z0y+2*zR)*1e3, (zmax)*1e3, color='cyan', alpha=0.3)
    plt.ylim(ymin, ymax)


def M2_radius_plot(z, d, lambda0, strict=False, z0=None, d0=None):
    """
    Plot radii, beam fits, and asymptotes.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]

    Returns:
        nothing
    """
    params, errors, used = M2_fit(z, d, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    d0, z0, Theta, M2, zR = params
    d0_std, _, Theta_std, M2_std, _ = errors

    plt.figure(1, figsize=(12, 8))

    # fitted line
    zmin = min(np.min(z-z0), -4*zR) * 1.05 + z0
    zmax = max(np.max(z-z0), +4*zR) * 1.05 + z0
    plt.xlim((zmin-z0)*1e3, (zmax-z0)*1e3)

    z_fit = np.linspace(zmin, zmax)
    d_fit = np.sqrt(d0**2 + (Theta*(z_fit-z0))**2)
#    plt.plot((z_fit-z0)*1e3, d_fit*1e6/2, ':r')
#    plt.plot((z_fit-z0)*1e3, -d_fit*1e6/2, ':r')
    d_fit_lo = np.sqrt((d0-d0_std)**2 + ((Theta-Theta_std)*(z_fit-z0))**2)
    d_fit_hi = np.sqrt((d0+d0_std)**2 + ((Theta+Theta_std)*(z_fit-z0))**2)

    # asymptotes
    r_left = -(z0-zmin)*np.tan(Theta/2)*1e6
    r_right = (zmax-z0)*np.tan(Theta/2)*1e6
    plt.plot([(zmin-z0)*1e3, (zmax-z0)*1e3], [r_left, r_right], '--b')
    plt.plot([(zmin-z0)*1e3, (zmax-z0)*1e3], [-r_left, -r_right], '--b')

    # xticks along top axis
    ticks = [(i*zR)*1e3 for i in range(int((zmin-z0)/zR), int((zmax-z0)/zR)+1)]
    ticklabels1 = ["%.0f" % (z+z0*1e3) for z in ticks]
    ticklabels2 = []
    for i in  range(int((zmin-z0)/zR), int((zmax-z0)/zR)+1):
        if i == 0:
            ticklabels2 = np.append(ticklabels2, "0")
        elif i == -1:
            ticklabels2 = np.append(ticklabels2, r"-$z_R$")
        elif i == 1:
            ticklabels2 = np.append(ticklabels2, r"$z_R$")
        else:
            ticklabels2 = np.append(ticklabels2, r"%d$z_R$"%i)
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(ticklabels1, fontsize=14)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(ticklabels2, fontsize=14)

    # usual labels for graph
    ax1.set_xlabel('Axial Location (mm)', fontsize=14)
    ax1.set_ylabel('Beam radius (µm)', fontsize=14)
    title = r'$w_0=d_0/2$=%.0f±%.0fµm,  ' % (d0/2*1e6, d0_std/2*1e6)
    title += r'$M^2$ = %.2f±%.2f,  ' % (M2, M2_std)
    title += r'$\lambda$=%.0f nm' % (lambda0*1e9)
    plt.title(title, fontsize=16)

    # show the divergence angle
    s = r'$\Theta$  = %.2f±%.2f mrad' % (Theta*1e3, Theta_std*1e3)
    plt.text(2*zR*1e3, 0, s, ha='left', va='center', fontsize=16)
    arc_x = 1.5*zR*1e3
    arc_y = 1.5*zR*np.tan(Theta/2)*1e6
    plt.annotate('', (arc_x, -arc_y), (arc_x, arc_y),
                 arrowprops=dict(arrowstyle="<->",
                                 connectionstyle="arc3, rad=-0.2"))

    # show the Rayleigh ranges
    ymin = max(max(d_fit), max(d))
    ymin *= -1/2 * 1e6
    plt.text(0, ymin, '$-z_R<z-z_0<z_R$', ha='center', va='bottom', fontsize=16)
    x = (zmax-z0 + 2*zR)/2 * 1e3
    plt.text(x, ymin, '$2z_R < z-z_0$', ha='center', va='bottom', fontsize=16)
    x = (zmin-z0 - 2*zR)/2 * 1e3
    plt.text(x, ymin, '$z-z_0 < -2z_R$', ha='center', va='bottom', fontsize=16)

    ax1.axvspan((-zR)*1e3, (+zR)*1e3, color='cyan', alpha=0.3)
    ax1.axvspan((-2*zR)*1e3, (zmin-z0)*1e3, color='cyan', alpha=0.3)
    ax1.axvspan((+2*zR)*1e3, (zmax-z0)*1e3, color='cyan', alpha=0.3)

    # show the fit
    zz = (z_fit-z0)*1e3
    lo = d_fit_lo*1e6/2
    hi = d_fit_hi*1e6/2
    ax1.fill_between(zz, lo, hi, color='red', alpha=0.5)
    ax1.fill_between(zz, -lo, -hi, color='red', alpha=0.5)

    # show perfect gaussian caustic when unphysical M2 arises
    if M2 < 1:
        Theta00 = 4 * lambda0 / (np.pi * d0)
        r_00 = np.sqrt(d0**2 + (Theta00*zz*1e-3)**2)/2 * 1e6
        plt.plot(zz, r_00, ':k', lw=2, label="M²=1")
        plt.plot(zz, -r_00, ':k', lw=2)
        plt.legend(loc="lower right")

    # data points
    ax1.plot((z[used]-z0)*1e3, d[used]*1e6/2, 'ok', label='used')
    ax1.plot((z[used]-z0)*1e3, -d[used]*1e6/2, 'ok')
    ax1.plot((z[unused]-z0)*1e3, d[unused]*1e6/2, 'ok', mfc='none', label='unused')
    ax1.plot((z[unused]-z0)*1e3, -d[unused]*1e6/2, 'ok', mfc='none')
    if sum(z[unused]) > 0:
        ax1.legend(loc='center left')


def M2_focus_plot(w0, lambda0, f, z0, M2=1):
    """
    Plot a beam from its waist through a lens to its focus.

    The lens is at `z=0` with respect to the beam waist. All distances to
    the left of the lens are negative and those to the right are positive.

    The beam has a waist at `z0`.  If the beam waist is at the front focal
    plane of the lens then `z0=-f`.

    Args:
        w0: beam radius at waist [m]
        lambda0: wavelength of beam [m]
        f: focal length of lens [m]
        z0: location of beam waist [m]
        M2: beam propagation factor [-]

    Returns:
        nothing.
    """
    # plot the beam from just before the waist to the lens
    left = 1.1*z0
    z = np.linspace(left, 0)
    r = beam_radius(w0, lambda0, z, z0=z0, M2=M2)
    plt.fill_between(z*1e3, -r*1e6, r*1e6, color='red', alpha=0.2)

    # find the gaussian beam parameters for the beam after the lens
    w0_after = w0 * magnification(w0, lambda0, z0, f, M2=M2)
    z0_after = image_distance(w0, lambda0, z0, f, M2=M2)
    zR_after = z_rayleigh(w0_after, lambda0, M2)

    # plot the beam after the lens
    right = max(2*f, z0_after+4*zR_after)
    z_after = np.linspace(0, right)
    r_after = beam_radius(w0_after, lambda0, z_after, z0=z0_after, M2=M2)

    # plt.axhline(w0_after*1.41e6)
    plt.fill_between(z_after*1e3, -r_after*1e6, r_after*1e6, color='red', alpha=0.2)

    # locate the lens and the two beam waists
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black')
    plt.axvline(z0*1e3, color='black', linestyle=':')
    plt.axvline(z0_after*1e3, color='black', linestyle=':')

    # finally, show the ±1 Rayleigh distance
    zRmin = max(0, (z0_after-zR_after))*1e3
    zRmax = (z0_after+zR_after)*1e3
    plt.axvspan(zRmin, zRmax, color='blue', alpha=0.1)

    plt.xlabel('Axial Position Relative to Lens (mm)')
    plt.ylabel('Beam Radius (microns)')
    title = "$w_0$=%.0fµm, $z_0$=%.0fmm, " % (w0*1e6, z0*1e3)
    title += "$w_0'$=%.0fµm, $z_0'$=%.0fmm, " % (w0_after*1e6, z0_after*1e3)
    title += "$z_R'$=%.0fmm" % (zR_after*1e3)
    plt.title(title)
