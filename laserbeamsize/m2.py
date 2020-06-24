# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=unbalanced-tuple-unpacking

"""
A module for finding M2 values for a laser beam.

Finding the center and dimensions of a monochrome image of a beam is simple::

    import numpy as np
    import laserbeamsize as lbs

    lambda0 = 632.8/1e6 # mm
    z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770])
    d = np.array([0.5976914 , 0.57246158, 0.54747159, 0.55427816, 0.47916078,
           0.40394918, 0.41464084, 0.39929649, 0.3772103 , 0.39076051,
           0.32638856, 0.39693297])*2

    M2_analysis(z, d, lambda0)
"""

import numpy as np
import matplotlib.gridspec
import matplotlib.pyplot as plt


__all__ = ('z_rayleigh',
           'beam_radius',
           'focused_diameter',
           'abc_fit',
           'beam_params',
           'beam_param_errors',
           'M2_analysis',
           'M2_graph',
           'M2_graph2',
           'M2_report',
           'M2_report2',
           )

def z_rayleigh(w0, lambda0):
    """
    Return the Rayleigh distance.
    Args:
        w0 : minimum beam radius [m]
        lambda0: wavelength of light [m]
    Returns:
        distance where irradiance drops by 1/2 [m]
    """
    return np.pi * w0**2/lambda0


def beam_radius(w0, lambda0, z, M2=1, z0=0, model='laboratory'):
    """
    Return the beam radius at an axial location.

    Args:
        w0 : minimum beam radius [m]
        z0 : axial location of beam waist [m]
        M2 : beam propagation factor [-]
        lambda0: wavelength of light [m]
        z : axial location of desired beam radius [m]
    Returns:
        Beam radius [m]
    """
    zz = (z-z0)/z_rayleigh(w0, lambda0)

    if model in ('illuminator', 'constant waist'):
        return w0*np.sqrt(1+(M2*zz)**2)

    if model in ('laboratory', 'constant divergence'):
        return w0*np.sqrt(M2**2+zz**2)

    return w0*M2*np.sqrt(1+zz**2)


def focused_diameter(f, lambda0, d, M2=1):
    """
    Diameter of diffraction-limited focused beam.

    see eq 6b from Roundy, "Current Technology of Beam Profile Measurements"
    in Laser Beam Shaping: Theory and Techniques by Dickey, 2000

    Args:
        lambda0: wavelength of light [m]
        f : focal length of lens [m]
        d : diameter of limiting aperture [m]
        M2: beam propagation factor
    Returns:
        Beam diameter [m]
    """
    return 4 * M2**2 * lambda0 * f / (np.pi * d)

def _quad_fn(z, a, b, c):
    return a + b*z + c*z*z

def abc_fit(z, d):
    """
    Return the hyperbolic fit to the measured diameters.

    Follows ISO 11146-1 section 9

    d(z)**2 = a + b*z + c*z**2

    Args:
        z : axial position of beam measurement [mm]
        d : beam diameter [mm]
    Returns:
        coefficients and their errors
    """
    # fit data using SciPy's Levenberg-Marquart method
    nlfit, nlpcov = np.polyfit(z, d**2, 2, cov=True)

    # unpack fitting parameters
    c, b, a = nlfit

    # unpack uncertainties in fitting parameters from diagonal of covariance matrix
    delta_c, delta_b, delta_a = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]

    return a, b, c, delta_a, delta_b, delta_c


def abc_fit_old(z, d):
    """
    Return the hyperbolic fit to the measured diameters.

    Follows ISO 11146-1 section 9

    d(z)**2 = a + b*z + c*z**2

    Args:
        z : axial position of beam measurement [mm]
        d : beam diameter [mm]
    Returns:
        coefficients and their errors
    """
    # fit data using SciPy's Levenberg-Marquart method
    nlfit, nlpcov = scipy.optimize.curve_fit(_quad_fn, z, d**2)

    # unpack fitting parameters
    a, b, c = nlfit

    # unpack uncertainties in fitting parameters from diagonal of covariance matrix
    delta_a, delta_b, delta_c = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]

    return a, b, c, delta_a, delta_b, delta_c


def beam_params(a, b, c, lambda0):
    """"
    Return beam parameters associated with hyperbolic fit

    d**2(z) = a + b*z + c*z

    Args:
        a, b, c: fitted coefficients
        lambda0: wavelength [mm]
    Returns:
        d0, Theta0, z0, zR, M2
    """
    z0 = -b/(2*c)
    Theta0 = np.sqrt(c)/2

    disc = np.sqrt(4*a*c-b*b)/2
    M2 = np.pi/4/lambda0*disc
    d0 = disc / np.sqrt(c)
    zR = disc/c

    return d0, Theta0, z0, zR, M2


def beam_param_errors(a, b, c, delta_a, delta_b, delta_c, lambda0):
    """"
    Return errors in beam parameters associated with hyperbolic fit

    d**2(z) = a + b*z + c*z

    Args:
        a, b, c: fitted coefficients
        lambda0: wavelength [mm]
    Returns:
        dd0, dTheta0, dz0, dzR, dM2
    """
    d0, _, z0, zR, M2 = beam_params(a, b, c, lambda0)

    delta_z0 = z0*np.sqrt(delta_b**2/b**2 + delta_c**2/c**2)

    delta_d0 = np.sqrt(delta_a**2/4 + (b*delta_b/4/c)**2 + (b**2*delta_c/8/c**2)**2) / d0

    delta_Theta0 = delta_c/2/np.sqrt(c)

    delta_zR = np.sqrt(4*c**4*delta_a**2 + b**2*c**2*delta_b**2 + (b**2-2*a*c)**2*delta_c**2)/(4*c**3) / zR

    delta_M2 = np.pi**2 * np.sqrt(4*c**2*delta_a**2 + b**2*delta_b**2 + 4*a**2*delta_c**2)/(64*lambda0**2) / M2

    return delta_d0, delta_Theta0, delta_z0, delta_zR, delta_M2


def M2_graph(z, d, lambda0, extra=0.2):

    c, b, a = np.polyfit(z, d**2, 2)
    d0, Theta0, z0, zR, M2 = beam_params(a, b, c, lambda0)

    zz = np.linspace(min(z)*(1-extra), max(z)*(1+extra), 100)
    ffit = np.sqrt(a + b * zz + c * zz**2)
    plt.plot(zz, ffit/2, ':k')
    plt.plot(zz, -ffit/2, ':k')

    plt.plot(z, d/2, 'ob', markersize=2)
    plt.plot(z, -d/2, 'ob', markersize=2)
    plt.title(r'$\Theta$=%.2f mradians, $M^2$=%.2f' % (Theta0, M2))

    plt.axvline(z0)
    plt.axvline(z0-zR)

    plt.axhline(w0)
    plt.axhline(-w0)


def M2_report(z, d, lambda0):

    c, b, a = np.polyfit(z, d**2, 2)
    d0, Theta0, z0, zR, M2 = beam_params(a, b, c, lambda0)
    w0 = d0/2

    s = "M2    = %.2f\n" % M2
    s += "w0    = %.2f mm\n" % w0
    s += "Theta = %.2f milliradians\n" % (1000*Theta0)
    s += "zR    = %.0f mm\n" % zR
    s += "z0    = %.0f mm\n" % z0

    return s


def M2_analysis(z, dx, dy, lambda0, f):

    cx, bx, ax = np.polyfit(z, dx**2, 2)
    d0x, Theta0x, z0x, zRx, M2x = beam_params(ax, bx, cx, lambda0)
    w0x = d0x/2

    cy, by, ay = np.polyfit(z, dy**2, 2)
    d0y, Theta0y, z0y, zRy, M2y = beam_params(ay, by, cy, lambda0)
    w0y = d0y/2

#    w0 = np.sqrt((w0x**2 + w0y**2)/2)
#    delta_z0 = abs(z0x - z0y)

    s = "M2    = %.2f\n" % M2x
    s += "w0    = %.2f mm\n" % w0x
    s += "Theta = %.2f milliradians\n" % (1000*Theta0x)
    s += "zR    = %.0f mm\n" % zRx
    s += "beam waist z0    = %.0f mm\n" % z0x

    s = "M2    = %.2f\n" % M2y
    s += "w0    = %.2f mm\n" % w0y
    s += "Theta = %.2f milliradians\n" % (1000*Theta0y)
    s += "zR    = %.0f mm\n" % zRy
    s += "beam waist z0    = %.0f mm\n" % z0y

    return s


def M2_report2(z, dx, dy, phi, lambda0, f=None):

    a, b, c, delta_a, delta_b, delta_c = lbs.abc_fit(z, dx)
    d0x, Theta0x, z0x, zRx, M2x = lbs.beam_params(a, b, c, lambda0)
    dd0x, dTheta0x, dz0x, dzRx, dM2x = lbs.beam_param_errors(a, b, c, delta_a, delta_b, delta_c, lambda0)

    a, b, c, delta_a, delta_b, delta_c = lbs.abc_fit(z, dy)
    d0y, Theta0y, z0y, zRy, M2y = lbs.beam_params(a, b, c, lambda0)
    dd0y, dTheta0y, dz0y, dzRy, dM2y = lbs.beam_param_errors(a, b, c, delta_a, delta_b, delta_c, lambda0)

    w0x = d0x/2
    w0y = d0y/2

    z0 = (z0x + z0y) / 2
    dz0 = np.sqrt((dz0x**2+dz0y**2)/2)

    d0 = (d0x + d0y) / 2
    dd0 = np.sqrt((dd0x**2+dd0y**2)/2)

    zR = (zRx + zRy) / 2
    dzR = np.sqrt((dzRx**2+dzRy**2)/2)

    Theta0 = (Theta0x + Theta0y) / 2
    dTheta0 = np.sqrt((dTheta0x**2+dTheta0y**2)/2)

    M2 = (M2x + M2y) / 2
    dM2 = np.sqrt((dM2x**2+dM2y**2)/2)

    phi_ave = np.degrees(np.mean(phi))
    phi_std = np.degrees(np.std(phi))

    tag = ''
    if f is not None:
        tag = " of the focused beam"

    s = "Beam propagation parameters derived from hyperbolic fit\n"
    s += "Beam waist location%s\n"  %tag
    s += "        z0 = %.0f ± %.0f mm\n" % (z0, dz0)
    s += "       z0x = %.0f ± %.0f mm\n" % (z0x, dz0x)
    s += "       z0y = %.0f ± %.0f mm\n" % (z0y, dz0y)

    s += "Beam waist diameter%s\n"  %tag
    s += "        d0 = %.2f ± %.2f mm\n" % (d0, dd0)
    s += "       d0x = %.2f ± %.2f mm\n" % (d0x, dd0x)
    s += "       d0y = %.2f ± %.2f mm\n" % (d0y, dd0y)

    s += "Azimuthal Angle%s\n"  %tag
    s += "       phi = %.0f ± %.0f°\n" % (phi_ave, phi_std)

    s += "Rayleigh Length%s\n"  %tag
    s += "        zR = %.0f ± %.0f mm\n" % (zR, dzR)
    s += "       zRx = %.0f ± %.0f mm\n" % (zRx, dzRx)
    s += "       zRy = %.0f ± %.0f mm\n" % (zRy, dzRy)

    s += "Divergence Angle%s\n"  %tag
    s += "     theta = %.2f ± %.2f milliradians\n" % (Theta0*1000, dTheta0*1000)
    s += "   theta_x = %.2f ± %.2f milliradians\n" % (Theta0x*1000, dTheta0x*1000)
    s += "   theta_y = %.2f ± %.2f milliradians\n" % (Theta0y*1000, dTheta0y*1000)

    s += "Beam Propagation Ratio%s\n"  %tag
    s += "        M2 = %.2f ± %.2f\n" % (M2, dM2)
    s += "       M2x = %.2f ± %.2f\n" % (M2x, dM2x)
    s += "       M2y = %.2f ± %.2f\n" % (M2y, dM2y)

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

def M2_graph2(z, d, lambda0):
    a, b, c, delta_a, delta_b, delta_c = abc_fit(z, d)
    d0, Theta0, z0, zR, M2 = beam_params(a, b, c, lambda0)
    dd0, dTheta0, dz0, dzR, dM2 = beam_param_errors(a, b, c, delta_a, delta_b, delta_c, lambda0)

    # create fitting function from fitted parameters
    z_fit = np.linspace(min(z), max(z), 128)
    d_fit = np.sqrt(a + b * z_fit + c * z_fit**2)

    # Calculate residuals and reduced chi squared
    resids = d - np.sqrt(a + b * z + c * z**2)

    # Create figure window to plot data
    fig = plt.figure(1, figsize=(8, 8))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[6, 2])

    # Top plot: data and fit
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(z_fit, d_fit, ':k')
    ax1.plot(z, d, 'or')
    ax1.set_xlabel('')
    ax1.set_ylabel('beam diameter (mm)')
    #ax1.text(0.7, 0.95, 'a = {0:0.1f}$\pm${1:0.1f}'.format(a, delta_a), transform = ax1.transAxes)
    #ax1.text(0.7, 0.90, 'b = {0:0.6f}$\pm${1:0.4f}'.format(b, delta_b), transform = ax1.transAxes)
    #ax1.text(0.7, 0.85, 'c = {0:0.7f}$\pm${1:0.7f}'.format(c, delta_c), transform = ax1.transAxes)
    ax1.text(0.7, 0.95, '$d_0$ = %.2f±%.2f mm' % (d0, dd0), transform=ax1.transAxes)
    theta = Theta0*1000
    dtheta = dTheta0*1000
    ax1.text(0.7, 0.90, r'$\Theta$  = %.2f±%.2f mrad' % (theta, dtheta), transform=ax1.transAxes)

    ax1.text(0.7, 0.80, '$z_0$  = %.0f±%.0f mm' % (z0, dz0), transform=ax1.transAxes)
    ax1.text(0.7, 0.75, '$z_R$  = %.0f±%.0f mm' % (zR, dzR), transform=ax1.transAxes)
    ax1.text(0.7, 0.65, '$M^2$ = %.1f±%.1f ' % (M2, dM2), transform=ax1.transAxes)

    ax1.axvline(z0)
    ax1.axvline(z0-zR)

    ax1.axvline(z0+zR)
#    ax1.axhspan(d0+dd0, d0-dd0, alpha=0.2)
    ax1.axhline(d0, color='black', lw=1)
    ax1.set_title('$d(z) = a+bz+cz^2$')

    # Bottom plot: residuals
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(z, resids, "ro")
    ax2.axhline(color="gray", zorder=-1)
    ax2.set_xlabel('axial position $z$ (mm)')
    ax2.set_ylabel('residuals (mm)')
