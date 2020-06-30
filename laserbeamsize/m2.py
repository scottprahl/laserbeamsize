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
import scipy.optimize


__all__ = ('z_rayleigh',
           'beam_radius',
           'focused_diameter',
           'abc_fit',
           'M2_graph',
           'M2_graph2',
           'M2_report',
           'M2_report2',
           'radius_fit_plot',
           )

def z_rayleigh(w0, lambda0):
    """
    Return the Rayleigh distance for a Gaussian beam.

    Args:
        w0: minimum beam radius [m]
        lambda0: wavelength of light [m]
    Returns:
        distance where irradiance drops by 1/2 [m]
    """
    return np.pi * w0**2/lambda0


def beam_radius(w0, lambda0, z, M2=1, z0=0, model='laboratory'):
    """
    Return the beam radius at an axial location.

    Args:
        w0: minimum beam radius [m]
        z0: axial location of beam waist [m]
        M2: beam propagation factor [-]
        lambda0: wavelength of light [m]
        z: axial location of desired beam radius [m]
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
        f: focal length of lens [m]
        d: diameter of limiting aperture [m]
        M2: beam propagation factor [-]
    Returns:
        Beam diameter [m]
    """
    return 4 * M2**2 * lambda0 * f / (np.pi * d)


def abc_fit(z, d, lambda0):
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


def _beam_diameter_squared(z, d0, z0, Theta):
    """Fitting function."""
    return d0**2 + (Theta*(z-z0))**2

def beam_fit(z, d, lambda0):
    """
    Return the hyperbolic fit to the measured diameters.

    Follows ISO 11146-1 section 9 but `a`, `b`, and `c` have been
    replaced by beam parameters `d0`, `z0`, and Theta.  The equation
    for the beam diameter `d(z)` is

    d(z)**2 = d0**2 + Theta**2 * (z-z0)**2

    A non-linear curve fit is done to determine the beam parameters and the
    standard deviations of those parameters.  The beam parameters are returned
    in one array and the errors in a separate array::

        d0: beam waist diameter [m]
        z0: axial location of beam waist [m]
        M2: beam propagation parameter [-]
        Theta: full beam divergence angle [radians]
        zR: Rayleigh distance [m]

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]

    Returns:
        params, errors
    """
    # fit data using SciPy's Levenberg-Marquart method
    nlfit, nlpcov = scipy.optimize.curve_fit(_beam_diameter_squared, z, d**2)

    # unpack fitting parameters
    d0, z0, Theta = nlfit

    # unpack uncertainties in fitting parameters from diagonal of covariance matrix
    d0_std, z0_std, Theta_std = [np.sqrt(nlpcov[j, j]) for j in range(nlfit.size)]

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


def M2_report(z, d, lambda0, f=None):
    """
    Return string describing a single set of beam measurements.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters [m]
        lambda0: wavelength of the laser [m]

    Returns:
        Formatted string suitable for printing.
    """
    params, errors = beam_fit(z, d, lambda0)
    d0, z0, Theta, M2, zR = params
    d0_std, z0_std, Theta_std, M2_std, zR_std = errors

    tag = ''
    if f is not None:
        tag = " of the focused beam"

    s = "Beam propagation parameters derived from hyperbolic fit\n"
    s += "       M^2 = %.2f ± %.2f\n" % (M2, M2_std)
    s += "\n"
    s += "       d_0 = %.0f ± %.0f µm\n" % (d0*1e6, d0_std*1e6)
    s += "       w_0 = %.0f ± %.0f µm\n" % (d0/2*1e6, d0_std/2*1e6)
    s += "\n"
    s += "       z_0 = %.0f ± %.0f mm\n" % (z0*1e3, z0_std*1e3)
    s += "       z_R = %.0f ± %.0f mm\n" % (zR*1e3, zR_std*1e3)
    s += "\n"
    s += "     Theta = %.2f ± %.2f milliradians\n" % (Theta*1e3, Theta_std*1e3)

    return s


def M2_report2(z, dx, dy, lambda0, f=None):
    """
    Return string describing a two sets of beam measurements.

    Args:
        z: array of axial position of beam measurements [m]
        dx: array of beam diameters for semi-major axis [m]
        dy: array of beam diameters for semi-minor axis [m]
        lambda0: wavelength of the laser [m]

    Returns:
        Formatted string suitable for printing.
    """
    params, errors = beam_fit(z, dx, lambda0)
    d0x, z0x, Thetax, M2x, zRx = params
    d0x_std, z0x_std, Thetax_std, M2x_std, zRx_std = errors

    params, errors = beam_fit(z, dy, lambda0)
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

def _fit_plot(z, d, lambda0):
    """
    Helper function that plots the beam and its fit.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]

    Returns:
        residuals, z0, zR
    """
    params, errors = beam_fit(z, d, lambda0)
    d0, z0, Theta, M2, zR = params
    d0_std, z0_std, Theta_std, M2_std, zR_std = errors

    # fitted line
    zmin = min(np.min(z), z0-zR)
    zmax = max(np.max(z), z0+zR)
    z_fit = np.linspace(zmin, zmax)
    d_fit = np.sqrt(d0**2 + (Theta*(z_fit-z0))**2)
    plt.plot(z_fit*1e3, d_fit*1e6, ':k')

    # data points
    plt.plot(z*1e3, d*1e6, 'or')
    plt.xlabel('')
    plt.ylabel('')

    tax = plt.gca().transAxes
    plt.text(0.05, 0.30, '$M^2$ = %.1f±%.1f ' % (M2, M2_std), transform=tax)
    plt.text(0.05, 0.25, '$d_0$ = %.0f±%.0f µm' % (d0*1e6, d0_std*1e6), transform=tax)
    plt.text(0.05, 0.15, '$z_0$  = %.0f±%.0f mm' % (z0*1e3, z0_std*1e3), transform=tax)
    plt.text(0.05, 0.10, '$z_R$  = %.0f±%.0f mm' % (zR*1e3, zR_std*1e3), transform=tax)
    plt.text(0.05, 0.05, r'$\Theta$  = %.2f±%.2f mrad' % (Theta*1e3, Theta_std*1e3), transform=tax)

    plt.axvline(z0*1e3, color='black', lw=1, ls='dashdot')
    plt.axvspan((z0-zR)*1e3, (z0+zR)*1e3, color='blue', alpha=0.1)

    plt.axhline(d0*1e6, color='black', lw=1)
    plt.axhspan((d0+d0_std)*1e6, (d0-d0_std)*1e6, color='red', alpha=0.1)
    plt.title(r'$d^2(z) = d_0^2 + M^4 \Theta^2 (z-z_0)^2$')

    residuals = d - np.sqrt(d0**2 + (Theta*(z-z0))**2)
    return residuals, z0, zR


def M2_graph(z, d, lambda0):
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
    residualsx, z0, zR = _fit_plot(z, d, lambda0)
    plt.ylabel('beam diameter (µm)')
    plt.ylim(0, 1.1*max(d)*1e6)

    fig.add_subplot(gs[1])
    plt.plot(z*1e3, residualsx*1e6, "ro")
    plt.axhline(color="gray", zorder=-1)
    plt.xlabel('axial position $z$ (mm)')
    plt.ylabel('residuals (µm)')
    plt.axvspan((z0-zR)*1e3, (z0+zR)*1e3, color='blue', alpha=0.1)


def M2_graph2(z, dx, dy, lambda0):
    """
    Plot the semi-major and semi-minor beam fits and residuals.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]

    Returns:
        nothing
    """
    ymax = 1.1 * max(np.max(dx), np.max(dy)) * 1e6

    # Create figure window to plot data
    fig = plt.figure(1, figsize=(12, 8))
    gs = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[6, 2])

    fig.add_subplot(gs[0, 0])
    residualsx, z0, zR = _fit_plot(z, dx, lambda0)
    plt.ylabel('beam diameter (µm)')
    plt.title('Semi-major Axis Diameters')
    plt.ylim(0, ymax)

    fig.add_subplot(gs[1, 0])
    ax = plt.gca()
    plt.plot(z*1e3, residualsx*1e6, "ro")
    plt.axhline(color="gray", zorder=-1)
    plt.xlabel('axial position $z$ (mm)')
    plt.ylabel('residuals (µm)')
    plt.axvspan((z0-zR)*1e3, (z0+zR)*1e3, color='blue', alpha=0.1)

    fig.add_subplot(gs[0, 1])
    residualsy, z0, zR = _fit_plot(z, dy, lambda0)
    plt.title('Semi-minor Axis Diameters')
    plt.ylim(0, ymax)

    ymax = max(np.max(residualsx), np.max(residualsy)) * 1e6
    ymin = min(np.min(residualsx), np.min(residualsy)) * 1e6
    ax.set_ylim(ymin, ymax)

    fig.add_subplot(gs[1, 1])
    plt.plot(z*1e3, residualsy*1e6, "ro")
    plt.axhline(color="gray", zorder=-1)
    plt.xlabel('axial position $z$ (mm)')
    plt.ylabel('')
    plt.axvspan((z0-zR)*1e3, (z0+zR)*1e3, color='blue', alpha=0.1)
    plt.ylim(ymin, ymax)


def radius_fit_plot(z, d, lambda0):
    """
    Plot radii, beam fits, and asymptotes.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]

    Returns:
        nothing
    """
    params, errors = beam_fit(z, d, lambda0)
    d0, z0, Theta, M2, zR = params
    d0_std, z0_std, Theta_std, M2_std, zR_std = errors

    fig = plt.figure(1, figsize=(12, 8))

    # fitted line
    zmin = min(np.min(z-z0), -4*zR) * 1.05 + z0
    zmax = max(np.max(z-z0), +4*zR) * 1.05 + z0
    plt.xlim((zmin-z0)*1e3,(zmax-z0)*1e3)

    z_fit = np.linspace(zmin, zmax)
    d_fit = np.sqrt(d0**2 + (Theta*(z_fit-z0))**2)
    plt.plot((z_fit-z0)*1e3, d_fit*1e6/2, ':r')
    plt.plot((z_fit-z0)*1e3, -d_fit*1e6/2, ':r')
    d_fit_lo = np.sqrt((d0-d0_std)**2 + ((Theta-Theta_std)*(z_fit-z0))**2)
    d_fit_hi = np.sqrt((d0+d0_std)**2 + ((Theta+Theta_std)*(z_fit-z0))**2)

    # asymptotes
    r_left = -(z0-zmin)*np.tan(Theta/2)*1e6
    r_right = (zmax-z0)*np.tan(Theta/2)*1e6
    plt.plot([(zmin-z0)*1e3, (zmax-z0)*1e3], [r_left, r_right], '--b')
    plt.plot([(zmin-z0)*1e3, (zmax-z0)*1e3], [-r_left, -r_right], '--b')

    # xticks
    ticks = [(i*zR)*1e3 for i in range(int((zmin-z0)/zR),int((zmax-z0)/zR)+1)]
    ticklabels1 = ["%.0f" % (z+z0*1e3) for z in ticks]
    ticklabels2 = [r"%d $z_R$" % round(z/1e3/zR) for z in ticks]
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(ticklabels1, fontsize=14)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(ticklabels2, fontsize=14)

    ax1.set_xlabel('Axial location in laboratory (mm)', fontsize=14)
    ax1.set_ylabel('Beam radius (µm)', fontsize=14)
#    ax2.set_xlabel('Axial location relative to beam waist (Rayleigh distances)', fontsize=14)
    plt.title('$M^2$ = %.2f±%.2f, $\lambda$=%.0f nm' % (M2, M2_std, lambda0*1e9), fontsize=16)


    tax = plt.gca().transAxes
#     plt.text(0.5, 0.95, '$M^2$ = %.1f±%.1f ' % (M2, M2_std), transform=tax, ha='center', fontsize=16, bbox=dict(facecolor='white',edgecolor='white'))
#     plt.text(0.6, 0.5, r'$\Theta$  = %.2f±%.2f mrad' % (Theta*1e3, Theta_std*1e3), transform=tax, ha='left', va='center', fontsize=16, bbox=dict(facecolor='white',edgecolor='white'))
#     plt.text(0.5, 0.03, '$|z-z_0|<z_R$', transform=tax, ha='center', fontsize=16, bbox=dict(facecolor='white',edgecolor='white'))
#     plt.text(0.85, 0.03, '$2z_R < |z-z_0|$', transform=tax, ha='center', fontsize=16, bbox=dict(facecolor='white',edgecolor='white'))
#     plt.text(0.15, 0.03, '$|z-z_0|>2z_R$', transform=tax, ha='center', fontsize=16, bbox=dict(facecolor='white',edgecolor='white'))
#    plt.text(0.5, 0.95, '$M^2$ = %.1f±%.1f ' % (M2, M2_std), transform=tax, ha='center', fontsize=16)
    plt.text(0.5, 0.03, '$-z_R<z-z_0<z_R$', transform=tax, ha='center', fontsize=16)
    plt.text(0.85, 0.03, '$2z_R < z-z_0$', transform=tax, ha='center', fontsize=16)
    plt.text(0.15, 0.03, '$z-z_0 < -2z_R$', transform=tax, ha='center', fontsize=16)
    plt.text(2*zR*1e3, 0, r'$\Theta$  = %.2f±%.2f mrad' % (Theta*1e3, Theta_std*1e3), ha='left', va='center', fontsize=16)
    arc_x = 1.5*zR*1e3
    arc_y = 1.5*zR*np.tan(Theta/2)*1e6
    plt.annotate('', (arc_x, -arc_y), (arc_x, arc_y),
                 arrowprops=dict(arrowstyle="<->",connectionstyle="arc3,rad=-0.2"))

#    plt.axvline(0, color='black', lw=1, ls='dashdot')
    ax1.axvspan((-zR)*1e3, (+zR)*1e3, color='yellow', alpha=0.2)
    ax1.axvspan((-2*zR)*1e3, (zmin-z0)*1e3, color='yellow', alpha=0.2)
    ax1.axvspan((+2*zR)*1e3, (zmax-z0)*1e3, color='yellow', alpha=0.2)

#    plt.axhline(d0*1e6, color='black', lw=1)
#    plt.axhspan((d0+d0_std)*1e6, (d0-d0_std)*1e6, color='red', alpha=0.1)
#    s = r'$w^2(z) = w_0^2 + (M^4 \Theta^2/4) (z-z_0)^2$'
#    s += r"  $M^2$=%.2f," % M2
#    s += r"  $\Theta$=%.1f mrad" % (1000 * Theta)
#    plt.title(s)
#    ax1.grid(True)
    ax1.fill_between((z_fit-z0)*1e3, d_fit_lo*1e6/2, d_fit_hi*1e6/2, color='red', alpha=0.5)
    ax1.fill_between((z_fit-z0)*1e3, -d_fit_lo*1e6/2, -d_fit_hi*1e6/2, color='red', alpha=0.5)

    # data points
    ax1.plot((z-z0)*1e3, d*1e6/2, 'ok')
    ax1.plot((z-z0)*1e3, -d*1e6/2, 'ok')


