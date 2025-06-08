"""
A module for finding M² values for a laser beam.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

Finding the beam waist size, location, and M² for a beam is straightforward::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import laserbeamsize as lbs
    >>>
    >>> lambda0 = 632.8e-9  # meters
    >>> z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770]) * 1e-3
    >>> r = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397]) * 1e-6
    >>>
    >>> # create a graphic of the fit
    >>> lbs.M2_diameter_plot(z, 2 * r, lambda0)
    >>> plt.show()
    >>>
    >>> # create a better graphic of the fit
    >>> lbs.M2_radius_plot(z, 2 * r, lambda0)
    >>> plt.show()
"""

import numpy as np
import matplotlib.gridspec
import matplotlib.pyplot as plt
import laserbeamsize as lbs

__all__ = ("M2_diameter_plot", "M2_radius_plot", "M2_focus_plot")


def _fit_plot(z, d, lambda0, strict=False, z0=None, d0=None):
    """
    Plot beam diameters and ISO 11146 fit.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]
        strict: (optional) boolean for strict usage of ISO 11146
        z0: (optional) axial location of beam waist [m]
        d0: (optional) beam waist diameter [m]

    Returns:
        residuals: array with differences between fit and data
        z0: location of focus
        zR: Rayleigh distance for beam
    """
    params, errors, used = lbs.M2_fit(z, d, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    d0, z0, Theta, M2, zR = params
    d0_std, z0_std, Theta_std, M2_std, zR_std = errors

    # fitted line
    zmin = min(np.min(z), z0 - 4 * zR)
    zmax = max(np.max(z), z0 + 4 * zR)
    #    plt.xlim(zmin, zmax)
    z_fit = np.linspace(zmin, zmax)
    #    d_fit = np.sqrt(d0**2 + (Theta * (z_fit - z0))**2)
    #    plt.plot(z_fit * 1e3, d_fit * 1e6, ':k')
    d_fit_lo = np.sqrt((d0 - d0_std) ** 2 + ((Theta - Theta_std) * (z_fit - z0)) ** 2)
    d_fit_hi = np.sqrt((d0 + d0_std) ** 2 + ((Theta + Theta_std) * (z_fit - z0)) ** 2)
    plt.fill_between(z_fit * 1e3, d_fit_lo * 1e6, d_fit_hi * 1e6, color="red", alpha=0.5)

    # show perfect gaussian caustic when unphysical M2 arises
    if M2 < 1:
        Theta00 = 4 * lambda0 / (np.pi * d0)
        d_00 = np.sqrt(d0**2 + (Theta00 * (z_fit - z0)) ** 2)
        plt.plot(z_fit * 1e3, d_00 * 1e6, ":k", lw=2, label="M²=1")
        plt.legend(loc="lower right")

    plt.fill_between(z_fit * 1e3, d_fit_lo * 1e6, d_fit_hi * 1e6, color="red", alpha=0.5)
    # data points
    plt.plot(z[used] * 1e3, d[used] * 1e6, "o", color="black", label="used")
    plt.plot(z[unused] * 1e3, d[unused] * 1e6, "ok", mfc="none", label="unused")
    plt.xlabel("")
    plt.ylabel("")

    tax = plt.gca().transAxes
    plt.text(0.05, 0.30, "$M^2$ = %.2f±%.2f " % (M2, M2_std), transform=tax)
    plt.text(0.05, 0.25, "$d_0$ = %.0f±%.0f µm" % (d0 * 1e6, d0_std * 1e6), transform=tax)
    plt.text(0.05, 0.15, "$z_0$  = %.0f±%.0f mm" % (z0 * 1e3, z0_std * 1e3), transform=tax)
    plt.text(0.05, 0.10, "$z_R$  = %.0f±%.0f mm" % (zR * 1e3, zR_std * 1e3), transform=tax)
    Theta_ = Theta * 1e3
    Theta_std_ = Theta_std * 1e3
    plt.text(0.05, 0.05, r"$\Theta$  = %.2f±%.2f mrad" % (Theta_, Theta_std_), transform=tax)

    plt.axvline(z0 * 1e3, color="black", lw=1)
    plt.axvspan((z0 - zR) * 1e3, (z0 + zR) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0 - 2 * zR) * 1e3, (zmin) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0 + 2 * zR) * 1e3, (zmax) * 1e3, color="cyan", alpha=0.3)

    #    plt.axhline(d0 * 1e6, color='black', lw=1)
    #    plt.axhspan((d0 + d0_std) * 1e6, (d0 - d0_std) * 1e6, color='red', alpha=0.1)
    plt.title(r"$d^2(z) = d_0^2 + \Theta^2 (z - z_0)^2$")
    if sum(z[unused]) > 0:
        plt.legend(loc="upper right")

    residuals = d - np.sqrt(d0**2 + (Theta * (z - z0)) ** 2)
    return residuals, z0, zR, used


def _M2_diameter_plot(z, d, lambda0, strict=False, z0=None, d0=None):
    """
    Plot the fitted beam and the residuals.

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]
        strict: (optional) boolean for strict usage of ISO 11146
        z0: (optional) axial location of beam waist [m]
        d0: (optional) beam waist diameter [m]

    Returns:
        nothing
    """
    fig = plt.figure(1, figsize=(12, 8))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[6, 2])

    fig.add_subplot(gs[0])
    residualsx, z0, zR, used = _fit_plot(z, d, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    zmin = min(np.min(z), z0 - 4 * zR)
    zmax = max(np.max(z), z0 + 4 * zR)

    plt.ylabel("beam diameter (µm)")
    plt.ylim(0, 1.1 * max(d) * 1e6)

    fig.add_subplot(gs[1])
    plt.plot(z * 1e3, residualsx * 1e6, "ro")
    plt.plot(z[used] * 1e3, residualsx[used] * 1e6, "ok", label="used")
    plt.plot(z[unused] * 1e3, residualsx[unused] * 1e6, "ok", mfc="none", label="unused")

    plt.axhline(color="gray", zorder=-1)
    plt.xlabel("axial position $z$ (mm)")
    plt.ylabel("residuals (µm)")
    plt.axvspan((z0 - zR) * 1e3, (z0 + zR) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0 - 2 * zR) * 1e3, (zmin) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0 + 2 * zR) * 1e3, (zmax) * 1e3, color="cyan", alpha=0.3)


def M2_diameter_plot(z, d_major, lambda0, d_minor=None, strict=False, z0=None, d0=None):
    """
    Plot the major and minor beam fits and residuals.

    Example::

        >>>> import numpy as np
        >>>> import laserbeamsize as lbs
        >>>> lambda0 = 632.8e-9  # meters
        >>>> z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770])
        >>>> r = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397])
        >>>> lbs.M2_diameter_plot(z * 1e-3, 2 * r * 1e-6, lambda0)
        >>>> plt.show()

    Args:
        z: array of axial position of beam measurements [m]
        lambda0: wavelength of the laser [m]
        d_major: array of major axis beam diameters [m]
        d_minor: (optional) array of minor axis beam diameters [m]
        strict: (optional) boolean for strict usage of ISO 11146
        z0: (optional) axial location of beam waist [m]
        d0: (optional) beam waist diameter [m]

    Returns:
        nothing
    """
    if d_minor is None:
        _M2_diameter_plot(z, d_major, lambda0, strict=strict, z0=z0, d0=d0)
        return

    ymax = 1.1 * max(np.max(d_major), np.max(d_minor)) * 1e6

    # Create figure window to plot data
    fig = plt.figure(1, figsize=(12, 8))
    gs = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[6, 2])

    # major axis plot
    fig.add_subplot(gs[0, 0])
    residualsx, z0x, zR, used = _fit_plot(z, d_major, lambda0, strict=strict, z0=z0, d0=d0)

    zmin = min(np.min(z), z0x - 4 * zR)
    zmax = max(np.max(z), z0x + 4 * zR)
    unused = np.logical_not(used)
    plt.ylabel("beam diameter (µm)")
    plt.title("Semi-major Axis Diameters")
    plt.ylim(0, ymax)

    # major axis residuals
    fig.add_subplot(gs[1, 0])
    ax = plt.gca()
    plt.plot(z[used] * 1e3, residualsx[used] * 1e6, "ok", label="used")
    plt.plot(z[unused] * 1e3, residualsx[unused] * 1e6, "ok", mfc="none", label="unused")
    plt.axhline(color="gray", zorder=-1)
    plt.xlabel("axial position $z$ (mm)")
    plt.ylabel("residuals (µm)")
    plt.axvspan((z0x - zR) * 1e3, (z0x + zR) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0x - 2 * zR) * 1e3, (zmin) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0x + 2 * zR) * 1e3, (zmax) * 1e3, color="cyan", alpha=0.3)

    # minor axis plot
    fig.add_subplot(gs[0, 1])
    residualsy, z0y, zR, used = _fit_plot(z, d_minor, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    plt.title("Minor Axis Diameters")
    plt.ylim(0, ymax)

    ymax = max(np.max(residualsx), np.max(residualsy)) * 1e6
    ymin = min(np.min(residualsx), np.min(residualsy)) * 1e6
    ax.set_ylim(ymin, ymax)

    # minor axis residuals
    fig.add_subplot(gs[1, 1])
    plt.plot(z[used] * 1e3, residualsy[used] * 1e6, "ok", label="used")
    plt.plot(z[unused] * 1e3, residualsy[unused] * 1e6, "ok", mfc="none", label="unused")
    plt.axhline(color="gray", zorder=-1)
    plt.xlabel("axial position $z$ (mm)")
    plt.ylabel("")
    plt.axvspan((z0y - zR) * 1e3, (z0y + zR) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0y - 2 * zR) * 1e3, (zmin) * 1e3, color="cyan", alpha=0.3)
    plt.axvspan((z0y + 2 * zR) * 1e3, (zmax) * 1e3, color="cyan", alpha=0.3)
    plt.ylim(ymin, ymax)


def M2_radius_plot(z, d, lambda0, strict=False, z0=None, d0=None):
    """
    Plot radii, beam fits, and asymptotes.

    Example::

        >>>> import numpy as np
        >>>> import laserbeamsize as lbs
        >>>> lambda0 = 632.8e-9  # meters
        >>>> z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770])
        >>>> r = np.array([597, 572, 547, 554, 479, 403, 415, 400, 377, 391, 326, 397])
        >>>> lbs.M2_radius_plot(z * 1e-3, 2 * r * 1e-6, lambda0)
        >>>> plt.show()

    Args:
        z: array of axial position of beam measurements [m]
        d: array of beam diameters  [m]
        lambda0: wavelength of the laser [m]
        strict: (optional) boolean for strict usage of ISO 11146
        z0: (optional) axial location of beam waist [m]
        d0: (optional) beam waist diameter [m]

    Returns:
        nothing
    """
    params, errors, used = lbs.M2_fit(z, d, lambda0, strict=strict, z0=z0, d0=d0)
    unused = np.logical_not(used)
    d0, z0, Theta, M2, zR = params
    d0_std, _, Theta_std, M2_std, _ = errors

    plt.figure(1, figsize=(12, 8))

    # fitted line
    zmin = min(np.min(z - z0), -4 * zR) * 1.05 + z0
    zmax = max(np.max(z - z0), +4 * zR) * 1.05 + z0
    plt.xlim((zmin - z0) * 1e3, (zmax - z0) * 1e3)

    z_fit = np.linspace(zmin, zmax)
    d_fit = np.sqrt(d0**2 + (Theta * (z_fit - z0)) ** 2)
    #    plt.plot((z_fit - z0) * 1e3, d_fit * 1e6 / 2, ':r')
    #    plt.plot((z_fit - z0) * 1e3, -d_fit * 1e6 / 2, ':r')
    d_fit_lo = np.sqrt((d0 - d0_std) ** 2 + ((Theta - Theta_std) * (z_fit - z0)) ** 2)
    d_fit_hi = np.sqrt((d0 + d0_std) ** 2 + ((Theta + Theta_std) * (z_fit - z0)) ** 2)

    # asymptotes
    r_left = -(z0 - zmin) * np.tan(Theta / 2) * 1e6
    r_right = (zmax - z0) * np.tan(Theta / 2) * 1e6
    plt.plot([(zmin - z0) * 1e3, (zmax - z0) * 1e3], [r_left, r_right], "--b")
    plt.plot([(zmin - z0) * 1e3, (zmax - z0) * 1e3], [-r_left, -r_right], "--b")

    # xticks along top axis
    ticks = [(i * zR) * 1e3 for i in range(int((zmin - z0) / zR), int((zmax - z0) / zR) + 1)]
    ticklabels1 = ["%.0f" % (z + z0 * 1e3) for z in ticks]
    ticklabels2 = []
    for i in range(int((zmin - z0) / zR), int((zmax - z0) / zR) + 1):
        if i == 0:
            ticklabels2 = np.append(ticklabels2, "0")
        elif i == -1:
            ticklabels2 = np.append(ticklabels2, r"-$z_R$")
        elif i == 1:
            ticklabels2 = np.append(ticklabels2, r"$z_R$")
        else:
            ticklabels2 = np.append(ticklabels2, r"%d$z_R$" % i)
    ax1 = plt.gca()
    ax2 = ax1.twiny()

    ax1.set_xticks(ticks)
    if len(ticks) > 10:
        ax1.set_xticklabels(ticklabels1, fontsize=14, rotation=90)
    else:
        ax1.set_xticklabels(ticklabels1, fontsize=14)

    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticks(ticks)
    if len(ticks) > 10:
        ax2.set_xticklabels(ticklabels2, fontsize=14, rotation=90)
    else:
        ax2.set_xticklabels(ticklabels2, fontsize=14)

    # usual labels for graph
    ax1.set_xlabel("Axial Location (mm)", fontsize=14)
    ax1.set_ylabel("Beam radius (µm)", fontsize=14)
    title = r"$w_0=d_0/2$=%.0f±%.0fµm,  " % (d0 / 2 * 1e6, d0_std / 2 * 1e6)
    title += r"$M^2$ = %.2f±%.2f,  " % (M2, M2_std)
    title += r"$\lambda$=%.0f nm" % (lambda0 * 1e9)
    plt.title(title, fontsize=16)

    # show the divergence angle
    s = r"$\Theta$  = %.2f±%.2f mrad" % (Theta * 1e3, Theta_std * 1e3)
    plt.text(2 * zR * 1e3, 0, s, ha="left", va="center", fontsize=16)
    arc_x = 1.5 * zR * 1e3
    arc_y = 1.5 * zR * np.tan(Theta / 2) * 1e6
    plt.annotate(
        "",
        (arc_x, -arc_y),
        (arc_x, arc_y),
        arrowprops={"arrowstyle": "<->", "connectionstyle": "arc3, rad=-0.2"},
    )

    # show the Rayleigh ranges
    ymin = max(np.max(d_fit), np.max(d))
    ymin *= -1 / 2 * 1e6
    plt.text(0, ymin, "$-z_R<z-z_0<z_R$", ha="center", va="bottom", fontsize=16)
    x = (zmax - z0 + 2 * zR) / 2 * 1e3
    plt.text(x, ymin, "$2z_R < z-z_0$", ha="center", va="bottom", fontsize=16)
    x = (zmin - z0 - 2 * zR) / 2 * 1e3
    plt.text(x, ymin, "$z-z_0 < -2z_R$", ha="center", va="bottom", fontsize=16)

    ax1.axvspan((-zR) * 1e3, (+zR) * 1e3, color="cyan", alpha=0.3)
    ax1.axvspan((-2 * zR) * 1e3, (zmin - z0) * 1e3, color="cyan", alpha=0.3)
    ax1.axvspan((+2 * zR) * 1e3, (zmax - z0) * 1e3, color="cyan", alpha=0.3)

    # show the fit
    zz = (z_fit - z0) * 1e3
    lo = d_fit_lo * 1e6 / 2
    hi = d_fit_hi * 1e6 / 2
    ax1.fill_between(zz, lo, hi, color="red", alpha=0.5)
    ax1.fill_between(zz, -lo, -hi, color="red", alpha=0.5)

    # show perfect gaussian caustic when unphysical M2 arises
    if M2 < 1:
        Theta00 = 4 * lambda0 / (np.pi * d0)
        r_00 = np.sqrt(d0**2 + (Theta00 * zz * 1e-3) ** 2) / 2 * 1e6
        plt.plot(zz, r_00, ":k", lw=2, label="M²=1")
        plt.plot(zz, -r_00, ":k", lw=2)
        plt.legend(loc="lower right")

    # data points
    ax1.plot((z[used] - z0) * 1e3, d[used] * 1e6 / 2, "ok", label="used")
    ax1.plot((z[used] - z0) * 1e3, -d[used] * 1e6 / 2, "ok")
    ax1.plot((z[unused] - z0) * 1e3, d[unused] * 1e6 / 2, "ok", mfc="none", label="unused")
    ax1.plot((z[unused] - z0) * 1e3, -d[unused] * 1e6 / 2, "ok", mfc="none")
    if sum(z[unused]) > 0:
        ax1.legend(loc="center left")


def M2_focus_plot(w0, lambda0, f, z0, M2=1):
    """
    Plot a beam from its waist through a lens to its focus.

    After calling this, use `plt.show()` to display the plot.

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
    left = 1.1 * z0
    z = np.linspace(left, 0)
    r = lbs.beam_radius(w0, lambda0, z, z0=z0, M2=M2)
    plt.fill_between(z * 1e3, -r * 1e6, r * 1e6, color="red", alpha=0.2)

    # find the gaussian beam parameters for the beam after the lens
    w0_after = w0 * lbs.magnification(w0, lambda0, z0, f, M2=M2)
    z0_after = lbs.image_distance(w0, lambda0, z0, f, M2=M2)
    zR_after = lbs.z_rayleigh(w0_after, lambda0, M2)

    # plot the beam after the lens
    right = max(2 * f, z0_after + 4 * zR_after)
    z_after = np.linspace(0, right)
    r_after = lbs.beam_radius(w0_after, lambda0, z_after, z0=z0_after, M2=M2)

    # plt.axhline(w0_after * 1.41e6)
    plt.fill_between(z_after * 1e3, -r_after * 1e6, r_after * 1e6, color="red", alpha=0.2)

    # locate the lens and the two beam waists
    plt.axhline(0, color="black", lw=1)
    plt.axvline(0, color="black")
    plt.axvline(z0 * 1e3, color="black", linestyle=":")
    plt.axvline(z0_after * 1e3, color="black", linestyle=":")

    # finally, show the ±1 Rayleigh distance
    zRmin = max(0, (z0_after - zR_after)) * 1e3
    zRmax = (z0_after + zR_after) * 1e3
    plt.axvspan(zRmin, zRmax, color="blue", alpha=0.1)

    plt.xlabel("Axial Position Relative to Lens (mm)")
    plt.ylabel("Beam Radius (microns)")
    title = "$w_0$=%.0fµm, $z_0$=%.0fmm, " % (w0 * 1e6, z0 * 1e3)
    title += "$w_0'$=%.0fµm, $z_0'$=%.0fmm, " % (w0_after * 1e6, z0_after * 1e3)
    title += "$z_R'$=%.0fmm" % (zR_after * 1e3)
    plt.title(title)
