"""
A module for generating a graphical analysis of beam size fitting.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

A graphic showing the image and extracted beam parameters is achieved by::

    >>> import imageio.v3 as iio
    >>> import matplotlib.pyplot as plt
    >>> import laserbeamsize as lbs
    >>>
    >>> repo = "https://github.com/scottprahl/laserbeamsize/raw/main/docs/"
    >>> image = iio.imread(repo + 't-hene.pgm')
    >>>
    >>> lbs.plot_image_analysis(image)
    >>> plt.show()

A mosaic of images might be created by::

    >>> import imageio.v3 as iio
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import laserbeamsize as lbs
    >>>
    >>> repo = "https://github.com/scottprahl/laserbeamsize/raw/main/docs/"
    >>> z1 = np.array([168,210,280,348,414,480], dtype=float)
    >>> fn1 = [repo + "t-%dmm.pgm" % number for number in z1]
    >>> images = [iio.imread(fn) for fn in fn1]
    >>>
    >>> options = {'z':z1/1000, 'pixel_size':0.00375, 'units':'mm', 'crop':True}
    >>> lbs.plot_image_montage(images, **options, iso_noise=False)
    >>> plt.show()
"""

import numpy as np
import numpy.typing as npt
import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .analysis import beam_size
from .background import iso_background, subtract_iso_background
from .image_tools import axes_arrays, crop_image_to_rect, crop_image_to_integration_rect
from .image_tools import ellipse_arrays, rotated_rect_arrays, major_axis_arrays, minor_axis_arrays

__all__ = (
    "beam_ellipticity",
    "plot_beam_diagram",
    "plot_image_analysis",
    "plot_image_and_fit",
    "plot_image_montage",
)


def beam_ellipticity(d_major: float, d_minor: float) -> tuple[float, float]:
    """
    Calculate the ellipticity of the beam.

    The ISO 11146 standard defines ellipticity as the "ratio between the
    minimum and maximum beam widths".  These widths (diameters) returned
    by `beam_size()` can be used to make this calculation.

    When `ellipticity > 0.87`, then the beam profile may be considered to have
    circular symmetry. The equivalent beam diameter is the root mean square
    of the beam diameters.

    Args:
        d_major: x diameter of the beam spot
        d_minor: y diameter of the beam spot
    Returns:
        ellipticity: varies from 0 (line) to 1 (round)
        d_circular: equivalent diameter of a circular beam
    """
    if d_minor < d_major:
        ellipticity = d_minor / d_major
    elif d_major < d_minor:
        ellipticity = d_major / d_minor
    else:
        ellipticity = 1

    d_circular = np.sqrt((d_major**2 + d_minor**2) / 2)

    return ellipticity, d_circular


def plot_beam_diagram() -> None:
    """Draw a simple astigmatic beam ellipse with labels."""
    phi = np.radians(30)
    xc, yc, d_major, d_minor = 0, 0, 50, 25

    plt.subplots(1, 1, figsize=(6, 6))

    # If the aspect ratio is not `equal` then the major and minor axes
    # will not appear to be orthogonal to each other!
    plt.axes().set_aspect("equal")

    xp, yp = ellipse_arrays(xc, yc, d_major, d_minor, phi)
    plt.plot(xp, yp, "k", lw=2)

    scale = 1
    diameters = 3
    rect_major = d_major * diameters
    rect_minor = d_minor * diameters
    xp, yp = rotated_rect_arrays(xc, yc, rect_major, rect_minor, phi) * scale
    plt.plot(xp, yp, ":b", lw=2)

    sint = np.sin(phi) / 2
    cost = np.cos(phi) / 2
    plt.plot([xc - d_major * cost, xc + d_major * cost], [yc + d_major * sint, yc - d_major * sint], ":b")
    plt.plot([xc + d_minor * sint, xc - d_minor * sint], [yc + d_minor * cost, yc - d_minor * cost], ":r")

    # draw axes
    plt.annotate(
        "x",
        xy=(-25, 0),
        xytext=(25, 0),
        arrowprops={"arrowstyle": "<-"},
        va="center",
        fontsize=16,
    )

    plt.annotate(
        "y",
        xy=(0, 25),
        xytext=(0, -25),
        arrowprops={"arrowstyle": "<-"},
        ha="center",
        fontsize=16,
    )

    plt.annotate(r"$\phi$", xy=(13, -2.5), fontsize=16)
    plt.annotate(
        "",
        xy=(15.5, 0),
        xytext=(14, -8.0),
        arrowprops={"arrowstyle": "<-", "connectionstyle": "arc3, rad=-0.2"},
    )

    plt.annotate(r"$d_{major}$", xy=(-17, 7), color="blue", fontsize=16)
    plt.annotate(r"$d_{minor}$", xy=(-4, -8), color="red", fontsize=16)

    plt.xlim(-30, 30)
    plt.ylim(30, -30)  # inverted to match image coordinates!
    plt.axis("off")


def plot_visible_dotted_line(xpts: npt.NDArray[np.floating], ypts: npt.NDArray[np.floating]) -> None:
    """Draw a dotted line that is is visible against images."""
    # White solid line underneath
    plt.plot(xpts, ypts, color="white", linewidth=1, solid_capstyle="round")
    # Black dashes on top
    plt.plot(xpts, ypts, color="black", linewidth=1, linestyle=(0, (3, 2)), solid_capstyle="round")


def set_zero_to_lightgray(cmap_name: str, min_val: float, max_val: float) -> mcolors.ListedColormap:
    """Create a colormap where zero maps to gray."""
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 256))

    # index that corresponds to zero
    idx = 0
    if min_val < 0 <= max_val:
        idx = int(256 * abs(min_val) / (max_val - min_val))
    idx = int(np.clip(idx, 0, len(colors) - 1))

    colors[idx] = [0.827, 0.827, 0.827, 1.0]

    return mcolors.ListedColormap(colors)


def _format_beam_title(d_major: float | None, d_minor: float | None, units: str, z: float | None = None) -> str:
    """
    Return a standardized title string describing the beam diameters.

    If z position is not None, then it should be specified in meters.
    The z position will be converted to mm for display.

    Args:
        d_major: major diameter in units specified
        d_minor: minor diameter in units specified
        units: units for diameters
        z: (optional) if present add z-position to title

    Returns:
        title
    """

    def _fmt(val, label):
        if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            return f"{label} fail"
        if units == "mm":
            return f"{label}={val:.2f}{units}"
        return f"{label}={val:.0f}{units}"

    s1 = _fmt(d_major, "$d_{major}$")
    s2 = _fmt(d_minor, "$d_{minor}$")
    s = f"{s1}, {s2}"

    if z is None:
        return s

    # z is in meters; display in mm as in the montage
    return f"z={z * 1e3:.0f}mm, {s}"


def _prepare_beam_analysis(
    image: np.ndarray, corner_fraction: float, nT: float, iso_noise: bool, **kwargs
) -> tuple[float, float, float, float, float | None, float]:
    """
    Common setup for beam analysis: extract beam_size parameters and calculate beam properties.

    Args:
        image: 2D array of image with beam spot
        corner_fraction: the fractional size of corner rectangles
        nT: how many standard deviations to subtract
        iso_noise: if True then allow negative pixel values
        **kwargs: extra options to pass to beam_size()

    Returns:
        tuple: (diameters, xc_px, yc_px, d_major_px, d_minor_px, phi)
    """
    diameters = kwargs.get("mask_diameters", 3)

    # only pass along arguments that apply to beam_size()
    beamsize_keys = ["mask_diameters", "max_iter", "phi_fixed"]
    bs_args = dict((k, kwargs[k]) for k in beamsize_keys if k in kwargs)
    bs_args["iso_noise"] = iso_noise
    bs_args["nT"] = nT
    bs_args["corner_fraction"] = corner_fraction

    # find center and diameters (all in pixels)
    xc_px, yc_px, d_major_px, d_minor_px, phi = beam_size(image, **bs_args)

    return diameters, xc_px, yc_px, d_major_px, d_minor_px, phi


def _setup_scale_and_labels(pixel_size: float | None, units: str) -> tuple[float, str, str]:
    """
    Determine scaling factor and axis labels.

    Args:
        pixel_size: size of pixels (None for pixel units)
        units: string for physical units

    Returns:
        tuple: (scale, label, units_str)
    """
    if pixel_size is None:
        scale: float = 1
        unit_str = "px"
    else:
        scale = pixel_size
        unit_str = units

    label = "Distance from Center [%s]" % unit_str
    return scale, label, unit_str


def _crop_image_if_needed(
    o_image: np.ndarray,
    xc_px: float,
    yc_px: float,
    d_major_px: float,
    d_minor_px: float | None,
    phi: float,
    crop: bool | list,
    scale: float,
    diameters: float,
) -> tuple[np.ndarray, float, float]:
    """
    Crop the image according to the crop parameter.

    Args:
        o_image: original image
        xc_px: beam center coordinates (in pixels)
        yc_px: beam center coordinates (in pixels)
        d_major_px: major beam diameter (in pixels)
        d_minor_px: minor beam diameter (in pixels)
        phi: angle tilt (in radians)
        crop: cropping specification (False, True, or [v, h] list)
        scale: pixel scaling factor
        diameters: number of diameters for integration rectangle

    Returns:
        tuple: (cropped_image, new_xc_px, new_yc_px)
    """
    if isinstance(crop, list):
        ymin = yc_px - crop[0] / 2 / scale
        ymax = yc_px + crop[0] / 2 / scale
        xmin = xc_px - crop[1] / 2 / scale
        xmax = xc_px + crop[1] / 2 / scale
        cropped, new_xc, new_yc = crop_image_to_rect(o_image, xc_px, yc_px, xmin, xmax, ymin, ymax)
        if cropped is None or new_xc is None or new_yc is None:
            return o_image, xc_px, yc_px
        return cropped, new_xc, new_yc

    if crop:
        return crop_image_to_integration_rect(o_image, xc_px, yc_px, d_major_px, d_minor_px, phi, diameters)

    return o_image, xc_px, yc_px


def _draw_beam_overlays(
    xc_px: float, yc_px: float, d_major_px: float, d_minor_px: float | None, phi: float, diameters: float, scale: float
) -> None:
    """
    Draw ellipse, axes, and integration rectangle on current plot.

    Args:
        xc_px: beam center coordinates (in pixels)
        yc_px: beam center coordinates (in pixels)
        d_major_px: major beam diameter (in pixels)
        d_minor_px: minor beam diameter (in pixels)
        phi: angle tilt (in radians)
        diameters: number of diameters for integration rectangle
        scale: pixel scaling factor
    """
    # Calculate rectangle dimensions (in pixels)
    rect_minor_px = None
    rect_major_px = d_major_px * diameters - 0.5  # -0.5 centres the rect on the pixel grid

    if d_minor_px is not None:
        rect_minor_px = d_minor_px * diameters - 0.5  # -0.5 centres the rect on the pixel grid

        # Draw ellipse around beam
        xp_px, yp_px = ellipse_arrays(xc_px, yc_px, d_major_px, d_minor_px, phi)
        plot_visible_dotted_line((xp_px - xc_px) * scale, (yp_px - yc_px) * scale)

        # Draw integration rectangle around beam
        xp_px, yp_px = rotated_rect_arrays(xc_px - 0.5, yc_px - 0.5, rect_major_px, rect_minor_px, phi)
        plot_visible_dotted_line((xp_px - xc_px) * scale, (yp_px - yc_px) * scale)

    # major and minor axes
    xp1_px, yp1_px, xp2_px, yp2_px = axes_arrays(xc_px, yc_px, rect_major_px, rect_minor_px, phi)

    plot_visible_dotted_line((xp1_px - xc_px) * scale, (yp1_px - yc_px) * scale)
    if d_minor_px is not None and xp2_px is not None and yp2_px is not None:
        plot_visible_dotted_line((xp2_px - xc_px) * scale, (yp2_px - yc_px) * scale)


def _plot_image_with_beam_overlay(
    image: np.ndarray,
    xc_px: float,
    yc_px: float,
    d_major_px: float,
    d_minor_px: float | None,
    phi: float,
    diameters: float,
    scale: float,
    label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar: bool = True,
) -> matplotlib.image.AxesImage:
    """
    Core function to plot an image with beam overlays.

    Used by both plot_image_and_fit and plot_image_analysis.

    Args:
        image: 2D image to display
        xc_px: beam center in pixels
        yc_px: beam center in pixels
        d_major_px: major diameter in pixels
        d_minor_px: minor diameter in pixels
        phi: beam angle in radians
        diameters: integration rectangle size multiplier
        scale: pixel to unit conversion
        label: axis label string
        cmap: colormap
        vmin: optional colorbar minimum
        vmax: optional colorbar maximum
        title: optional plot title
        colorbar: whether to show colorbar

    Returns:
        im: the image object
    """
    v_px, h_px = image.shape
    extent = (-xc_px * scale, (h_px - xc_px) * scale, (v_px - yc_px) * scale, -yc_px * scale)

    # establish colorbar limits
    if vmax is None:
        vmax = image.max()
    if vmin is None:
        vmin = image.min()

    # add gray to cmap around zero
    ccmap = set_zero_to_lightgray(cmap, vmin, vmax)

    # display image
    im = plt.imshow(image, extent=extent, cmap=ccmap, vmax=vmax, vmin=vmin)
    im.cmap.set_bad(color="black")
    plt.xlabel(label)
    plt.ylabel(label)

    # Draw beam overlays (ellipse, axes, integration rectangle)
    _draw_beam_overlays(xc_px, yc_px, d_major_px, d_minor_px, phi, diameters, scale)

    # set limits on axes
    plt.xlim(-xc_px * scale, (h_px - xc_px) * scale)
    plt.ylim((v_px - yc_px) * scale, -yc_px * scale)

    if title:
        plt.title(title)

    # show colorbar
    if colorbar:
        plt.colorbar(im, fraction=0.046 * v_px / h_px, pad=0.04)

    return im


def plot_image_and_fit(
    o_image: np.ndarray,
    pixel_size: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    units: str = "µm",
    crop: bool | list = False,
    colorbar: bool = False,
    cmap: str = "gist_ncar",
    corner_fraction: float = 0.035,
    nT: float = 3,
    iso_noise: bool = True,
    **kwargs,
) -> tuple[float, float, float, float | None, float]:
    """
    Plot the image, fitted ellipse, integration area, and major/minor axes.

    If pixel_size is defined, then the returned measurements are in units of
    pixel_size.

    This function helpful when creating a mosaics of all images captured for an
    experiment.

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    All cropping is done after analysis and therefore only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        o_image: 2D array of image with beam spot
        pixel_size: (optional) size of pixels
        vmin: (optional) minimum value for colorbar
        vmax: (optional) maximum value for colorbar
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        colorbar: (optional) show the color bar,
        cmap: (optional) colormap to use
        corner_fraction: (optional) the fractional size of corner rectangles
        nT: (optional) how many standard deviations to subtract
        iso_noise: (optional) if True then allow negative pixel values
        kwargs: additional arguments passed through to beam_size

    Returns:
        xc, yc, d_major, d_minor, phi
    """
    # Common beam analysis setup
    diameters, xc_px, yc_px, d_major_px, d_minor_px, phi = _prepare_beam_analysis(
        o_image, corner_fraction, nT, iso_noise, **kwargs
    )

    # Setup scale and labels
    scale, label, unit_str = _setup_scale_and_labels(pixel_size, units)

    # Crop image if necessary (analysis is already done on o_image)
    image, xc_px, yc_px = _crop_image_if_needed(
        o_image, xc_px, yc_px, d_major_px, d_minor_px, phi, crop, scale, diameters
    )

    # For display, use the same ISO-11146 background subtraction used by
    # plot_image_analysis (subplot 2,2,2). This drives the background toward
    # zero so that the colormap can put zero at gray.
    working_image = subtract_iso_background(image, corner_fraction=corner_fraction, nT=nT, iso_noise=iso_noise)

    # Convert diameters to the requested units for the title
    d_major = d_major_px * scale
    d_minor = d_minor_px * scale if d_minor_px is not None else None

    title = _format_beam_title(d_major, d_minor, unit_str)

    # Use helper function for plotting; vmin/vmax still honored but now apply
    # to the background-subtracted image.
    _plot_image_with_beam_overlay(
        working_image,
        xc_px,
        yc_px,
        d_major_px,
        d_minor_px,
        phi,
        diameters,
        scale,
        label,
        cmap,
        vmin,
        vmax,
        title=title,
        colorbar=colorbar,
    )

    if d_minor_px is not None:
        ds = d_minor_px * scale
    else:
        ds = None

    return xc_px * scale, yc_px * scale, d_major_px * scale, ds, phi


def plot_image_analysis(
    o_image: np.ndarray,
    title: str = "Original",
    pixel_size: float | None = None,
    units: str = "µm",
    crop: bool | list = False,
    cmap: str = "gist_ncar",
    corner_fraction: float = 0.035,
    nT: float = 3,
    iso_noise: bool = True,
    **kwargs,
) -> None:
    """
    Create a visual report for image fitting.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    All cropping is done after analysis and therefore only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        o_image: 2D image of laser beam
        title: (optional) title for upper left plot
        pixel_size: (optional) size of pixels
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        cmap: (optional) colormap to use
        corner_fraction: (optional) the fractional size of corner rectangles
        nT: (optional) how many standard deviations to subtract
        iso_noise: if True then allow negative pixel values
        **kwargs: extra options to modify display

    Returns:
        nothing
    """
    # Common beam analysis setup
    diameters, xc_px, yc_px, d_major_px, d_minor_px, phi = _prepare_beam_analysis(
        o_image, corner_fraction, nT, iso_noise, **kwargs
    )

    # Setup scale and labels
    scale, label, units_str = _setup_scale_and_labels(pixel_size, units)

    # Crop image if necessary
    image, xc_px, yc_px = _crop_image_if_needed(
        o_image, xc_px, yc_px, d_major_px, d_minor_px, phi, crop, scale, diameters
    )

    # subtract background
    working_image = subtract_iso_background(image, corner_fraction=corner_fraction, nT=nT, iso_noise=iso_noise)
    bkgnd, _ = iso_background(image, corner_fraction=corner_fraction, nT=nT)

    min_ = image.min()
    max_ = image.max()
    vv_px, hh_px = image.shape

    # scale all the dimensions
    v_s = vv_px * scale
    h_s = hh_px * scale
    r_major_s = d_major_px * scale / 2

    plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(right=1.0)

    # add gray to cmap around zero
    ccmap = set_zero_to_lightgray(cmap, min_, max_)

    # original image
    plt.subplot(2, 2, 1)
    im = plt.imshow(image, cmap=ccmap)
    im.cmap.set_bad(color="black")  # color for padded values
    plt.colorbar(im, fraction=0.046 * v_s / h_s, pad=0.04)
    plt.clim(min_, max_)
    plt.xlabel("Position [px]")
    plt.ylabel("Position [px]")
    plt.title(title + ", center at (%.0f, %.0f) px" % (xc_px, yc_px))

    # working image
    plt.subplot(2, 2, 2)

    # diameters in display units for a consistent title
    d_major = d_major_px * scale if d_major_px is not None else None
    d_minor = d_minor_px * scale if d_minor_px is not None else None
    work_title = _format_beam_title(d_major, d_minor, units_str)

    _plot_image_with_beam_overlay(
        working_image,
        xc_px,
        yc_px,
        d_major_px,
        d_minor_px,
        phi,
        diameters,
        scale,
        label,
        cmap,
        vmin=None,
        vmax=None,
        title=work_title,
        colorbar=True,
    )

    extra = 1.03

    # find max and min for both plots so that both plots have equal sizes

    rect_major_px = d_major_px * diameters
    _, _, z_major, s_major_px = major_axis_arrays(image, xc_px, yc_px, rect_major_px, phi)
    a_major = np.sqrt(8 / np.pi) / d_major_px * abs(np.sum(z_major - bkgnd) * (s_major_px[1] - s_major_px[0]))

    a_minor: float = 0
    z_minor = np.array([0])
    r_minor_s: float = 0
    if d_minor_px is not None:
        r_minor_s = d_minor_px * scale / 2
        rect_minor_px = d_minor_px * diameters
        _, _, z_minor, s_minor_px = minor_axis_arrays(image, xc_px, yc_px, rect_minor_px, phi)
        a_minor = np.sqrt(8 / np.pi) / d_minor_px * abs(np.sum(z_minor - bkgnd) * (s_minor_px[1] - s_minor_px[0]))

    baseline = float(a_major) * np.exp(-2 * (diameters / 2) ** 2) + bkgnd
    base_e2 = float(a_major) * np.exp(-2) + bkgnd

    z_min = 0
    z_max = np.max([a_major, np.max(z_major), a_minor, np.max(z_minor)]) * extra + baseline

    plt.subplot(2, 2, 3)
    plt.plot(s_major_px * scale, z_major, "sb", markersize=2)
    plt.plot(s_major_px * scale, z_major, "-b", lw=0.5)
    # gaussian and label
    z_values = bkgnd + a_major * np.exp(-8 * (s_major_px / d_major_px) ** 2)
    plt.plot(s_major_px * scale, z_values, "k")
    plt.text(0, bkgnd + a_major, "  Gaussian Fit")
    # double arrow and label
    plt.annotate("", (-r_major_s, base_e2), (r_major_s, base_e2), arrowprops={"arrowstyle": "<->"})
    if r_major_s < max(s_major_px) * scale / 2:
        plt.text(r_major_s, base_e2, "  $d_{major}$=%.0f %s" % (2 * r_major_s, units_str), va="center", ha="left")
    else:
        plt.text(0, 1.1 * base_e2, "$d_{major}$=%.0f %s" % (2 * r_major_s, units_str), va="bottom", ha="center")
    plt.xlabel("Distance from Center [%s]" % units_str)
    plt.ylabel("Pixel Value")
    plt.title("Major Axis")
    plt.ylim(z_min, z_max)
    plt.xlim(min(s_major_px) * scale, max(s_major_px) * scale)

    if d_minor_px is not None:  # plot of values along minor axis
        plt.subplot(2, 2, 4)
        plt.plot(s_minor_px * scale, z_minor, "sb", markersize=2)
        plt.plot(s_minor_px * scale, z_minor, "-b", lw=0.5)
        z_values = bkgnd + a_minor * np.exp(-8 * (s_minor_px / d_minor_px) ** 2)
        plt.plot(s_minor_px * scale, z_values, "k")
        # double arrow and label
        plt.annotate("", (-r_minor_s, base_e2), (r_minor_s, base_e2), arrowprops={"arrowstyle": "<->"})
        if r_minor_s < max(s_minor_px) * scale / 2:
            plt.text(
                r_minor_s,
                base_e2,
                "  $d_{minor}$=%.0f %s" % (2 * r_minor_s, units_str),
                va="center",
                ha="left",
            )
        else:
            plt.text(0, 1.1 * base_e2, "$d_{minor}$=%.0f %s" % (2 * r_minor_s, units_str), va="bottom", ha="center")

        plt.text(0, bkgnd + a_minor, "  Gaussian Fit")
        plt.xlabel("Distance from Center [%s]" % units_str)
        plt.ylabel("Pixel Value")
        plt.title("Minor Axis")
        plt.ylim(z_min, z_max)
        plt.xlim(min(s_minor_px) * scale, max(s_minor_px) * scale)

    else:
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, "Fit failed.", ha="center", va="center")

    # add more horizontal space between plots
    plt.subplots_adjust(wspace=0.3)


def plot_image_montage(
    images: list[np.ndarray],
    z: npt.NDArray[np.floating] | None = None,
    cols: int = 3,
    pixel_size: float | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    units: str = "µm",
    crop: bool | list = False,
    cmap: str = "gist_ncar",
    corner_fraction: float = 0.035,
    nT: float = 3,
    iso_noise: bool = True,
    **kwargs,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Create a beam size montage for a set of images.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    All cropping is done after analysis and therefore only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        images: array of 2D images of the laser beam
        z: (optional) array of axial positions of images (always in meters!)
        cols: (optional) number of columns in the montage
        pixel_size: (optional) size of pixels
        vmax: (optional) maximum gray level to use
        vmin: (optional) minimum gray level to use
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        cmap: (optional) colormap to use
        corner_fraction: (optional) the fractional size of corner rectangles
        nT: (optional) how many standard deviations to subtract
        iso_noise: (optional) if True then allow negative pixel values
        **kwargs: (optional) extra options to modify display

    Returns:
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
    """
    # arrays to save diameters
    d_major = np.zeros(len(images))
    d_minor = np.zeros(len(images))

    # calculate the number of rows needed in the montage
    rows = (len(images) - 1) // cols + 1

    # when pixel_size is not specified, units default to pixels
    if pixel_size is None:
        units = "px"

    # gather all the options that are fixed for every image in the montage
    options = {
        "pixel_size": pixel_size,
        "vmax": vmax,
        "vmin": vmin,
        "units": units,
        "crop": crop,
        "cmap": cmap,
        "corner_fraction": corner_fraction,
        "nT": nT,
        "iso_noise": iso_noise,
        **kwargs,
    }

    # now set up the grid of subplots
    plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    for i, im in enumerate(images):
        plt.subplot(rows, cols, i + 1)

        # should we add color bar?
        cb = vmax is not None and (i + 1 == cols)

        # plot the image and gather the beam diameters
        _, _, d_major[i], d_minor[i], _ = plot_image_and_fit(im, **options, colorbar=cb)

        # add a title using the shared formatter
        if z is None:
            title = _format_beam_title(d_major[i], d_minor[i], units)
        else:
            title = _format_beam_title(d_major[i], d_minor[i], units, z=z[i])

        plt.title(title)

        # omit y-labels on all but first column
        if i % cols:
            plt.ylabel("")
            if isinstance(crop, list):
                plt.yticks([])

        # omit x-labels on all but last row
        if i < (rows - 1) * cols:
            plt.xlabel("")
            if isinstance(crop, list):
                plt.xticks([])

    for i in range(len(images), rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.axis("off")

    return d_major, d_minor
