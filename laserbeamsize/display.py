# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=too-many-statements

"""
A module for graphing and displaying the beam size fit.

A full graphic can be created by::

    >>>> import imageio
    >>>> import numpy as np
    >>>> import laserbeamsize as lbs
    >>>> beam_image = imageio.v2.imread("t-hene.pgm")
    >>>> x, y, dx, dy, phi = lbs.beam_size(beam_image)
    >>>> print("The center of the beam ellipse is at (%.0f, %.0f)" % (x, y))
    >>>> print("The ellipse diameter (closest to horizontal) is %.0f pixels" % dx)
    >>>> print("The ellipse diameter (closest to   vertical) is %.0f pixels" % dy)
    >>>> print("The ellipse is rotated %.0f° ccw from the horizontal" % (phi * 180/3.1416))
    >>>> lbs.plot_image_analysis(beam_image)
    >>>> plt.show()

A mosaic of images might be created by::

    >>>> # read images for each location
    >>>> z = np.array([89,94,99,104,109,114,119,124,129,134,139], dtype=float) #[mm]
    >>>> filenames = ["%d.pgm" % location for location in z]
    >>>> images = [imageio.v2.imread(filename) for filename in filenames]
    >>>> lbs.plot_image_montage(images, z * 1e-3, pixel_size=3.75, crop=True)
    >>>> plt.show()

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
import matplotlib.pyplot as plt
import laserbeamsize.image_tools as tools
import laserbeamsize.background as back
from laserbeamsize.analysis import beam_size

__all__ = ('beam_ellipticity',
           'plot_beam_diagram',
           'plot_image_analysis',
           'plot_image_and_fit',
           'plot_image_montage'
           )


def beam_ellipticity(dx, dy):
    """
    Calculate the ellipticity of the beam.

    The ISO 11146 standard defines ellipticity as the "ratio between the
    minimum and maximum beam widths".  These widths (diameters) returned
    by `beam_size()` can be used to make this calculation.

    When `ellipticity > 0.87`, then the beam profile may be considered to have
    circular symmetry. The equivalent beam diameter is the root mean square
    of the beam diameters.

    Args:
        dx: x diameter of the beam spot
        dy: y diameter of the beam spot
    Returns:
        ellipticity: varies from 0 (line) to 1 (round)
        d_circular: equivalent diameter of a circular beam
    """
    if dy < dx:
        ellipticity = dy / dx
    elif dx < dy:
        ellipticity = dx / dy
    else:
        ellipticity = 1

    d_circular = np.sqrt((dx**2 + dy**2) / 2)

    return ellipticity, d_circular


def plot_beam_diagram():
    """Draw a simple astigmatic beam ellipse with labels."""
    theta = np.radians(30)
    xc, yc, dx, dy = 0, 0, 50, 25

    plt.subplots(1, 1, figsize=(6, 6))

    # If the aspect ratio is not `equal` then the major and minor radii
    # do not appear to be orthogonal to each other!
    plt.axes().set_aspect('equal')

    xp, yp = tools.ellipse_arrays(xc, yc, dx, dy, theta)
    plt.plot(xp, yp, 'k', lw=2)

    xp, yp = tools.rotated_rect_arrays(xc, yc, dx, dy, theta)
    plt.plot(xp, yp, ':b', lw=2)

    sint = np.sin(theta) / 2
    cost = np.cos(theta) / 2
    plt.plot([xc - dx * cost, xc + dx * cost], [yc + dx * sint, yc - dx * sint], ':b')
    plt.plot([xc + dy * sint, xc - dy * sint], [yc + dy * cost, yc - dy * cost], ':r')

    # draw axes
    plt.annotate("x'", xy=(-25, 0), xytext=(25, 0),
                 arrowprops={'arrowstyle': '<-'}, va='center', fontsize=16)

    plt.annotate("y'", xy=(0, 25), xytext=(0, -25),
                 arrowprops={'arrowstyle': '<-'}, ha='center', fontsize=16)

    plt.annotate(r'$\phi$', xy=(13, -2.5), fontsize=16)
    plt.annotate('', xy=(15.5, 0), xytext=(14, -8.0),
                 arrowprops={'arrowstyle': '<-', 'connectionstyle': 'arc3, rad=-0.2'})

    plt.annotate(r'$d_x$', xy=(-17, 7), color='blue', fontsize=16)
    plt.annotate(r'$d_y$', xy=(-4, -8), color='red', fontsize=16)

    plt.xlim(-30, 30)
    plt.ylim(30, -30)  # inverted to match image coordinates!
    plt.axis('off')


def plot_visible_dotted_line(xpts, ypts):
    """Draw a dotted line that is is visible against images."""
    plt.plot(xpts, ypts, '-', color='#FFD700')
    plt.plot(xpts, ypts, ':', color='#0057B8')


def plot_image_and_fit(o_image,
                       pixel_size=None,
                       vmin=None,
                       vmax=None,
                       units='µm',
                       crop=False,
                       colorbar=False,
                       cmap='gist_ncar',
                       corner_fraction=0.035,
                       nT=3,
                       iso_noise=False,
                       **kwargs):
    """
    Plot the image, fitted ellipse, integration area, and semi-major/minor axes.

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

    All cropping is done after analysis and therefosre only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        o_image: 2D array of image with beam spot
        pixel_size: (optional) size of pixels
        vmin: (optional) minimum value for colorbar
        vmax: (optional) maximum value for colorbar
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        colorbar (optional) show the color bar,
        cmap: (optional) colormap to use

    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    # only pass along arguments that apply to beam_size()
    beamsize_keys = ['mask_diameters', 'max_iter', 'phi']
    bs_args = dict((k, kwargs[k]) for k in beamsize_keys if k in kwargs)
    bs_args['iso_noise'] = iso_noise
    bs_args['nT'] = nT
    bs_args['corner_fraction'] = corner_fraction

    # find center and diameters
    xc, yc, dx, dy, phi = beam_size(o_image, **bs_args)

    # establish scale and correct label
    if pixel_size is None:
        scale = 1
        label = 'Pixels'
    else:
        scale = pixel_size
        label = 'Position (%s)' % units

    # crop image if necessary
    if isinstance(crop, list):
        ymin = yc - crop[0] / 2 / scale  # in pixels
        ymax = yc + crop[0] / 2 / scale
        xmin = xc - crop[1] / 2 / scale
        xmax = xc + crop[1] / 2 / scale
        image, xc, yc = tools.crop_image_to_rect(o_image, xc, yc, xmin, xmax, ymin, ymax)
    elif crop:
        image, xc, yc = tools.crop_image_to_integration_rect(o_image, xc, yc, dx, dy, phi)
    else:
        image = o_image

    # establish maximum colorbar value
    if vmax is None:
        vmax = image.max()
    if vmin is None:
        vmin = image.min()

    # extents may be changed by scale
    v, h = image.shape
    extent = np.array([-xc, h - xc, v - yc, -yc]) * scale

    # display image and axes labels
    im = plt.imshow(image, extent=extent, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.xlabel(label)
    plt.ylabel(label)

    # draw semi-major and semi-minor axes
    xp, yp = tools.axes_arrays(xc, yc, dx, dy, phi)
    plot_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # show ellipse around beam
    xp, yp = tools.ellipse_arrays(xc, yc, dx, dy, phi)
    plot_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # show integration area around beam
    xp, yp = tools.rotated_rect_arrays(xc, yc, dx, dy, phi)
    plot_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # set limits on axes
    plt.xlim(-xc * scale, (h - xc) * scale)
    plt.ylim((v - yc) * scale, -yc * scale)

    # show colorbar
    if colorbar:
        v, h = image.shape
        plt.colorbar(im, fraction=0.046 * v / h, pad=0.04)

    return xc * scale, yc * scale, dx * scale, dy * scale, phi


def plot_image_analysis(o_image,
                        title='Original',
                        pixel_size=None,
                        units='µm',
                        crop=False,
                        cmap='gist_ncar',
                        corner_fraction=0.035,
                        nT=3,
                        iso_noise=False,
                        **kwargs):
    """
    Create a visual report for image fitting.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    All cropping is done after analysis and therefosre only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        o_image: 2D image of laser beam
        title: (optional) title for upper left plot
        pixel_size: (optional) size of pixels
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        cmap: (optional) colormap to use
    Returns:
        nothing
    """
    # only pass along arguments that apply to beam_size()
    bs_args = dict((k, kwargs[k]) for k in ['mask_diameters', 'max_iter', 'phi'] if k in kwargs)
    bs_args['iso_noise'] = iso_noise
    bs_args['nT'] = nT
    bs_args['corner_fraction'] = corner_fraction

    # find center and diameters
    xc, yc, dx, dy, phi = beam_size(o_image, **bs_args)

    # determine scaling and labels
    if pixel_size is None:
        scale = 1
        unit_str = ''
        units = 'pixels'
        label = 'Pixels from Center'
    else:
        scale = pixel_size
        unit_str = '[%s]' % units
        label = 'Distance from Center %s' % unit_str

    # crop image as appropriate
    if isinstance(crop, list):
        ymin = yc - crop[0] / 2 / scale  # in pixels
        ymax = yc + crop[0] / 2 / scale
        xmin = xc - crop[1] / 2 / scale
        xmax = xc + crop[1] / 2 / scale
        image, xc, yc = tools.crop_image_to_rect(o_image, xc, yc, xmin, xmax, ymin, ymax)
    elif crop:
        image, xc, yc = tools.crop_image_to_integration_rect(o_image, xc, yc, dx, dy, phi)
    else:
        image = o_image

    # subtract background
    working_image = back.subtract_iso_background(image, corner_fraction=corner_fraction,
                                                 nT=nT, iso_noise=iso_noise)
    bkgnd, _ = back.iso_background(image, corner_fraction=corner_fraction, nT=nT)

    min_ = image.min()
    max_ = image.max()
    vv, hh = image.shape

    # determine the sizes of the semi-major and semi-minor axes
    r_major = max(dx, dy) / 2.0
    r_minor = min(dx, dy) / 2.0

    # scale all the dimensions
    v_s = vv * scale
    h_s = hh * scale
    xc_s = xc * scale
    yc_s = yc * scale
    r_mag_s = r_major * scale
    d_mag_s = r_mag_s * 2
    r_min_s = r_minor * scale
    d_min_s = r_min_s * 2

    plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(right=1.0)

    # original image
    plt.subplot(2, 2, 1)
    im = plt.imshow(image, cmap=cmap)
    plt.colorbar(im, fraction=0.046 * v_s / h_s, pad=0.04)
    plt.clim(min_, max_)
    plt.xlabel('Position (pixels)')
    plt.ylabel('Position (pixels)')
    plt.title(title)

    # working image
    plt.subplot(2, 2, 2)
    extent = np.array([-xc_s, h_s - xc_s, v_s - yc_s, -yc_s])
    im = plt.imshow(working_image, extent=extent, cmap=cmap)
    xp, yp = tools.ellipse_arrays(xc, yc, dx, dy, phi) * scale
    plot_visible_dotted_line(xp - xc_s, yp - yc_s)

    xp, yp = tools.axes_arrays(xc, yc, dx, dy, phi) * scale
    plot_visible_dotted_line(xp - xc_s, yp - yc_s)

    xp, yp = tools.rotated_rect_arrays(xc, yc, dx, dy, phi) * scale
    plot_visible_dotted_line(xp - xc_s, yp - yc_s)

    plt.colorbar(im, fraction=0.046 * v_s / h_s, pad=0.04)
#    plt.clim(min_, max_)
    plt.xlim(-xc_s, h_s - xc_s)
    plt.ylim(v_s - yc_s, -yc_s)
    plt.xlabel(label)
    plt.ylabel(label)
    plt.title('Image w/o background, center at (%.0f, %.0f) %s' % (xc_s, yc_s, units))

    # plot of values along semi-major axis
    _, _, z, s = tools.major_axis_arrays(image, xc, yc, dx, dy, phi)
    a = np.sqrt(2 / np.pi) / r_major * abs(np.sum(z - bkgnd) * (s[1] - s[0]))
    baseline = a * np.exp(-2) + bkgnd

    plt.subplot(2, 2, 3)
    plt.plot(s * scale, z, 'sb', markersize=2)
    plt.plot(s * scale, z, '-b', lw=0.5)
    z_values = bkgnd + a * np.exp(-2 * (s / r_major)**2)
    plt.plot(s * scale, z_values, 'k')
    plt.annotate('', (-r_mag_s, baseline), (r_mag_s, baseline),
                 arrowprops={'arrowstyle': '<->'})
    plt.text(0, 1.1 * baseline, 'dx=%.0f %s' % (d_mag_s, units), va='bottom', ha='center')
    plt.text(0, bkgnd + a, '  Gaussian Fit')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Major Axis')
    plt.title('Semi-Major Axis')
    # plt.gca().set_ylim(bottom=0)

    # plot of values along semi-minor axis
    _, _, z, s = tools.minor_axis_arrays(image, xc, yc, dx, dy, phi)
    a = np.sqrt(2 / np.pi) / r_minor * abs(np.sum(z - bkgnd) * (s[1] - s[0]))
    baseline = a * np.exp(-2) + bkgnd

    plt.subplot(2, 2, 4)
    plt.plot(s * scale, z, 'sb', markersize=2)
    plt.plot(s * scale, z, '-b', lw=0.5)
    z_values = bkgnd + a * np.exp(-2 * (s / r_minor)**2)
    plt.plot(s * scale, z_values, 'k')
    plt.annotate('', (-r_min_s, baseline), (r_min_s, baseline),
                 arrowprops={'arrowstyle': '<->'})
    plt.text(0, 1.1 * baseline, 'dy=%.0f %s' % (d_min_s, units), va='bottom', ha='center')
    plt.text(0, bkgnd + a, '  Gaussian Fit')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Minor Axis')
    plt.title('Semi-Minor Axis')
    # plt.gca().set_ylim(bottom=0)

    # add more horizontal space between plots
    plt.subplots_adjust(wspace=0.3)


def plot_image_montage(images,
                      z=None,
                      cols=3,
                      pixel_size=None,
                      vmax=None,
                      vmin=None,
                      units='µm',
                      crop=False,
                      cmap='gist_ncar',
                      corner_fraction=0.035,
                      nT=3,
                      iso_noise=False,
                      **kwargs):
    """
    Create a beam size montage for a set of images.

    If `crop` is a two parameter list `[v, h]` then `v` and `h` are
    interpreted as the vertical and horizontal sizes of the rectangle.  The
    size is in pixels unless `pixel_size` is specified.  In that case the
    rectangle sizes are in whatever units `pixel_size` is .

    If `crop==True` then the displayed image is cropped to the ISO 11146 integration
    rectangle.

    All cropping is done after analysis and therefosre only affects
    what is displayed.  If the image needs to be cropped before analysis
    then that must be done before calling this function.

    Args:
        images: array of 2D images of the laser beam
        z: (optional) array of axial positions of images (always in meters!)
        cols: (optional) number of columns in the montage
        pixel_size: (optional) size of pixels
        vmax: (optional) maximum gray level to use
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        cmap: (optional) colormap to use
    Returns:
        dx: semi-major diameter
        dy: semi-minor diameter
    """
    # arrays to save diameters
    dx = np.zeros(len(images))
    dy = np.zeros(len(images))

    # calculate the number of rows needed in the montage
    rows = (len(images) - 1) // cols + 1

    # when pixel_size is not specified, units default to pixels
    if pixel_size is None:
        units = 'pixels'

    # gather all the options that are fixed for every image in the montage
    options = {'pixel_size': pixel_size,
               'vmax': vmax,
               'vmin': vmin,
               'units': units,
               'crop': crop,
               'cmap': cmap,
               'corner_fraction': corner_fraction,
               'nT': nT,
               'iso_noise': iso_noise,
               **kwargs}

    # now set up the grid of subplots
    plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    for i, im in enumerate(images):
        plt.subplot(rows, cols, i + 1)

        # should we add color bar?
        cb = not (vmax is None) and (i + 1 == cols)

        # plot the image and gather the beam diameters
        _, _, dx[i], dy[i], _ = plot_image_and_fit(im, **options, colorbar=cb)

        # add a title
        if units == 'mm':
            s = "dx=%.2f%s, dy=%.2f%s" % (dx[i], units, dy[i], units)
        else:
            s = "dx=%.0f%s, dy=%.0f%s" % (dx[i], units, dy[i], units)
        if z is None:
            plt.title(s)
        else:
            plt.title("z=%.0fmm, %s" % (z[i] * 1e3, s))

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

    return dx, dy
