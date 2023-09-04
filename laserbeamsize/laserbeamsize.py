# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-lines
# pylint: disable=protected-access
# pylint: disable=consider-using-enumerate
# pylint: disable=consider-using-f-string

"""
A module for finding the beam size in an monochrome image.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

Simple and fast calculation of beam sizes from a single monochrome image based
on the ISO 11146 method of variances.  Some effort has been made to make
the algorithm less sensitive to background offset and noise.

Finding the center and diameters of a beam in a monochrome image is simple::

    >>>> import imageio
    >>>> import numpy as np
    >>>> import laserbeamsize as lbs
    >>>> beam_image = imageio.imread("t-hene.pgm")
    >>>> x, y, dx, dy, phi = lbs.beam_size(beam_image)
    >>>> print("The center of the beam ellipse is at (%.0f, %.0f)" % (x, y))
    >>>> print("The ellipse diameter (closest to horizontal) is %.0f pixels" % dx)
    >>>> print("The ellipse diameter (closest to   vertical) is %.0f pixels" % dy)
    >>>> print("The ellipse is rotated %.0f° ccw from the horizontal" % (phi * 180/3.1416))

A full graphic can be created by::

    >>>> lbs.beam_size_plot(beam_image)
    >>>> plt.show()

A mosaic of images might be created by::

    >>>> # read images for each location
    >>>> z = np.array([89,94,99,104,109,114,119,124,129,134,139], dtype=float) #[mm]
    >>>> filenames = ["%d.pgm" % location for location in z]
    >>>> images = [imageio.imread(filename) for filename in filenames]
    >>>> lbs.beam_size_montage(images, z * 1e-3, pixel_size=3.75, crop=True)
    >>>> plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import laserbeamsize.image_tools as tools
import laserbeamsize.background as back
import laserbeamsize.masks as masks

__all__ = ('basic_beam_size',
           'basic_beam_size_naive',
           'beam_size',
           'beam_ellipticity',
           'draw_beam_figure',
           'beam_size_plot',
           'beam_size_and_plot',
           'beam_size_montage'
           )


def basic_beam_size(original):
    """
    Determine the beam center, diameters, and tilt using ISO 11146 standard.

    Find the center and sizes of an elliptical spot in an 2D array.

    The function does nothing to eliminate background noise.  It just finds the first
    and second order moments and returns the beam parameters. Consequently
    a beam spot in an image with a constant background will fail badly.

    FWIW, this implementation is roughly 800X faster than one that finds
    the moments using for loops.

    Args:
        image: 2D array of image with beam spot
    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    image = original.astype(float)
    v, h = image.shape

    # total of all pixels
    p = np.sum(image, dtype=float)     # float avoids integer overflow

    # sometimes the image is all zeros, just return
    if p == 0:
        return int(h / 2), int(v / 2), 0, 0, 0

    # find the centroid
    hh = np.arange(h, dtype=float)      # float avoids integer overflow
    vv = np.arange(v, dtype=float)      # ditto
    xc = np.sum(np.dot(image, hh)) / p
    yc = np.sum(np.dot(image.T, vv)) / p

    # find the variances
    hs = hh - xc
    vs = vv - yc
    xx = np.sum(np.dot(image, hs**2)) / p
    xy = np.dot(np.dot(image.T, vs), hs) / p
    yy = np.sum(np.dot(image.T, vs**2)) / p

    # Ensure that the case xx==yy is handled correctly
    if xx == yy:
        disc = np.abs(2 * xy)
        phi = np.sign(xy) * np.pi / 4
    else:
        diff = xx - yy
        disc = np.sign(diff) * np.sqrt(diff**2 + 4 * xy**2)
        phi = 0.5 * np.arctan(2 * xy / diff)

    # finally, the major and minor diameters
    dx = np.sqrt(8 * (xx + yy + disc))
    dy = np.sqrt(8 * (xx + yy - disc))

    # phi is negative because image is inverted
    phi *= -1

    return xc, yc, dx, dy, phi


def beam_size(image,
              mask_diameters=3,
              corner_fraction=0.035,
              nT=3,
              max_iter=25,
              phi=None,
              iso_noise=False):
    """
    Determine beam parameters in an image with noise.

    The function first estimates the elliptical spot by excluding all points
    that are less than the average value found in the corners of the image.

    These beam parameters are then used to determine a rectangle that surrounds
    the elliptical spot.  The rectangle size is `mask_diameters` times the spot
    diameters.  This is the integration region used for estimate a new beam
    spot.

    This process is repeated until two successive spot sizes match again as
    outlined in ISO 11146

    `corner_fraction` determines the size of the corners. ISO 11146-3
    recommends values from 2-5%.  The default value of 3.5% works pretty well.

    `mask_diameters` is the size of the rectangular mask in diameters
    of the ellipse.  ISO 11146 states that `mask_diameters` should be 3.
    This default value works fine.

    `nT` accounts for noise in the background.  The background is estimated
    using the values in the corners of the image as `mean+nT * stdev`. ISO 11146
    states that `2<nT<4`.  The default value works fine.

    `max_iter` is the maximum number of iterations done before giving up.

    Args:
        image: 2D array of image of beam
        mask_diameters: the size of the integration rectangle in diameters
        corner_fraction: the fractional size of the corners
        nT: the multiple of background noise to remove
        max_iter: maximum number of iterations.
        phi: (optional) fixed tilt of ellipse in radians
    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    if len(image.shape) > 2:
        raise Exception('Color images are not supported.  Convert to gray/monochrome.')

    print('corner ', back.corner_background(image))
    print('image ', back.image_background(image))
    # remove background
    image_without_background = back.subtract_image_background(
        image, corner_fraction, nT, iso_noise=iso_noise)

    # initial guess at beam properties
    print("finding beam with iso_noise=",iso_noise)
    if iso_noise:
        all_kwargs = {'mask_diameters': mask_diameters,
                      'corner_fraction': corner_fraction,
                      'nT': nT,
                      'max_iter': max_iter,
                      'phi': phi,
                      'iso_noise':False}
        xc, yc, dx, dy, phi_ = beam_size(image, **all_kwargs)
    else:
        xc, yc, dx, dy, phi_ = basic_beam_size(image_without_background)

    for _iteration in range(1, max_iter):

        phi_ = phi or phi_

        # save current beam properties for later comparison
        xc2, yc2, dx2, dy2 = xc, yc, dx, dy

        # create a mask so only values within the mask are used
        mask = masks.rotated_rect_mask(image, xc, yc, dx, dy, phi_, mask_diameters)
        masked_image = np.copy(image_without_background)

        # zero values outside mask (rotation allows mask pixels to differ from 0 or 1)
        masked_image[mask < 0.5] = 0
        plt.imshow(masked_image)
        plt.show()

        xc, yc, dx, dy, phi_ = basic_beam_size(masked_image)
        print('iteration %d' % _iteration)
        print("    old  new")
        print("x  %4d %4d" %(xc2, xc))
        print("y  %4d %4d" %(yc2, yc))
        print("dx %4d %4d" %(dx2, dx))
        print("dy %4d %4d" %(dy2, dy))
        print("min", np.min(masked_image))

        if abs(xc - xc2) < 1 and abs(yc - yc2) < 1 and abs(dx - dx2) < 1 and abs(dy - dy2) < 1:
            break

    phi_ = phi or phi_

    return xc, yc, dx, dy, phi_


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


def basic_beam_size_naive(image):
    """
    Slow but simple implementation of ISO 11146 beam standard.

    This is identical to `basic_beam_size()` and is the obvious way to
    program the calculation of the necessary moments.  It is slow.

    Args:
        image: 2D array of image with beam spot in it
    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    v, h = image.shape

    # locate the center just like ndimage.center_of_mass(image)
    p = 0.0
    xc = 0.0
    yc = 0.0
    for i in range(v):
        for j in range(h):
            p += image[i, j]
            xc += image[i, j] * j
            yc += image[i, j] * i
    xc = int(xc / p)
    yc = int(yc / p)

    # calculate variances
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for i in range(v):
        for j in range(h):
            xx += image[i, j] * (j - xc)**2
            xy += image[i, j] * (j - xc) * (i - yc)
            yy += image[i, j] * (i - yc)**2
    xx /= p
    xy /= p
    yy /= p

    # compute major and minor axes as well as rotation angle
    dx = 2 * np.sqrt(2) * np.sqrt(xx + yy + np.sign(xx - yy) * np.sqrt((xx - yy)**2 + 4 * xy**2))
    dy = 2 * np.sqrt(2) * np.sqrt(xx + yy - np.sign(xx - yy) * np.sqrt((xx - yy)**2 + 4 * xy**2))
    phi = 2 * np.arctan2(2 * xy, xx - yy)

    return xc, yc, dx, dy, phi


def draw_beam_figure():
    """Draw a simple astigmatic beam ellipse with labels."""
    theta = np.radians(30)
    xc = 0
    yc = 0
    dx = 50
    dy = 25

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


def draw_visible_dotted_line(xpts, ypts):
    """Draw a dotted line that is is visible against images."""
    plt.plot(xpts, ypts, '-', color='#FFD700')
    plt.plot(xpts, ypts, ':', color='#0057B8')


def beam_size_and_plot(o_image,
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

    This is helpful for creating a mosaics of all the images created for an
    experiment.

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
    draw_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # show ellipse around beam
    xp, yp = tools.ellipse_arrays(xc, yc, dx, dy, phi)
    draw_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # show integration area around beam
    xp, yp = tools.rotated_rect_arrays(xc, yc, dx, dy, phi)
    draw_visible_dotted_line((xp - xc) * scale, (yp - yc) * scale)

    # set limits on axes
    plt.xlim(-xc * scale, (h - xc) * scale)
    plt.ylim((v - yc) * scale, -yc * scale)

    # show colorbar
    if colorbar:
        v, h = image.shape
        plt.colorbar(im, fraction=0.046 * v / h, pad=0.04)

    return xc * scale, yc * scale, dx * scale, dy * scale, phi


def beam_size_plot(o_image,
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
        title: optional title for upper left plot
        pixel_size: (optional) size of pixels
        units: (optional) string used for units used on axes
        crop: (optional) crop image to integration rectangle
        cmap: (optional) colormap to use
    Returns:
        nothing
    """
    # only pass along arguments that apply to beam_size()
    beamsize_keys = ['mask_diameters', 'max_iter', 'phi', 'iso_noise']
    bs_args = dict((k, kwargs[k]) for k in beamsize_keys if k in kwargs)
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
    working_image = back.subtract_image_background(image, corner_fraction=corner_fraction,
                                              nT=nT, iso_noise=iso_noise)
    bkgnd, _ = back.image_background(image, corner_fraction=corner_fraction, nT=nT)

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
    draw_visible_dotted_line(xp - xc_s, yp - yc_s)

    xp, yp = tools.axes_arrays(xc, yc, dx, dy, phi) * scale
    draw_visible_dotted_line(xp - xc_s, yp - yc_s)

    xp, yp = tools.rotated_rect_arrays(xc, yc, dx, dy, phi) * scale
    draw_visible_dotted_line(xp - xc_s, yp - yc_s)

    plt.colorbar(im, fraction=0.046 * v_s / h_s, pad=0.04)
#    plt.clim(min_, max_)
    plt.xlim(-xc_s, h_s - xc_s)
    plt.ylim(v_s - yc_s, -yc_s)
    plt.xlabel(label)
    plt.ylabel(label)
    plt.title('Image w/o background, center at (%.0f, %.0f) %s' % (xc_s, yc_s, units))

    # plot of values along semi-major axis
    _, _, z, s = tools.major_axis_arrays(o_image, xc, yc, dx, dy, phi)
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
    plt.text(0, a, '  Gaussian Fit')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Major Axis')
    plt.title('Semi-Major Axis')
    #plt.gca().set_ylim(bottom=0)

    # plot of values along semi-minor axis
    _, _, z, s = tools.minor_axis_arrays(o_image, xc, yc, dx, dy, phi)
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
    plt.text(0, a, '  Gaussian Fit')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Minor Axis')
    plt.title('Semi-Minor Axis')
    #plt.gca().set_ylim(bottom=0)

    # add more horizontal space between plots
    plt.subplots_adjust(wspace=0.3)


def beam_size_montage(images,
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
        _, _, dx[i], dy[i], _ = beam_size_and_plot(im, **options, colorbar=cb)

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
