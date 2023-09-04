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
"""

import numpy as np
import matplotlib.pyplot as plt
import laserbeamsize.background as back
from laserbeamsize.masks import rotated_rect_mask

__all__ = ('basic_beam_size',
           'beam_size',
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
        raise ValueError('Color images not supported. Image must be 2D.')

    print('corner ', back.corner_background(image))
    print('image ', back.image_background(image))
    # remove background
    image_without_background = back.subtract_image_background(
        image, corner_fraction, nT, iso_noise=iso_noise)

    # initial guess at beam properties
    print("finding beam with iso_noise=", iso_noise)
    if iso_noise:
        all_kwargs = {'mask_diameters': mask_diameters,
                      'corner_fraction': corner_fraction,
                      'nT': nT,
                      'max_iter': max_iter,
                      'phi': phi,
                      'iso_noise': False}
        xc, yc, dx, dy, phi_ = beam_size(image, **all_kwargs)
    else:
        xc, yc, dx, dy, phi_ = basic_beam_size(image_without_background)

    for _iteration in range(1, max_iter):

        phi_ = phi or phi_

        # save current beam properties for later comparison
        xc2, yc2, dx2, dy2 = xc, yc, dx, dy

        # create a mask so only values within the mask are used
        mask = rotated_rect_mask(image, xc, yc, dx, dy, phi_, mask_diameters)
        masked_image = np.copy(image_without_background)

        # zero values outside mask (rotation allows mask pixels to differ from 0 or 1)
        masked_image[mask < 0.5] = 0
        plt.imshow(masked_image)
        plt.show()

        xc, yc, dx, dy, phi_ = basic_beam_size(masked_image)
        print('iteration %d' % _iteration)
        print("    old  new")
        print("x  %4d %4d" % (xc2, xc))
        print("y  %4d %4d" % (yc2, yc))
        print("dx %4d %4d" % (dx2, dx))
        print("dy %4d %4d" % (dy2, dy))
        print("min", np.min(masked_image))

        if abs(xc - xc2) < 1 and abs(yc - yc2) < 1 and abs(dx - dx2) < 1 and abs(dy - dy2) < 1:
            break

    phi_ = phi or phi_

    return xc, yc, dx, dy, phi_


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
