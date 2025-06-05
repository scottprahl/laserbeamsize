"""
A module for finding the beam size in an monochrome image.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

Simple and fast calculation of beam sizes from a single monochrome image based
on the ISO 11146 method of variances.  Some effort has been made to make
the algorithm automatically handle background noise.

Finding the center and diameters of a beam in a monochrome image is simple::

    >>> import imageio.v3 as iio
    >>> import laserbeamsize as lbs
    >>>
    >>> file = "https://github.com/scottprahl/laserbeamsize/raw/main/docs/t-hene.pgm"
    >>> image = iio.imread(file)
    >>>
    >>> x, y, dx, dy, phi = lbs.beam_size(image)
    >>> print("The center of the beam ellipse is at (%.0f, %.0f)" % (x, y))
    >>> print("The ellipse diameter (closest to horizontal) is %.0f pixels" % dx)
    >>> print("The ellipse diameter (closest to   vertical) is %.0f pixels" % dy)
    >>> print("The ellipse is rotated %.0f° ccw from the horizontal" % (phi * 180/3.1416))
"""

import numpy as np
import laserbeamsize as lbs

__all__ = (
    "basic_beam_size",
    "beam_size",
)


def basic_beam_size(original, phi=None):
    """
    Determine the beam center, diameters, and tilt using ISO 11146 standard.

    Find the center and sizes of an elliptical spot in an 2D array.

    The function does nothing to eliminate background noise.  It just finds the first
    and second order moments and returns the beam parameters. Consequently
    a beam spot in an image with a constant background will fail badly.

    FWIW, this implementation is roughly 800X faster than one that finds
    the moments using for loops.

    When background noise dominates then a diameter of 1 is returned.

    Args:
        original: 2D array of image with beam spot
        phi: (optional) fixed rotation angle [radians]
            If ``None`` the angle is determined from the image.

    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    image = original.astype(float)
    v, h = image.shape

    if phi is not None:
        x0 = (h - 1) / 2
        y0 = (v - 1) / 2
        image = lbs.rotate_image(image, x0, y0, -phi)

    # total of all pixels
    p = np.sum(image, dtype=float)  # float avoids integer overflow

    # sometimes the image is all zeros, just return
    if p == 0:
        return int(h / 2), int(v / 2), 0, 0, 0

    # find the centroid
    hh = np.arange(h, dtype=float)  # float avoids integer overflow
    vv = np.arange(v, dtype=float)  # ditto
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
        phi_ = np.sign(xy) * np.pi / 4
    else:
        diff = xx - yy
        disc = np.sign(diff) * np.sqrt(diff**2 + 4 * xy**2)
        phi_ = 0.5 * np.arctan(2 * xy / diff)

    dx = 1
    dy = 1
    if xx + yy + disc > 0:  # fails when negative noise dominates
        dx = np.sqrt(8 * (xx + yy + disc))

    if xx + yy - disc > 0:
        dy = np.sqrt(8 * (xx + yy - disc))

    # phi is negative because image is inverted
    phi_ *= -1

    if phi is not None:
        xc, yc = lbs.image_tools.rotate_points(xc, yc, x0, y0, phi)
        phi_ = phi

    return xc, yc, dx, dy, phi_


def _validate_inputs(
    image, mask_diameters=3, corner_fraction=0.035, nT=3, max_iter=25, phi=None
):
    """
    Ensure arguments to validate inputs are sane.

    This is separate to keep the beam_size() a reasonable size.
    """
    if len(image.shape) > 2:
        raise ValueError("Color images not supported. Image must be 2D.")

    if mask_diameters <= 0 or mask_diameters > 5:
        raise ValueError("mask_diameters must be a positive number less than 5.")

    if corner_fraction < 0 or corner_fraction > 0.25:
        raise ValueError("corner_fraction must be a positive number less than 0.25.")

    if nT < 2 or nT > 4:
        raise ValueError("nT must be between 2 and 4.")

    if max_iter < 0 or not isinstance(max_iter, int):
        raise ValueError("max_iter must be a non-negative integer.")

    if phi is not None and abs(phi) > 2.1 * np.pi:
        raise ValueError("the angle phi should be in radians!")


def beam_size(
    image,
    mask_diameters=3,
    corner_fraction=0.035,
    nT=3,
    max_iter=25,
    phi=None,
    iso_noise=True,
):
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
        mask_diameters: (optional) the size of the integration rectangle in diameters
        corner_fraction: (optional) the fractional size of the corners
        nT: (optional) the multiple of background noise to remove
        max_iter: (optional) maximum number of iterations.
        phi: (optional) fixed tilt of ellipse in radians
        iso_noise: if True then allow negative pixel values

    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated ccw [radians]
    """
    _validate_inputs(image, mask_diameters, corner_fraction, nT, max_iter, phi)

    # zero background for initial guess at beam size
    image_no_bkgnd = lbs.subtract_iso_background(
        image, corner_fraction=corner_fraction, nT=nT, iso_noise=False
    )
    xc, yc, dx, dy, phi_ = basic_beam_size(image_no_bkgnd, phi=phi)

    if iso_noise:  # follow iso background guidelines (positive & negative bkgnd values)
        image_no_bkgnd = lbs.subtract_iso_background(
            image, corner_fraction=corner_fraction, nT=nT, iso_noise=True
        )

    for _iteration in range(1, max_iter):

        if phi is not None:
            phi_ = phi

        # save current beam properties for later comparison
        xc2, yc2, dx2, dy2 = xc, yc, dx, dy

        # create a mask so only values within the mask are used
        mask = lbs.rotated_rect_mask(image, xc, yc, dx, dy, phi_, mask_diameters)
        masked_image = np.copy(image_no_bkgnd)

        # zero values outside mask (rotation allows mask pixels to differ from 0 or 1)
        masked_image[mask < 0.5] = 0

        # find the new parameters
        xc, yc, dx, dy, phi_ = basic_beam_size(masked_image, phi=phi)

        if (
            abs(xc - xc2) < 1
            and abs(yc - yc2) < 1
            and abs(dx - dx2) < 1
            and abs(dy - dy2) < 1
        ):
            break

    if phi is not None:
        phi_ = phi

    return xc, yc, dx, dy, phi_


def basic_beam_size_naive(image, phi=None):
    """
    Slow but simple implementation of ISO 11146 beam standard.

    This is identical to `basic_beam_size()` and is the obvious way to
    program the calculation of the necessary moments.  It is slow.

    Args:
        image: 2D array of image with beam spot in it
        phi: (optional) fixed rotation angle [radians]
            If ``None`` the angle is determined from the image.

    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    v, h = image.shape

    if phi is not None:
        x0 = (h - 1) / 2
        y0 = (v - 1) / 2
        image = lbs.rotate_image(image, x0, y0, -phi)

    # locate the center just like ndimage.center_of_mass(image)
    p = 0.0
    xc = 0.0
    yc = 0.0
    for i in range(v):
        for j in range(h):
            p += image[i, j]
            xc += image[i, j] * j
            yc += image[i, j] * i
    xc /= p
    yc /= p

    # calculate variances
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for i in range(v):
        for j in range(h):
            xx += image[i, j] * (j - xc) ** 2
            xy += image[i, j] * (j - xc) * (i - yc)
            yy += image[i, j] * (i - yc) ** 2
    xx /= p
    xy /= p
    yy /= p

    dx = (
        2
        * np.sqrt(2)
        * np.sqrt(xx + yy + np.sign(xx - yy) * np.sqrt((xx - yy) ** 2 + 4 * xy**2))
    )
    dy = (
        2
        * np.sqrt(2)
        * np.sqrt(xx + yy - np.sign(xx - yy) * np.sqrt((xx - yy) ** 2 + 4 * xy**2))
    )
    phi_ = 2 * np.arctan2(2 * xy, xx - yy)

    if phi is not None:
        xc, yc = lbs.image_tools.rotate_points(xc, yc, x0, y0, phi)
        phi_ = phi

    return xc, yc, dx, dy, phi_
