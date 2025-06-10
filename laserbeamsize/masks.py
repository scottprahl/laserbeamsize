"""
Routines for creating image masks helpful for beam analysis.

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
import laserbeamsize as lbs

__all__ = (
    "corner_mask",
    "perimeter_mask",
    "rotated_rect_mask",
    "elliptical_mask",
    "iso_background_mask",
)


def elliptical_mask(image, xc, yc, d_major, d_minor, phi):
    """
    Create a boolean mask for a rotated elliptical disk.

    The returned mask is the same size as `image`.

    Args:
        image: 2D array
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: semi-major ellipse diameter
        d_minor: semi-minor ellipse diameter
        phi: angle between horizontal and major axes [radians]

    Returns:
        masked_image: 2D array with True values inside ellipse
    """
    v, h = image.shape
    y, x = np.ogrid[:v, :h]

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    rx = d_major / 2
    ry = d_minor / 2
    xx = x - xc
    yy = y - yc
    r2 = (xx * cosphi - yy * sinphi) ** 2 / rx**2 + (xx * sinphi + yy * cosphi) ** 2 / ry**2
    the_mask = r2 <= 1

    return the_mask


def corner_mask(image, corner_fraction=0.035):
    """
    Create boolean mask for image with corners marked as True.

    Each of the four corners is a fixed percentage of the entire image.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`
    the default is 0.035=3.5% of the iamge.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        masked_image: 2D array with True values in four corners
    """
    v, h = image.shape
    n = int(v * corner_fraction)
    m = int(h * corner_fraction)

    the_mask = np.full_like(image, False, dtype=bool)
    the_mask[:n, :m] = True
    the_mask[:n, -m:] = True
    the_mask[-n:, :m] = True
    the_mask[-n:, -m:] = True
    return the_mask


def perimeter_mask(image, corner_fraction=0.035):
    """
    Create boolean mask for image with a perimeter marked as True.

    The perimeter is the same width as the corners created by corner_mask
    which is a fixed percentage (default 3.5%) of the entire image.

    Args:
        image : the image to work with
        corner_fraction: determines the width of the perimeter
    Returns:
        masked_image: 2D array with True values around rect perimeter
    """
    v, h = image.shape
    n = int(v * corner_fraction)
    m = int(h * corner_fraction)

    the_mask = np.full_like(image, False, dtype=bool)
    the_mask[:, :m] = True
    the_mask[:, -m:] = True
    the_mask[:n, :] = True
    the_mask[-n:, :] = True
    return the_mask


def rotated_rect_mask_slow(image, xc, yc, d_major, d_minor, phi, mask_diameters=3):
    """
    Create ISO 11146 rectangular mask for specified beam.

    ISO 11146-2 §7.2 states that integration should be carried out over
    "a rectangular integration area which is centred to the beam centroid,
    defined by the spatial first order moments, orientated parallel to
    the principal axes of the power density distribution, and sized
    three times the beam widths".

    This routine creates a mask with `true` values for each pixel in
    the image that should be part of the integration.

    The rectangular mask is `mask_diameters' times the pixel diameters
    of the ellipse.

    The rectangular mask is rotated about (xc, yc) so that it is aligned
    with the elliptical spot.

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: semi-major ellipse diameter
        d_minor: semi-minor ellipse diameter
        phi: angle between horizontal and major axes [radians]
        mask_diameters: number of diameters to include

    Returns:
        masked_image: 2D array with True values inside rectangle
    """
    raw_mask = np.full_like(image, 0, dtype=float)
    v, h = image.shape
    rx = mask_diameters * d_major / 2
    ry = mask_diameters * d_minor / 2
    vlo = max(0, int(yc - ry))
    vhi = min(v, int(yc + ry))
    hlo = max(0, int(xc - rx))
    hhi = min(h, int(xc + rx))

    raw_mask[vlo:vhi, hlo:hhi] = 1
    rot_mask = lbs.rotate_image(raw_mask, xc, yc, phi) >= 0.5
    return rot_mask


def rotated_rect_mask(image, xc, yc, d_major, d_minor, phi, mask_diameters=3):
    """
    Create a boolean mask of a rotated rectangle within an image using NumPy.

    Create ISO 11146 rectangular mask for specified beam.

    ISO 11146-2 §7.2 states that integration should be carried out over
    "a rectangular integration area which is centred to the beam centroid,
    defined by the spatial first order moments, orientated parallel to
    the principal axes of the power density distribution, and sized
    three times the beam widths".

    This routine creates a mask with `true` values for each pixel in
    the image that should be part of the integration.

    The rectangular mask is `mask_diameters` times the pixel diameters
    of the ellipse.

    The rectangular mask is rotated about (xc, yc) and then drawn using PIL

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: semi-major ellipse diameter
        d_minor: semi-minor ellipse diameter
        phi: angle between horizontal and major axes [radians]
        mask_diameters: number of diameters to include
    Returns:
        masked_image: 2D array with True values inside rectangle
    """
    height, width = image.shape
    rx = mask_diameters * d_major / 2
    ry = mask_diameters * d_minor / 2

    # create a meshgrid of pixel coordinates
    y, x = np.ogrid[:height, :width]
    x = x - xc
    y = y - yc

    # rotate coordinates by -phi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x_rot = x * cos_phi + y * sin_phi
    y_rot = -x * sin_phi + y * cos_phi

    # define mask for points inside the rotated rectangle
    mask = (np.abs(x_rot) <= rx) & (np.abs(y_rot) <= ry)
    return mask


def iso_background_mask(image, corner_fraction=0.035, nT=3):
    """
    Return a mask indicating the background pixels in an image.

    We estimate the mean and standard deviation using the values in the
    corners.  All pixel values that fall below the mean+nT*stdev are considered
    unilluminated (background) pixels.

    Args:
        image : the image to work with
        nT: how many standard deviations to subtract
        corner_fraction: the fractional size of corner rectangles
    Returns:
        background_mask: 2D array of True/False values
    """
    # estimate background
    ave, std = lbs.corner_background(image, corner_fraction=corner_fraction)

    # defined ISO/TR 11146-3:2004, equation 59
    threshold = ave + nT * std

    background_mask = image < threshold

    return background_mask
