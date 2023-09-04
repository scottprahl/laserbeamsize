# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-lines
# pylint: disable=protected-access
# pylint: disable=consider-using-enumerate
# pylint: disable=consider-using-f-string

"""
Routines for creating image masks helpful for beam analysis.

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
from PIL import Image, ImageDraw
from laserbeamsize.image_tools import rotate_image

__all__ = ('corner_mask',
           'perimeter_mask',
           'rotated_rect_mask',
           'elliptical_mask',
           )

def elliptical_mask(image, xc, yc, dx, dy, phi):
    """
    Create a boolean mask for a rotated elliptical disk.

    The returned mask is the same size as `image`.

    Args:
        image: 2D array
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        masked_image: 2D array with True values inside ellipse
    """
    v, h = image.shape
    y, x = np.ogrid[:v, :h]

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    rx = dx / 2
    ry = dy / 2
    xx = x - xc
    yy = y - yc
    r2 = (xx * cosphi - yy * sinphi)**2 / rx**2 + (xx * sinphi + yy * cosphi)**2 / ry**2
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


def rotated_rect_mask_slow(image, xc, yc, dx, dy, phi, mask_diameters=3):
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
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        masked_image: 2D array with True values inside rectangle
    """
    raw_mask = np.full_like(image, 0, dtype=float)
    v, h = image.shape
    rx = mask_diameters * dx / 2
    ry = mask_diameters * dy / 2
    vlo = max(0, int(yc - ry))
    vhi = min(v, int(yc + ry))
    hlo = max(0, int(xc - rx))
    hhi = min(h, int(xc + rx))

    raw_mask[vlo:vhi, hlo:hhi] = 1
    rot_mask = rotate_image(raw_mask, xc, yc, phi) >= 0.5
    return rot_mask


def rotated_rect_mask(image, xc, yc, dx, dy, phi, mask_diameters=3):
    """
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
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
        mask_diameters: number of diameters to include
    Returns:
        masked_image: 2D array with True values inside rectangle
    """
    v, h = image.shape
    rx = mask_diameters * dx / 2
    ry = mask_diameters * dy / 2

    s = np.sin(-phi)
    c = np.cos(-phi)

    xx, xy = rx * c, rx * s
    yx, yy = - ry * s, ry * c

    x1, y1 = xc + xx + yx, yc + xy + yy
    x2, y2 = xc + xx - yx, yc + xy - yy
    x3, y3 = xc - xx - yx, yc - xy - yy
    x4, y4 = xc - xx + yx, yc - xy + yy

    g = Image.new('L', (h, v), 0)
    ImageDraw.Draw(g).polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)], outline=1, fill=1)
    mask = np.array(g)
    return mask.astype(bool)
