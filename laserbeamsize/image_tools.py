# pylint: disable=import-error
"""
Image manipulation routines needed for beam analysis.

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


__all__ = (
    "rotate_image",
    "axes_arrays",
    "ellipse_arrays",
    "major_axis_arrays",
    "minor_axis_arrays",
    "rotated_rect_arrays",
    "create_test_image",
    "crop_image_to_rect",
    "crop_image_to_integration_rect",
    "create_cmap",
    "create_plus_minus_cmap",
)


def line(r0, c0, r1, c1):
    """
    Returns coordinates of line between two points.

    Bresenham's line algorithm (adapted from scikit-image).

    Generate the coordinates of points between (r0, c0) and (r1, c1).

    Args:
        r0: Row of the starting point.
        c0: Column of the starting point.
        r1: Row of the ending point.
        c1: Column of the ending point.

    Returns:
        rr, cc : Coordinates of pixels in the line.
    """
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    steep = abs(r1 - r0) > abs(c1 - c0)

    if steep:
        r0, c0 = c0, r0
        r1, c1 = c1, r1

    if c0 > c1:
        c0, c1 = c1, c0
        r0, r1 = r1, r0

    dr = abs(r1 - r0)
    dc = c1 - c0
    error = dc // 2
    r = r0

    if r0 < r1:
        step = 1
    else:
        step = -1

    rr = []
    cc = []

    for c in range(c0, c1 + 1):
        if steep:
            rr.append(c)
            cc.append(r)
        else:
            rr.append(r)
            cc.append(c)

        error -= dr
        if error < 0:
            r += step
            error += dc

    return np.array(rr), np.array(cc)


def rotate_points(x, y, x0, y0, phi):
    """
    Rotate x and y around designated center (x0, y0).

    Args:
        x: x-values of point or array of points to be rotated
        y: y-values of point or array of points to be rotated
        x0: horizontal center of rotation
        y0: vertical center of rotation
        phi: angle to rotate (+ is ccw) in radians

    Returns:
        x, y: locations of rotated points
    """
    xp = x - x0
    yp = y - y0

    s = np.sin(-phi)
    c = np.cos(-phi)

    xf = xp * c - yp * s
    yf = xp * s + yp * c

    xf += x0
    yf += y0

    return xf, yf


def values_along_line(image, x0, y0, x1, y1):
    """
    Return x, y, z, and distance values along discrete pixels from (x0, y0) to (x1, y1).

    This version ensures that no duplicate (x, y) values appear due to integer rounding.
    It also works when some of the line is outside the image

    Args:
        image: 2D numpy array (image[y, x])
        x0: x position of start of line (in pixels)
        y0: y position of start of line (in pixels)
        x1: x position of end of line (in pixels)
        y1: y position of end of line (in pixels)

    Returns:
        x: x-pixel indices along the line
        y: y-pixel indices along the line
        z: image values at each (x, y)
        s: distance from center of line (in pixels)
    """
    height, width = image.shape

    # Full set of pixel indices (row = y, col = x)
    rr_full, cc_full = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))

    # Total distance (from true endpoints)
    total_distance = np.hypot(x1 - x0, y1 - y0)

    # Compute relative distance along the full line
    s = np.linspace(0, 1, len(rr_full))
    d_full = (s - 0.5) * total_distance

    # Now mask to keep only valid pixels
    mask = (rr_full >= 0) & (rr_full < height) & (cc_full >= 0) & (cc_full < width)
    rr = rr_full[mask]
    cc = cc_full[mask]
    d = d_full[mask]

    z = image[rr, cc]

    return cc.astype(float), rr.astype(float), z.astype(float), d


def major_axis_arrays(image, xc, yc, d_major, _d_minor, phi, diameters=2):
    """
    Return x, y, z, and distance values along major axis.

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: major axis (i.e, major diameter)
        _d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        diameters: number of diameters to use
    Returns:
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x, y positions
        s: distance from start of minor axis to x, y position
    """
    r = diameters * d_major / 2
    rx = r * np.cos(phi)
    ry = -r * np.sin(phi)
    return values_along_line(image, xc - rx, yc - ry, xc + rx, yc + ry)


def minor_axis_arrays(image, xc, yc, d_major, _d_minor, phi, diameters=2):
    """
    Return x, y, z, and distance values along minor axis.

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: major axis (i.e, major diameter)
        _d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        diameters: number of diameters to use
    Returns:
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x, y positions
        s: distance from start of minor axis to x, y position
    """
    # use major diameter so plots look better
    r = diameters * d_major / 2
    # minor axis is rotated 90° from major axis
    rx = r * np.cos(phi + np.pi / 2)
    ry = -r * np.sin(phi + np.pi / 2)

    return values_along_line(image, xc - rx, yc - ry, xc + rx, yc + ry)


def rotate_image(original, x0, y0, phi):
    """
    Create image rotated about specified centerpoint.

    The image is rotated about a centerpoint (x0, y0) and then
    cropped to the original size such that the centerpoint remains
    in the same location.

    Args:
        original: the image to work with
        x0:     column
        y0:     row
        phi: angle [radians]

    Returns:
        image: rotated 2D array with same dimensions as original
    """
    if phi is None:
        return original

    # center of original image
    cy, cx = (np.array(original.shape) - 1) / 2.0

    # rotate image using defaults mode='constant' and cval=0.0
    rotated = scipy.ndimage.rotate(original, np.degrees(phi), order=1)

    # center of rotated image, defaults mode='constant' and cval=0.0
    ry, rx = (np.array(rotated.shape) - 1) / 2.0

    # position of (x0, y0) in rotated image
    new_x0, new_y0 = rotate_points(x0, y0, cx, cy, phi)
    new_x0 += rx - cx
    new_y0 += ry - cy

    voff = int(new_y0 - y0)
    hoff = int(new_x0 - x0)

    # crop so center remains in same location as original
    ov, oh = original.shape
    rv, rh = rotated.shape

    rv1 = max(voff, 0)
    sv1 = max(-voff, 0)
    vlen = min(voff + ov, rv) - rv1

    rh1 = max(hoff, 0)
    sh1 = max(-hoff, 0)
    hlen = min(hoff + oh, rh) - rh1

    # move values into zero-padded array
    s = np.full_like(original, 0)
    sv1_end = sv1 + vlen
    sh1_end = sh1 + hlen
    rv1_end = rv1 + vlen
    rh1_end = rh1 + hlen
    s[sv1:sv1_end, sh1:sh1_end] = rotated[rv1:rv1_end, rh1:rh1_end]
    return s


def rotated_rect_arrays(xc, yc, d_major, d_minor, phi, mask_diameters=3):
    """
    Return x, y arrays to draw a rotated rectangle.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        mask_diameters: hide pixels outside this number of diameters

    Returns:
        x, y : two arrays for points on corners of rotated rectangle
    """
    rx = mask_diameters * d_major / 2
    ry = mask_diameters * d_minor / 2

    # rectangle with center at (xc, yc)
    x = np.array([-rx, -rx, +rx, +rx, -rx]) + xc
    y = np.array([-ry, +ry, +ry, -ry, -ry]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return np.array([x_rot, y_rot])


def axes_arrays(xc, yc, d_major, d_minor, phi, mask_diameters=3):
    """
    Return x, y arrays needed to draw axes of ellipse.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        mask_diameters: hide pixels outside this number of diameters

    Returns:
        x, y arrays needed to draw axes of ellipse
    """
    rx = mask_diameters * d_major / 2
    ry = mask_diameters * d_minor / 2

    # major and minor ellipse axes with center at (xc, yc)
    x = np.array([-rx, rx, 0, 0, 0]) + xc
    y = np.array([0, 0, 0, -ry, ry]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return np.array([x_rot, y_rot])


def ellipse_arrays(xc, yc, d_major, d_minor, phi, npoints=200):
    """
    Return x, y arrays to draw a rotated ellipse.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        npoints: (optional) number of points to use for ellipse

    Returns:
        x, y : two arrays of points on the ellipse
    """
    t = np.linspace(0, 2 * np.pi, npoints)
    a = d_major / 2 * np.cos(t)
    b = d_minor / 2 * np.sin(t)
    xp = xc + a * np.cos(phi) - b * np.sin(phi)
    yp = yc - a * np.sin(phi) - b * np.cos(phi)
    return np.array([xp, yp])


def create_test_image(h, v, xc, yc, d_major, d_minor, phi, noise=0, ntype="poisson", max_value=255):
    """
    Create a 2D test image with an elliptical beam and possible noise.

    Create a v x h image with an elliptical beam with specified center and
    beam dimensions.  By default the values in the image will range from 0 to
    255. The default image will have no background and no noise.

    Args:
        h: number of columns in 2D test image
        v: number of rows in 2D test image
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        noise: (optional) magnitued of normally distributed pixel noise to add
        ntype: (optional) type of noise to use
        max_value: (optional) all values in image fall between 0 and `max_value`

    Returns:
        image: an unsigned 2D integer array of a Gaussian elliptical spot
    """
    if max_value < 0 or max_value >= 2**16:
        raise ValueError("max_value must be positive and less than 65535")

    if not isinstance(h, int) or h <= 0:
        raise ValueError("number of columns must be positive")

    if not isinstance(v, int) or v <= 0:
        raise ValueError("number of rows must be positive")

    if phi is not None and abs(phi) > 2.1 * np.pi:
        raise ValueError("the angle phi should be in radians!")

    rx = d_major / 2
    ry = d_minor / 2

    image0 = np.zeros([v, h])

    y, x = np.ogrid[:v, :h]

    scale = max_value - 3 * noise
    image0 = scale * np.exp(-2 * (x - xc) ** 2 / rx**2 - 2 * (y - yc) ** 2 / ry**2)

    image1 = rotate_image(image0, xc, yc, phi)

    if noise > 0:
        if ntype == "poisson":
            # noise is the mean value of the distribution
            image1 += np.random.poisson(noise, size=(v, h))

        if ntype == "constant":
            # noise is the mean value of the distribution
            image1 += noise

        if ntype in ("gaussian", "normal"):
            # noise is the mean value of the distribution
            image1 += np.random.normal(noise, np.sqrt(noise), size=(v, h))

        if ntype in ("flat", "uniform"):
            # noise is the mean value of the distribution
            image1 += np.random.uniform(0, noise, size=(v, h))

        # after adding noise, the signal may exceed the range 0 to max_value
        np.place(image1, image1 > max_value, max_value)
        np.place(image1, image1 < 0, 0)

    if max_value < 2**8:
        return image1.astype(np.uint8)
    if max_value < 2**16:
        return image1.astype(np.uint16)
    return image1


def crop_image_to_rect(image, xc, yc, xmin, xmax, ymin, ymax):
    """
    Return image cropped to specified rectangle.

    Args:
        image: image of beam
        xc: horizontal center of beam
        yc: vertical center of beam
        xmin: left edge (pixels)
        xmax: right edge (pixels)
        ymin: top edge (pixels)
        ymax: bottom edge (pixels)

    Returns:
        cropped_image: cropped image
        new_xc, new_yc: new beam center (pixels)
    """
    v, h = image.shape
    xmin = max(0, int(xmin))
    xmax = min(h, int(xmax))
    ymin = max(0, int(ymin))
    ymax = min(v, int(ymax))
    new_xc = xc - xmin
    new_yc = yc - ymin
    return image[ymin:ymax, xmin:xmax], new_xc, new_yc


def crop_image_to_integration_rect(image, xc, yc, d_major, d_minor, phi):
    """
    Return image cropped to integration rectangle.

    Since the image is being cropped, the center of the beam will move.

    Args:
        image: image of beam
        xc: horizontal center of beam
        yc: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]

    Returns:
        cropped_image: cropped image
        new_xc: x-position of beam center in cropped image
        new_yc: y-position of beam center in cropped image
    """
    xp, yp = rotated_rect_arrays(xc, yc, d_major, d_minor, phi, mask_diameters=3)
    return crop_image_to_rect(image, xc, yc, min(xp), max(xp), min(yp), max(yp))


def create_cmap(vmin, vmax, band_percentage=4):
    """
    Create a colormap with a specific range, mapping vmin to 0 and vmax to 1.

    The colormap is interesting because negative values are blue and positive values
    are red.  Zero is shown as a white band: blue, dark blue, white, dark red, and red.
    The transition points between the colors are determined by the normalized range.

    Args:
        vmin (float): The minimum value of the range to be normalized.
        vmax (float): The maximum value of the range to be normalized.
        band_percentage (option): fraction of the entire band that is white

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The generated colormap with 255 colors.
    """
    r = vmin / (vmin - vmax)
    delta = band_percentage / 100
    colors = [(0, 0, 0.6), (0, 0, 1), (1, 1, 1), (1, 0, 0), (0.6, 0, 0)]
    positions = [0, (1 - delta) * r, r, (1 + delta) * r, 1]
    return LinearSegmentedColormap.from_list("plus_minus", list(zip(positions, colors)), N=255)


def create_plus_minus_cmap(data):
    """Create a color map with reds for positive and blues for negative values."""
    vmax = np.max(data)
    vmin = np.min(data)

    if 0 <= vmin <= vmax:
        return plt.get_cmap("Reds")
    if vmin <= vmax <= 0:
        return plt.get_cmap("Blues")

    return create_cmap(vmin, vmax)
