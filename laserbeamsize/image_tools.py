# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Image manipulation routines needed for beam analysis.

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = ('rotate_image',
           'axes_arrays',
           'ellipse_arrays',
           'major_axis_arrays',
           'minor_axis_arrays',
           'rotated_rect_arrays',
           'create_test_image',
           'crop_image_to_rect',
           'crop_image_to_integration_rect',
           'create_cmap',
           'create_plus_minus_cmap',
           )


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


def values_along_line(image, x0, y0, x1, y1, N=100):
    """
    Return x, y, z, and distance values between (x0, y0) and (x1, y1).

    Args:
        image: the image to work with
        x0: x-value of start of line
        y0: y-value of start of line
        x1: x-value of end of line
        y1: y-value of end of line
        N:  number of points in returned array
    Returns:
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x, y positions
        s: distance from start of minor axis to x, y position
    """
    d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    s = np.linspace(0, 1, N)

    x = x0 + s * (x1 - x0)
    y = y0 + s * (y1 - y0)

    xx = x.astype(int)
    yy = y.astype(int)

    zz = image[yy, xx]

    return xx, yy, zz, (s - 0.5) * d


def major_axis_arrays(image, xc, yc, dx, dy, phi, diameters=3):
    """
    Return x, y, z, and distance values along semi-major axis.

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
        diameters: number of diameters to use
    Returns:
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x, y positions
        s: distance from start of minor axis to x, y position
    """
    v, h = image.shape

    if dx > dy:
        rx = diameters * dx / 2
        left = max(xc - rx, 0)
        right = min(xc + rx, h - 1)
        x = np.array([left, right])
        y = np.array([yc, yc])
        xr, yr = rotate_points(x, y, xc, yc, phi)
    else:
        ry = diameters * dy / 2
        top = max(yc - ry, 0)
        bottom = min(yc + ry, v - 1)
        x = np.array([xc, xc])
        y = np.array([top, bottom])
        xr, yr = rotate_points(x, y, xc, yc, phi)

    return values_along_line(image, xr[0], yr[0], xr[1], yr[1])


def minor_axis_arrays(image, xc, yc, dx, dy, phi, diameters=3):
    """
    Return x, y, z, and distance values along semi-minor axis.

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
        diameters: number of diameters to use
    Returns:
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x, y positions
        s: distance from start of minor axis to x, y position
    """
    v, h = image.shape

    if dx <= dy:
        rx = diameters * dx / 2
        left = max(xc - rx, 0)
        right = min(xc + rx, h - 1)
        x = np.array([left, right])
        y = np.array([yc, yc])
        xr, yr = rotate_points(x, y, xc, yc, phi)
    else:
        ry = diameters * dy / 2
        top = max(yc - ry, 0)
        bottom = min(yc + ry, v - 1)
        x = np.array([xc, xc])
        y = np.array([top, bottom])
        xr, yr = rotate_points(x, y, xc, yc, phi)

    return values_along_line(image, xr[0], yr[0], xr[1], yr[1])


def rotate_image(original, x0, y0, phi):
    """
    Create image rotated about specified centerpoint.

    The image is rotated about a centerpoint (x0, y0) and then
    cropped to the original size such that the centerpoint remains
    in the same location.

    Args:
        image: the image to work with
        x:     column
        y:     row
        phi: angle [radians]
    Returns:
        image: rotated 2D array with same dimensions as original
    """
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


def rotated_rect_arrays(xc, yc, dx, dy, phi, mask_diameters=3):
    """
    Return x, y arrays to draw a rotated rectangle.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        x, y : two arrays for points on corners of rotated rectangle
    """
    rx = mask_diameters * dx / 2
    ry = mask_diameters * dy / 2

    # rectangle with center at (xc, yc)
    x = np.array([-rx, -rx, +rx, +rx, -rx]) + xc
    y = np.array([-ry, +ry, +ry, -ry, -ry]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return np.array([x_rot, y_rot])


def axes_arrays(xc, yc, dx, dy, phi, mask_diameters=3):
    """
    Return x, y arrays needed to draw semi-axes of ellipse.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        x, y arrays needed to draw semi-axes of ellipse
    """
    rx = mask_diameters * dx / 2
    ry = mask_diameters * dy / 2

    # major and minor ellipse axes with center at (xc, yc)
    x = np.array([-rx, rx, 0, 0, 0]) + xc
    y = np.array([0, 0, 0, -ry, ry]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return np.array([x_rot, y_rot])


def ellipse_arrays(xc, yc, dx, dy, phi, npoints=200):
    """
    Return x, y arrays to draw a rotated ellipse.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        x, y : two arrays of points on the ellipse
    """
    t = np.linspace(0, 2 * np.pi, npoints)
    a = dx / 2 * np.cos(t)
    b = dy / 2 * np.sin(t)
    xp = xc + a * np.cos(phi) - b * np.sin(phi)
    yp = yc - a * np.sin(phi) - b * np.cos(phi)
    return np.array([xp, yp])


def create_test_image(h, v, xc, yc, dx, dy, phi, noise=0, ntype='poisson', max_value=255):
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
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated ccw [radians]
        noise: normally distributed pixel noise to add to image
        max_value: all values in image fall between 0 and `max_value`
    Returns:
        image: an unsigned 2D integer array of a Gaussian elliptical spot
    """
    if max_value < 0 or max_value >= 2**16:
        raise ValueError('max_value must be positive and less than 65535')

    if not isinstance(h, int) or h <= 0:
        raise ValueError('number of columns must be positive')

    if not isinstance(v, int) or v <= 0:
        raise ValueError('number of rows must be positive')

    if abs(phi) > 2.1 * np.pi:
        raise ValueError('the angle phi should be in radians!')

    rx = dx / 2
    ry = dy / 2

    image0 = np.zeros([v, h])

    y, x = np.ogrid[:v, :h]

    scale = max_value - 3 * noise
    image0 = scale * np.exp(-2 * (x - xc)**2 / rx**2 - 2 * (y - yc)**2 / ry**2)

    image1 = rotate_image(image0, xc, yc, phi)

    if noise > 0:
        if ntype == 'poisson':
            # noise is the mean value of the distribution
            image1 += np.random.poisson(noise, size=(v, h))

        if ntype == 'constant':
            # noise is the mean value of the distribution
            image1 += noise

        if ntype in ('gaussian', 'normal'):
            # noise is the mean value of the distribution
            image1 += np.random.normal(noise, np.sqrt(noise), size=(v, h))

        if ntype in ('flat', 'uniform'):
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


def crop_image_to_integration_rect(image, xc, yc, dx, dy, phi):
    """
    Return image cropped to integration rectangle.

    Since the image is being cropped, the center of the beam will move.

    Args:
        image: image of beam
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        cropped_image: cropped image
        new_xc: x-position of beam center in cropped image
        new_yc: y-position of beam center in cropped image
    """
    xp, yp = rotated_rect_arrays(xc, yc, dx, dy, phi, mask_diameters=3)
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
    delta = band_percentage/100
    colors = [(0, 0, 0.6), (0, 0, 1), (1, 1, 1), (1, 0, 0), (0.6, 0, 0)]
    positions = [0, (1-delta)*r, r, (1+delta)*r, 1]
    return LinearSegmentedColormap.from_list("plus_minus", list(zip(positions, colors)), N=255)


def create_plus_minus_cmap(data):
    """Create a color map with reds for positive and blues for negative values."""
    vmax = np.max(data)
    vmin = np.min(data)
    if 0<=vmin<=vmax :
        return plt.get_cmap('Reds')
    if vmin<=vmax <= 0 :
        return plt.get_cmap('Blues')

    return create_cmap(vmin, vmax)
