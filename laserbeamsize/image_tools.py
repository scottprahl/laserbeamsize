"""
Image manipulation routines needed for beam analysis.

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
import numpy.typing as npt
from numpy import ma
import scipy.ndimage  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import matplotlib.colors
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


def line(r0: int, c0: int, r1: int, c1: int) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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


def rotate_points(
    x: float | npt.NDArray[np.floating],
    y: float | npt.NDArray[np.floating],
    x0: float,
    y0: float,
    phi: float,
) -> tuple[float | npt.NDArray[np.floating], float | npt.NDArray[np.floating]]:
    """
    Rotate x and y around designated center (x0, y0).

    Args:
        x: x-values of point or array of points to be rotated
        y: y-values of point or array of points to be rotated
        x0: horizontal center of rotation
        y0: vertical center of rotation
        phi: angle to rotate (+ is ccw as seen on screen) in radians.
            Because image coordinates have y increasing downward, the
            implementation uses -phi internally so that a positive angle
            produces a counter-clockwise rotation as viewed on screen.

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


def values_along_line(
    image: np.ndarray, x0: float, y0: float, x1: float, y1: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Return x, y, z, and distance values along discrete pixels from (x0, y0) to (x1, y1).

    This version ensures that no duplicate (x, y) values appear due to integer rounding.
    It also works when some of the line is outside the image. Masked pixels are excluded.

    Args:
        image: 2D numpy array (image[y, x]), may be a masked array
        x0: x position of start of line (in pixels)
        y0: y position of start of line (in pixels)
        x1: x position of end of line (in pixels)
        y1: y position of end of line (in pixels)

    Returns:
        x: x-pixel indices along the line (excluding masked pixels)
        y: y-pixel indices along the line (excluding masked pixels)
        z: image values at each (x, y) (excluding masked pixels)
        s: distance from center of line (in pixels)
    """
    height, width = image.shape

    # Full set of pixel indices (row = y, col = x)
    rr_full, cc_full = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))

    # Total distance (from true endpoints)
    total_distance = np.hypot(x1 - x0, y1 - y0)

    # Compute relative distance along the full line
    if total_distance > 0:
        s = np.linspace(0, 1, len(rr_full))
        d_full = (s - 0.5) * total_distance
    else:
        # Handle case where endpoints are the same
        d_full = np.zeros(len(rr_full))

    # Mask to keep only valid pixels (within image bounds)
    valid_bounds = (rr_full >= 0) & (rr_full < height) & (cc_full >= 0) & (cc_full < width)

    rr = rr_full[valid_bounds]
    cc = cc_full[valid_bounds]
    d = d_full[valid_bounds]

    # Extract values
    z = image[rr, cc]

    # If image is a masked array, filter out masked pixels
    if np.ma.is_masked(z):
        valid = ~z.mask
        cc = cc[valid]
        rr = rr[valid]
        d = d[valid]
        z = z.data[valid]

    return cc.astype(float), rr.astype(float), np.asarray(z, dtype=float), d


def _image_arrays(
    image: np.ndarray, xc_px: float, yc_px: float, line_length_px: float, phi: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Return x, y, z, and distance values along the minor axis.

    Args:
        image: the image to work with
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        line_length_px: total length of line
        phi: angle between major axis and horizontal axis [radians]

    Returns:
        x: index of horizontal pixels
        y: index of vertical pixels
        z: image values at each of the x, y positions
        s: position of z values along line
    """
    r_px = line_length_px / 2
    rx_px = r_px * np.cos(phi)
    ry_px = -r_px * np.sin(phi)

    x_start = xc_px - rx_px
    x_end = xc_px + rx_px
    y_start = yc_px - ry_px
    y_end = yc_px + ry_px
    return values_along_line(image, x_start, y_start, x_end, y_end)


def major_axis_arrays(
    image: np.ndarray, xc_px: float, yc_px: float, line_length_px: float, phi: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Return x, y, z, and distance values of an image along the major axis.

    Args:
        image: the image to work
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        line_length_px: total length of line
        phi: angle between major axis and horizontal axis [radians]

    Returns:
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x, y positions
        s: position of z values along line
    """
    return _image_arrays(image, xc_px, yc_px, line_length_px, phi)


def minor_axis_arrays(
    image: np.ndarray, xc_px: float, yc_px: float, line_length_px: float, phi: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Return x, y, z, and distance values along the minor axis.

    Args:
        image: the image to work with
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        line_length_px: total length of line
        phi: angle between major axis and horizontal axis [radians]

    Returns:
        x: index of horizontal pixels
        y: index of vertical pixels
        z: image values at each of the x, y positions
        s: position of z values along line
    """
    return _image_arrays(image, xc_px, yc_px, line_length_px, phi + np.pi / 2)


def rotate_image(original: np.ndarray, x0: float, y0: float, phi: float) -> np.ndarray:
    """
    Create image rotated about specified centerpoint.

    The image is rotated about a centerpoint (x0, y0) and then
    cropped to the original size such that the centerpoint remains
    in the same location.

    Args:
        original: the image to work with
        x0: column
        y0: row
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


def rotated_rect_arrays(
    xc_px: float, yc_px: float, d_major: float, d_minor: float, phi: float
) -> npt.NDArray[np.floating]:
    """
    Return x, y points for rotated rectangle with specified center.

    Args:
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]

    Returns:
        x, y : two arrays for points on corners of rotated rectangle
    """
    rx = d_major / 2
    ry = d_minor / 2

    # rectangle with center at (xc_px, yc_px)
    x = np.array([-rx, -rx, +rx, +rx, -rx]) + xc_px
    y = np.array([-ry, +ry, +ry, -ry, -ry]) + yc_px

    x_rot, y_rot = rotate_points(x, y, xc_px, yc_px, phi)

    return np.array([x_rot, y_rot])


def axes_arrays(xc_px: float, yc_px: float, d_major: float, d_minor: float | None, phi: float) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
    npt.NDArray[np.floating] | None,
]:
    """
    Return x, y arrays needed to draw axes of ellipse.

    Args:
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]

    Returns:
        x, y arrays needed to draw axes of ellipse
    """
    # major ellipse axis with center at (xc_px, yc_px)
    rx = d_major / 2
    x = np.array([-rx, rx]) + xc_px
    y = np.array([0, 0]) + yc_px
    xr1, yr1 = rotate_points(x, y, xc_px, yc_px, phi)
    x_rot1, y_rot1 = np.asarray(xr1), np.asarray(yr1)

    if d_minor is None:
        return x_rot1, y_rot1, None, None

    # minor ellipse axis with center at (xc_px, yc_px)
    ry = d_minor / 2
    x = np.array([0, 0]) + xc_px
    y = np.array([-ry, ry]) + yc_px
    xr2, yr2 = rotate_points(x, y, xc_px, yc_px, phi)
    x_rot2, y_rot2 = np.asarray(xr2), np.asarray(yr2)

    return x_rot1, y_rot1, x_rot2, y_rot2


def ellipse_arrays(
    xc_px: float, yc_px: float, d_major: float, d_minor: float, phi: float, npoints: int = 200
) -> npt.NDArray[np.floating]:
    """
    Return x, y arrays to draw a rotated ellipse.

    Args:
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
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
    xp = xc_px + a * np.cos(phi) - b * np.sin(phi)
    yp = yc_px - a * np.sin(phi) - b * np.cos(phi)
    return np.array([xp, yp])


def create_test_image(
    h: int,
    v: int,
    xc_px: float,
    yc_px: float,
    d_major: float,
    d_minor: float,
    phi: float,
    noise: float = 0,
    ntype: str = "poisson",
    max_value: int = 255,
) -> npt.NDArray[np.floating]:
    """
    Create a 2D test image with an elliptical beam and possible noise.

    Create a v x h image with an elliptical beam with specified center and
    beam dimensions.  By default the values in the image will range from 0 to
    255. The default image will have no background and no noise.

    Args:
        h: number of columns in 2D test image
        v: number of rows in 2D test image
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        noise: (optional) magnitude of pixel noise to add
        ntype: (optional) type of noise — one of "poisson" (default), "gaussian"/"normal",
            "flat"/"uniform", or "constant"
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
    image0 = scale * np.exp(-2 * (x - xc_px) ** 2 / rx**2 - 2 * (y - yc_px) ** 2 / ry**2)

    image1 = rotate_image(image0, xc_px, yc_px, phi)

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


def crop_image_to_rect2(
    image: np.ndarray, xc_px: float, yc_px: float, xmin: float, xmax: float, ymin: float, ymax: float
) -> tuple[np.ndarray, float, float]:
    """
    Return image cropped to specified rectangle.

    Args:
        image: image of beam
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        xmin: left edge (pixels)
        xmax: right edge (pixels)
        ymin: top edge (pixels)
        ymax: bottom edge (pixels)

    Returns:
        cropped_image: cropped image
        new_xc, new_yc: new beam center (pixels)
    """
    v, h = image.shape
    ixmin = max(0, int(xmin))
    ixmax = min(h, int(xmax))
    iymin = max(0, int(ymin))
    iymax = min(v, int(ymax))
    new_xc = xc_px - ixmin
    new_yc = yc_px - iymin
    return image[iymin:iymax, ixmin:ixmax], new_xc, new_yc


def crop_image_to_rect(
    image: np.ndarray, xc_px: float, yc_px: float, xmin: float, xmax: float, ymin: float, ymax: float
) -> tuple[np.ndarray | None, float | None, float | None]:
    """
    Return image cropped to specified rectangle.

    If the crop rectangle exceeds the image bounds, the returned image
    is padded with zeros to match the requested size, and those padded
    regions are masked.

    Args:
        image: image of beam
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        xmin: left edge (pixels)
        xmax: right edge (pixels)
        ymin: top edge (pixels)
        ymax: bottom edge (pixels)

    Returns:
        cropped_image: cropped masked array (None if resulting crop is too small)
        new_xc, new_yc: new beam center (pixels, None if crop is too small)
    """
    v, h = image.shape

    # Convert to integers
    xmin_req = int(xmin)
    xmax_req = int(xmax)
    ymin_req = int(ymin)
    ymax_req = int(ymax)

    # Calculate the requested crop size
    crop_height = ymax_req - ymin_req
    crop_width = xmax_req - xmin_req

    # Check if crop is too small (e.g., less than 3x3 pixels)
    if crop_height < 3 or crop_width < 3:
        return None, None, None

    # Determine valid region within image bounds
    xmin_valid = max(0, xmin_req)
    xmax_valid = min(h, xmax_req)
    ymin_valid = max(0, ymin_req)
    ymax_valid = min(v, ymax_req)

    # Check if there's any overlap with the original image
    if xmin_valid >= xmax_valid or ymin_valid >= ymax_valid:
        return None, None, None

    # Extract the valid portion of the image
    cropped = image[ymin_valid:ymax_valid, xmin_valid:xmax_valid]

    # Create mask array (True = masked/padded, False = valid data)
    mask = np.zeros((crop_height, crop_width), dtype=bool)

    # If crop extends beyond image, pad with zeros and set mask
    if xmin_req < 0 or xmax_req > h or ymin_req < 0 or ymax_req > v:
        # Create zero-padded array
        padded = np.zeros((crop_height, crop_width), dtype=image.dtype)

        # Calculate where to place the cropped image in the padded array
        pad_ymin = max(0, -ymin_req)
        pad_xmin = max(0, -xmin_req)
        pad_ymax = pad_ymin + (ymax_valid - ymin_valid)
        pad_xmax = pad_xmin + (xmax_valid - xmin_valid)

        # Place cropped image into padded array
        padded[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = cropped

        # Set mask to True everywhere except where we have valid data
        mask[:, :] = True
        mask[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = False

        # Create masked array
        cropped = ma.masked_array(padded, mask=mask)

    # Calculate new center coordinates
    new_xc = xc_px - xmin_req
    new_yc = yc_px - ymin_req

    return cropped, new_xc, new_yc


def crop_image_to_integration_rect(
    image: np.ndarray,
    xc_px: float,
    yc_px: float,
    d_major: float,
    d_minor: float | None,
    phi: float,
    mask_diameters: float = 3,
) -> tuple[np.ndarray, float, float]:
    """
    Return image cropped to integration rectangle.

    The integration rectangle is centered on (xc_px, yc_px) and has a length
    mask_diameters * d_major and width=max_diameters * d_minor.  The rectangle
    is rotated by phi radians.

    Since the image is being cropped, the center of the beam will move.

    Args:
        image: image of beam
        xc_px: horizontal center of beam
        yc_px: vertical center of beam
        d_major: major axis (i.e, major diameter)
        d_minor: minor axis (i.e, minor diameter)
        phi: angle between major axis and horizontal axis [radians]
        mask_diameters: multiplier for major/minor diameters

    Returns:
        cropped_image: cropped image
        new_xc: x-position of beam center in cropped image
        new_yc: y-position of beam center in cropped image
    """
    if d_minor is None:
        return image, xc_px, yc_px
    rect_major = mask_diameters * d_major
    rect_minor = mask_diameters * d_minor
    xp, yp = rotated_rect_arrays(xc_px, yc_px, rect_major, rect_minor, phi)
    cropped, new_xc, new_yc = crop_image_to_rect(image, xc_px, yc_px, min(xp), max(xp), min(yp), max(yp))
    if cropped is None or new_xc is None or new_yc is None:
        raise ValueError(
            "crop_image_to_integration_rect: resulting crop is too small "
            "(d_major=%.1f, d_minor=%.1f px, mask_diameters=%d)" % (d_major, d_minor, mask_diameters)
        )
    return cropped, new_xc, new_yc


def create_cmap(vmin: float, vmax: float, band_percentage: float = 4) -> LinearSegmentedColormap:
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


def create_plus_minus_cmap(data: npt.NDArray[np.floating]) -> matplotlib.colors.Colormap:
    """Create a color map with reds for positive and blues for negative values."""
    vmax = np.max(data)
    vmin = np.min(data)

    if 0 <= vmin <= vmax:
        return plt.get_cmap("Reds")
    if vmin <= vmax <= 0:
        return plt.get_cmap("Blues")

    return create_cmap(vmin, vmax)
