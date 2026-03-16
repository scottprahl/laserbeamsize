"""
Routines for removing background for beam analysis.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

Two functions are used to find the mean and standard deviation of images.
`corner_background()` uses just the corner pixels and `iso_background()` uses
all un-illuminated pixels::

    >>> import imageio.v3 as iio
    >>> import laserbeamsize as lbs
    >>>
    >>> file = "https://github.com/scottprahl/laserbeamsize/raw/main/docs/t-hene.pgm"
    >>> image = iio.imread(file)
    >>>
    >>> mean, stdev = lbs.corner_background(image)
    >>> print("The corner pixels have an average         %.1f ± %.1f)" % (mean, stdev))
    >>> mean, stdev = lbs.iso_background(image)
    >>> print("The un-illuminated pixels have an average %.1f ± %.1f)" % (mean, stdev))

In addition to these functions, there are a variety of subtraction functions to
remove the background.  The most useful is `subtract_iso_background()` which will
return an image with the average of the un-illuminated pixels subtracted::

    >>> import imageio.v3 as iio
    >>> import laserbeamsize as lbs
    >>>
    >>> file = "https://github.com/scottprahl/laserbeamsize/raw/main/docs/t-hene.pgm"
    >>> image = iio.imread(file)
    >>>
    >>> clean_image = subtract_iso_background(image)

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
import numpy.typing as npt

__all__ = (
    "corner_mask",
    "perimeter_mask",
    "rotated_rect_mask",
    "elliptical_mask",
    "iso_background_mask",
    "corner_background",
    "iso_background",
    "subtract_background_image",
    "subtract_constant",
    "subtract_corner_background",
    "subtract_iso_background",
    "subtract_tilted_background",
)


def _get_unmasked_bounding_box(image: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    """
    Get the bounding box of the unmasked region in a masked image.

    Args:
        image: potentially masked image

    Returns:
        tuple: (row_min, row_max, col_min, col_max, v_inner, h_inner)
               If image is not masked, returns full image bounds.
               If no unmasked pixels, returns None.
    """
    v, h = image.shape

    if np.ma.is_masked(image):
        # Find rows and columns that contain unmasked pixels
        img_mask = np.ma.getmaskarray(image)
        unmasked_rows = np.where(~img_mask.all(axis=1))[0]
        unmasked_cols = np.where(~img_mask.all(axis=0))[0]

        if len(unmasked_rows) == 0 or len(unmasked_cols) == 0:
            # No unmasked pixels
            return None

        # Bounding box of unmasked region
        row_min = unmasked_rows[0]
        row_max = unmasked_rows[-1] + 1
        col_min = unmasked_cols[0]
        col_max = unmasked_cols[-1] + 1

        # Size of the unmasked region
        v_inner = row_max - row_min
        h_inner = col_max - col_min
    else:
        # Use the full image
        row_min, row_max = 0, v
        col_min, col_max = 0, h
        v_inner = v
        h_inner = h

    return row_min, row_max, col_min, col_max, v_inner, h_inner


def _apply_image_mask(mask: npt.NDArray[np.bool_], image: np.ndarray) -> npt.NDArray[np.bool_]:
    """
    Apply existing image mask to exclude already-masked pixels.

    Args:
        mask: boolean mask to modify
        image: potentially masked image

    Returns:
        modified mask with image.mask applied if present
    """
    if np.ma.is_masked(image):
        return mask & ~np.ma.getmaskarray(image)
    return mask


def elliptical_mask(
    image: np.ndarray, xc: float, yc: float, d_major: float, d_minor: float, phi: float
) -> npt.NDArray[np.bool_]:
    """
    Create a boolean mask for a rotated elliptical disk.

    The returned mask is the same size as `image`.

    If the image is already masked, only unmasked pixels within the
    ellipse will be marked as True.

    Args:
        image: 2D array (may be a masked array)
        xc: horizontal center of beam (in full image coordinates)
        yc: vertical center of beam (in full image coordinates)
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
    if rx == 0 or ry == 0:
        raise ValueError("d_major and d_minor must be non-zero for elliptical_mask()")
    xx = x - xc
    yy = y - yc
    r2 = (xx * cosphi - yy * sinphi) ** 2 / rx**2 + (xx * sinphi + yy * cosphi) ** 2 / ry**2
    the_mask = r2 <= 1

    return _apply_image_mask(the_mask, image)


def corner_mask(image: np.ndarray, corner_fraction: float = 0.035) -> npt.NDArray[np.bool_]:
    """
    Create boolean mask for image with corners marked as True.

    Each of the four corners is a fixed percentage of the entire image.
    ISO 11146-3 recommends values from 2-5% for `corner_fraction`
    the default is 0.035=3.5% of the image.

    If the image is already masked, the corners are taken from the
    bounding box of the unmasked region (where image.mask=False).

    Args:
        image: the image to work with (may be a masked array)
        corner_fraction: the fractional size of corner rectangles

    Returns:
        masked_image: 2D array with True values in four corners
    """
    v, h = image.shape

    # Get bounding box of unmasked region
    bbox = _get_unmasked_bounding_box(image)
    if bbox is None:
        # No unmasked pixels, return all False
        return np.full((v, h), False, dtype=bool)

    row_min, row_max, col_min, col_max, v_inner, h_inner = bbox

    # Calculate corner sizes based on inner region
    n = int(v_inner * corner_fraction)
    m = int(h_inner * corner_fraction)

    # Create the mask
    the_mask = np.full((v, h), False, dtype=bool)

    # Mark the four corners of the inner region
    the_mask[row_min : row_min + n, col_min : col_min + m] = True  # top-left
    the_mask[row_min : row_min + n, col_max - m : col_max] = True  # top-right
    the_mask[row_max - n : row_max, col_min : col_min + m] = True  # bottom-left
    the_mask[row_max - n : row_max, col_max - m : col_max] = True  # bottom-right

    return _apply_image_mask(the_mask, image)


def perimeter_mask(image: np.ndarray, corner_fraction: float = 0.035) -> npt.NDArray[np.bool_]:
    """
    Create boolean mask for image with a perimeter marked as True.

    The perimeter is the same width as the corners created by corner_mask
    which is a fixed percentage (default 3.5%) of the entire image.

    If the image is already masked, the perimeter is taken from the
    bounding box of the unmasked region (where image.mask=False).

    Args:
        image: the image to work with (may be a masked array)
        corner_fraction: determines the width of the perimeter
    Returns:
        masked_image: 2D array with True values around rect perimeter
    """
    v, h = image.shape

    # Get bounding box of unmasked region
    bbox = _get_unmasked_bounding_box(image)
    if bbox is None:
        # No unmasked pixels, return all False
        return np.full((v, h), False, dtype=bool)

    row_min, row_max, col_min, col_max, v_inner, h_inner = bbox

    # Calculate perimeter width based on inner region
    n = int(v_inner * corner_fraction)
    m = int(h_inner * corner_fraction)

    the_mask = np.full((v, h), False, dtype=bool)

    # Mark the perimeter of the inner region
    the_mask[row_min:row_max, col_min : col_min + m] = True  # left edge
    the_mask[row_min:row_max, col_max - m : col_max] = True  # right edge
    the_mask[row_min : row_min + n, col_min:col_max] = True  # top edge
    the_mask[row_max - n : row_max, col_min:col_max] = True  # bottom edge

    return _apply_image_mask(the_mask, image)


def rotated_rect_mask(
    image: np.ndarray, xc: float, yc: float, d_major: float, d_minor: float, phi: float
) -> npt.NDArray[np.bool_]:
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

    The rectangular mask is rotated about (xc, yc) using NumPy.

    If the image is already masked, only unmasked pixels within the
    rectangle will be marked as True.

    Args:
        image: the image to work with (may be a masked array)
        xc: horizontal center of beam (in full image coordinates)
        yc: vertical center of beam (in full image coordinates)
        d_major: semi-major ellipse diameter
        d_minor: semi-minor ellipse diameter
        phi: angle between horizontal and major axes [radians]

    Returns:
        masked_image: 2D array with True values inside rectangle
    """
    height, width = image.shape
    rx = d_major / 2
    ry = d_minor / 2

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

    return _apply_image_mask(mask, image)


def iso_background_mask(image: np.ndarray, corner_fraction: float = 0.035, nT: float = 3) -> npt.NDArray[np.bool_]:
    """
    Return a mask indicating the background pixels in an image.

    We estimate the mean and standard deviation using the values in the
    corners.  All pixel values that fall below the mean+nT*stdev are considered
    unilluminated (background) pixels.

    If the image is already masked, only unmasked pixels will be considered
    for background determination.

    Args:
        image: the image to work with (may be a masked array)
        nT: how many standard deviations to subtract
        corner_fraction: the fractional size of corner rectangles
    Returns:
        background_mask: 2D array of True/False values
    """
    # estimate background
    ave, std = corner_background(image, corner_fraction=corner_fraction)

    # defined ISO/TR 11146-3:2004, equation 59
    threshold = ave + nT * std

    background_mask = image < threshold

    return _apply_image_mask(background_mask, image)


def subtract_background_image(original: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Subtract a background image from the image with beam.

    The function operates on 2D arrays representing grayscale images. Since the
    subtraction can result in negative pixel values, it is important that the
    return array be an array of float (instead of unsigned arrays that will wrap
    around.

    Args:
        original (numpy.ndarray): 2D array image with beam present.
        background (numpy.ndarray): 2D array image without beam.

    Returns:
        numpy.ndarray: 2D array with background subtracted

    Examples:
        >>> import numpy as np
        >>> original = np.array([[1, 2], [3, 4]])
        >>> background = np.array([[2, 1], [1, 1]])
        >>> subtract_background_image(original, background)
        array([[-1, 1],
               [2, 3]])
    """
    # Checking if the inputs are numpy arrays
    if not isinstance(original, np.ndarray) or not isinstance(background, np.ndarray):
        raise TypeError('Inputs "original" and "background" must be numpy arrays.')

    # Checking if the inputs are two-dimensional arrays
    if original.ndim != 2 or background.ndim != 2:
        raise ValueError('Inputs "original" and "background" must be two-dimensional arrays.')

    # Checking if the shapes of the inputs are equal
    if original.shape != background.shape:
        raise ValueError('Inputs "original" and "background" must have equal shapes.')

    # convert to signed version and subtract
    o = original.astype(float)
    b = background.astype(float)
    subtracted = o - b

    return subtracted


def subtract_constant(original: np.ndarray, background: float, iso_noise: bool = True) -> np.ndarray:
    """
    Return image with a constant value subtracted.

    Subtract threshold from entire image.  If iso_noise is False
    then negative values are set to zero.

    The returned array is an array of float with the shape of original.

    Args:
        original : the image to work with
        background: value to subtract every pixel
        iso_noise: if True then allow negative pixel values
    Returns:
        image: 2D float array with constant background subtracted
    """
    subtracted = original.astype(float)

    if not iso_noise:
        np.place(subtracted, subtracted < background, background)

    subtracted -= background
    return subtracted


def corner_background(image: np.ndarray, corner_fraction: float = 0.035) -> tuple[float, float]:
    """
    Return the mean and stdev of background in corners of image.

    The mean and standard deviation are estimated using the pixels from
    the rectangles in the four corners. The default size of these rectangles
    is 0.035 or 3.5% of the full image size.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    If corner_fraction=0, returns (0, 0) immediately without examining the image.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        corner_mean: average pixel value in corners
        corner_stdev: standard deviation of pixel values in corners
    """
    if corner_fraction == 0:
        return 0, 0
    mask = corner_mask(image, corner_fraction)
    img = np.ma.masked_array(image, ~mask)
    mean = np.mean(img)
    stdev = np.std(img)
    return mean, stdev


def iso_background(image: np.ndarray, corner_fraction: float = 0.035, nT: float = 3) -> tuple[float, float]:
    """
    Return the background for unilluminated pixels in an image.

    This follows one method described in ISO 11146-3 to determine the background
    in an image.

    We first estimate the mean and standard deviation using the values in the
    corners.  All pixel values that fall below the mean+nT*stdev are considered
    un-illuminated (background) pixels.  These are averaged to find the background
    value for the image.

    Args:
        image : the image to work with
        nT: how many standard deviations to subtract
        corner_fraction: the fractional size of corner rectangles
    Returns:
        mean, stdev: mean and stdev of background in the image
    """
    if corner_fraction <= 0 or corner_fraction > 0.25:
        raise ValueError("corner_fraction must be positive and less than 0.25.")

    # estimate background
    ave, std = corner_background(image, corner_fraction=corner_fraction)

    # defined ISO/TR 11146-3:2004, equation 59
    threshold = ave + nT * std

    # collect all pixels that fall below the threshold
    unilluminated = image[image <= threshold]

    if len(unilluminated) == 0:
        raise ValueError("est bkgnd=%.2f stdev=%.2f. No values in image are <= %.2f." % (ave, std, threshold))

    mean = np.mean(unilluminated)
    stdev = np.std(unilluminated)
    return mean, stdev


def subtract_iso_background(
    image: np.ndarray, corner_fraction: float = 0.035, nT: float = 3, iso_noise: bool = True
) -> np.ndarray:
    """
    Return image with ISO 11146 background subtracted.

    The mean and standard deviation are estimated using the pixels from
    the rectangles in the four corners. The default size of these rectangles
    is 0.035 or 3.5% of the full image size.

    The new image will have a constant with the corner mean subtracted.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    ISO 11146-3 recommends from 2-4 for `nT`.

    If iso_noise is False, then after subtracting the mean of the corners,
    pixels values < nT * stdev will be set to zero.

    If iso_noise is True, then no zeroing background is done.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
        nT: how many standard deviations to subtract
        iso_noise: if True then allow negative pixel values

    Returns:
        image: 2D array with background subtracted
    """
    back, sigma = iso_background(image, corner_fraction=corner_fraction, nT=nT)

    subtracted = image.astype(float)
    subtracted -= back

    if not iso_noise:  # zero pixels that fall within a few stdev
        threshold = nT * sigma
        np.place(subtracted, subtracted < threshold, 0)

    return subtracted


def subtract_corner_background(
    image: np.ndarray, corner_fraction: float = 0.035, nT: float = 3, iso_noise: bool = True
) -> np.ndarray:
    """
    Return image with background subtracted.

    The mean and standard deviation are estimated using the pixels from
    the rectangles in the four corners. The default size of these rectangles
    is 0.035 or 3.5% of the full image size.

    The new image will have a constant with the corner mean subtracted.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    ISO 11146-3 recommends from 2-4 for `nT`.

    If iso_noise is False, then after subtracting the mean of the corners,
    pixels values < nT * stdev will be set to zero.

    If iso_noise is True, then no zeroing background is done.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
        nT: how many standard deviations to subtract
        iso_noise: if True then allow negative pixel values

    Returns:
        image: 2D array with background subtracted
    """
    back, sigma = corner_background(image, corner_fraction)

    subtracted = image.astype(float)
    subtracted -= back

    if not iso_noise:  # zero pixels that fall within a few stdev
        threshold = nT * sigma
        np.place(subtracted, subtracted < threshold, 0)

    return subtracted


def subtract_tilted_background(image: np.ndarray, corner_fraction: float = 0.035) -> np.ndarray:
    """
    Return image with tilted planar background subtracted.

    Take all the points around the perimeter of an image and fit these
    to a tilted plane to determine the background to subtract.  Details of
    the linear algebra are at https://math.stackexchange.com/questions/99299

    Since the sample contains noise, it is important not to remove
    this noise at this stage and therefore we offset the plane so
    that one standard deviation of noise remains.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        image: 2D array with tilted planar background subtracted
    """
    v, h = image.shape
    xx, yy = np.meshgrid(range(h), range(v))

    mask = perimeter_mask(image, corner_fraction=corner_fraction)
    perimeter_values = image[mask]
    # coords is (y_value, x_value, 1) for each point in perimeter_values
    coords = np.stack((yy[mask], xx[mask], np.ones(np.size(perimeter_values))), 1)

    # fit a plane to all corner points
    rhs = np.array(perimeter_values).T
    A = np.array(coords)
    a, b, c = np.linalg.inv(A.T @ A) @ A.T @ rhs

    # calculate the fitted background plane
    z = a * yy + b * xx + c

    # find the standard deviation of the noise in the perimeter
    # and subtract this value from the plane
    # since we don't want to lose the image noise just yet
    z -= np.std(perimeter_values)

    # finally, subtract the plane from the original image
    return subtract_background_image(image, z)
