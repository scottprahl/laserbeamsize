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
"""

import numpy as np
import scipy.ndimage
import laserbeamsize as lbs

__all__ = (
    "corner_background",
    "iso_background",
    "subtract_background_image",
    "subtract_constant",
    "subtract_corner_background",
    "subtract_iso_background",
    "subtract_tilted_background",
)


def subtract_background_image(original, background):
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


def subtract_constant(original, background, iso_noise=True):
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


def corner_background(image, corner_fraction=0.035):
    """
    Return the mean and stdev of background in corners of image.

    The mean and standard deviation are estimated using the pixels from
    the rectangles in the four corners. The default size of these rectangles
    is 0.035 or 3.5% of the full image size.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        corner_mean: average pixel value in corners
    """
    if corner_fraction == 0:
        return 0, 0
    mask = lbs.corner_mask(image, corner_fraction)
    img = np.ma.masked_array(image, ~mask)
    mean = np.mean(img)
    stdev = np.std(img)
    return mean, stdev


def iso_background(image, corner_fraction=0.035, nT=3):
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


def _mean_filter(values):
    return np.mean(values)


def _std_filter(values):
    return np.std(values)


def image_background2(image, fraction=0.035, nT=3):
    """
    Return the background of an image.

    The trick here is identifying unilluminated pixels.  This is done by using
    using convolution to find the local average and standard deviation value for
    each pixel.  The local values are done over an n by m rectangle.

    ISO 11146-3 recommends using (n,m) values that are 2-5% of the image

    un-illuminated (background) pixels are all values that fall below the

    Args:
        image : the image to work with
        fraction: the fractional size of corner rectangles
        nT: how many standard deviations to subtract

    Returns:
        background: average background value across image
    """
    # average over a n x m moving kernel
    n, m = (fraction * np.array(image.shape)).astype(int)
    ave = scipy.ndimage.generic_filter(image, _mean_filter, size=(n, m))
    std = scipy.ndimage.generic_filter(image, _std_filter, size=(n, m))

    # defined ISO/TR 11146-3:2004, equation 61
    threshold = ave + nT * std / np.sqrt((n + 1) * (m + 1))

    # we only average the pixels that fall below the illumination threshold
    unilluminated = image[image < threshold]

    background = int(np.mean(unilluminated))
    return background


def subtract_iso_background(image, corner_fraction=0.035, nT=3, iso_noise=True):
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


def subtract_corner_background(image, corner_fraction=0.035, nT=3, iso_noise=True):
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


def subtract_tilted_background(image, corner_fraction=0.035):
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

    mask = lbs.perimeter_mask(image, corner_fraction=corner_fraction)
    perimeter_values = image[mask]
    # coords is (y_value, x_value, 1) for each point in perimeter_values
    coords = np.stack((yy[mask], xx[mask], np.ones(np.size(perimeter_values))), 1)

    # fit a plane to all corner points
    b = np.array(perimeter_values).T
    A = np.array(coords)
    a, b, c = np.linalg.inv(A.T @ A) @ A.T @ b

    # calculate the fitted background plane
    z = a * yy + b * xx + c

    # find the standard deviation of the noise in the perimeter
    # and subtract this value from the plane
    # since we don't want to lose the image noise just yet
    z -= np.std(perimeter_values)

    # finally, subtract the plane from the original image
    return subtract_background_image(image, z)
