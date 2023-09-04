# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-lines
# pylint: disable=protected-access
# pylint: disable=consider-using-enumerate
# pylint: disable=consider-using-f-string

"""
Routines for removing background for beam analysis.

Full documentation is available at <https://laserbeamsize.readthedocs.io>
"""

import numpy as np
import scipy.ndimage
from laserbeamsize.masks import corner_mask, perimeter_mask

__all__ = ('subtract_background_image',
           'subtract_constant',
           'subtract_tilted_background',
           'corner_background',
           'image_background',
           'subtract_image_background',
           'subtract_corner_background',
           )


def subtract_background_image(original,
                   background,
                   iso_noise=False):
    """
    Subtract a background image from the original image.

    iso_noise=True allows pixel values to be negative.  The standard ISO1146-3 states
    "proper background subtraction there must exist negative noise values in the
    corrected power density distribution. These negative values have to be included in
    the further evaluation in order to allow compensation of positive noise amplitudes."

    Most image files are comprised of unsigned bytes or unsigned ints.  Thus to accomodate
    negative pixel values the image must become a signed image.

    If iso_noise=False then background_noise then negative pixels are set to zero.

    This could be done as a simple loop with an if statement but the
    implementation below is about 250X faster for 960 x 1280 arrays.

    Args:
        original: 2D array of an image with a beam in it
        background: 2D array of an image without a beam
        iso_noise: when subtracting, allow pixels to become negative
    Returns:
        image: 2D float array with background subtracted (may be signed)
    """
    # convert to signed version and subtract
    o = original.astype(float)
    b = background.astype(float)
    subtracted = o - b

    if not iso_noise:
        np.place(subtracted, subtracted < 0, 0)        # zero all negative values
#        return subtracted.astype(original.dtype.name)  # matching original type

    return subtracted


def subtract_constant(original,
                      background,
                      iso_noise=False):
    """
    Return image with a constant value subtracted.

    Subtract threshold from entire image.  If iso_noise is False
    then negative values are set to zero.

    The returned array is an array of float with the shape of original.

    Args:
        original : the image to work with
        background: value to subtract every pixel
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
    mask = corner_mask(image, corner_fraction)
    img = np.ma.masked_array(image, ~mask)
    mean = np.mean(img)
    stdev = np.std(img)
    return mean, stdev


def image_background(image,
                     corner_fraction=0.035,
                     nT=3):
    """
    Return the background for unilluminated pixels in an image.

    We first estimate the mean and standard deviation using the values in the
    corners.  All pixel values that fall below the mean+nT*stdev are considered
    un-illuminated (background) pixels.  These are averaged to find the background
    value for the image.

    Args:
        image : the image to work with
        nT: how many standard deviations to subtract
        corner_fraction: the fractional size of corner rectangles
    Returns:
        background: average background value across image
    """
    # estimate background
    ave, std = corner_background(image, corner_fraction=corner_fraction)

    # defined ISO/TR 11146-3:2004, equation 59
    threshold = ave + nT * std

    # collect all pixels that fall below the threshold
    unilluminated = image[image < threshold]

    mean = np.mean(unilluminated)
    stdev = np.std(unilluminated)
    return mean, stdev


def _mean_filter(values):
    return np.mean(values)


def _std_filter(values):
    return np.std(values)


def image_background2(image,
                      fraction=0.035,
                      nT=3):
    """
    Return the background of an image.

    The trick here is identifying unilluminated pixels.  This is done by using 
    using convolution to find the local average and standard deviation value for
    each pixel.  The local values are done over an n by m rectangle.

    ISO 11146-3 recommends using (n,m) values that are 2-5% of the image
    
    un-illuminated (background) pixels are all values that fall below the

    Args:
        image : the image to work with
        nT: how many standard deviations to subtract
        corner_fraction: the fractional size of corner rectangles
    Returns:
        background: average background value across image
    """
    # average over a n x m moving kernel
    n, m = (fraction * np.array(image.shape)).astype(int)
    ave = scipy.ndimage.generic_filter(image, _mean_filter, size=(n,m))
    std = scipy.ndimage.generic_filter(image, _std_filter, size=(n,m))

    # defined ISO/TR 11146-3:2004, equation 61
    threshold = ave + nT * std/np.sqrt((n+1)*(m+1))

    # we only average the pixels that fall below the illumination threshold
    unilluminated = image[image < threshold]

    background = int(np.mean(unilluminated))
    return background


def subtract_image_background(image,
                    corner_fraction=0.035,
                    nT=3,
                    iso_noise=False):
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
    Returns:
        image: 2D array with background subtracted
    """
    back, sigma = image_background(image, corner_fraction=corner_fraction, nT=nT)

    subtracted = image.astype(float)
    subtracted -= back

    if not iso_noise:  # zero pixels that fall within a few stdev
        threshold = nT * sigma
        np.place(subtracted, subtracted < threshold, 0)

    return subtracted


def subtract_corner_background(image,
                    corner_fraction=0.035,
                    nT=3,
                    iso_noise=False):
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


def subtract_tilted_background(image,
                               corner_fraction=0.035,
                               iso_noise=False):
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
    return subtract_background_image(image, z, iso_noise=iso_noise)
