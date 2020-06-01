#pylint: disable=invalid-name
#pylint: disable=too-many-instance-attributes
#pylint: disable=anomalous-backslash-in-string
#pylint: disable=too-many-locals
#pylint: disable=too-many-arguments
"""
A module for finding the beam size in an image.

Simple and fast calculation of beam sizes from a single monochrome image based
on the ISO 11146 method of variances.  Some effort has been made to make
the algorithm less sensitive to background offset and noise.

Finding the center and dimensions of a good beam image::

    import imageio
    import laserbeamsize as lbs

    beam = imageio.imread("t-hene.pgm")
    x, y, dx, dy, phi = lbs.beam_size(beam)

    print("The image center is at (%g, %g)" % (x,y))
    print("The horizontal width is %.1f pixels" % dx)
    print("The vertical height is %.1f pixels" % dy)
    print("The beam oval is rotated %.1f°" % (phi*180/3.1416))
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

__all__ = ('subtract_image',
           'subtract_threshold',
           'corner_background',
           'corner_mask',
           'corner_subtract',
           'rotate_image',
           'rotated_rect_mask',
           'rotated_rect_arrays',
           'axes_arrays',
           'basic_beam_size',
           'basic_beam_size_naive',
           'beam_size',
           'beam_test_image',
           'draw_beam_figure',
           'ellipse_arrays',
           'elliptical_mask',
           'plot_image_and_ellipse',
           )

def rotate_points(x, y, x0, y0, phi):
    """Rotate x and y around designated center."""
    xp = x-x0
    yp = y-y0

    s = np.sin(-phi)
    c = np.cos(-phi)

    xf = xp * c - yp * s
    yf = xp * s + yp * c

    xf += x0
    yf += y0

    return xf, yf


def subtract_image(original, background):
    """
    Subtract background from original image.

    This is only needed because when subtracting some pixels may become
    negative.  Unfortunately when the arrays have an unsigned data type
    these negative values end up having very large pixel values.

    This could be done as a simple loop with an if statement but the
    implementation below is about 250X faster for 960 x 1280 arrays.

    Args:
        original: the image to work with
        background: the image to be subtracted
    Returns:
        subtracted image that matches the type of the original
    """
    # convert to signed version
    o = original.astype(int)
    b = background.astype(int)

    # subtract and zero negative entries
    r = o-b
    np.place(r, r < 0, 0)

    # return array that matches original type
    return r.astype(original.dtype.name)


def subtract_threshold(image, threshold):
    """
    Return image with constant subtracted.

    Subtract threshold from entire image.  Negative values are set to zero.

    Args:
        image : the image to work with
        threshold: value to subtract every pixel
    Returns:
        new image with threshold subtracted
    """
    subtracted = np.array(image)
    np.place(subtracted, subtracted < threshold, threshold)
    subtracted -= threshold
    return subtracted


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
        rotated image with same dimensions as original
    """
    # center of original image
    o_y, o_x = (np.array(original.shape)-1)/2.0

    # rotate image using defaults mode='constant' and cval=0.0
    rotated = scipy.ndimage.rotate(original, np.degrees(phi), order=1)

    # center of rotated image, defaults mode='constant' and cval=0.0
    r_y, r_x = (np.array(rotated.shape)-1)/2.0

    # new position of desired rotation center
    new_x0, new_y0 = rotate_points(x0, y0, o_x, o_y, phi)

    new_x0 += r_x - o_x
    new_y0 += r_y - o_y

    voff = int(new_y0-y0)
    hoff = int(new_x0-x0)

    # crop so center remains in same location as original
    ov, oh = original.shape
    rv, rh = rotated.shape

    rv_start = max(voff, 0)
    sv_start = max(-voff, 0)
    vlen = min(voff+ov, rv) - rv_start
    rv_end = rv_start+vlen
    sv_end = sv_start+vlen

    rh_start = max(hoff, 0)
    sh_start = max(-hoff, 0)
    hlen = min(hoff+oh, rh) - rh_start
    rh_end = rh_start+hlen
    sh_end = sh_start+hlen

    # move values into zero-padded array
    s = np.full_like(original, 0) # np.zeros() fails with boolean arrays
    s[sv_start:sv_end, sh_start:sh_end] = rotated[rv_start:rv_end, rh_start:rh_end]

    return s


def basic_beam_size(image):
    """
    Determine the beam center, diameters, and tilt using ISO 11146 standard.

    Find the center and sizes of an elliptical spot in an 2D array.

    The function does nothing to eliminate background noise.  It just finds the first
    and second order moments and returns the beam parameters. Consequently
    a beam spot in an image with a constant background will fail badly.

    FWIW, this implementation is roughly 800X faster than one that finds
    the moments using for loops.

    The returned parameters are::

        `xc`,`yc` is the center of the elliptical spot.

        `dx`,`dy` is the width and height of the elliptical spot.

        `phi` is tilt of the ellipse from the axis [radians]

    Parameters
    ----------
    image: 2D array
        image with beam spot in it

    Returns
    -------
    array:
        [xc, yc, dx, dy, phi]
    """
    v, h = image.shape

    # total of all pixels
    p = np.sum(image, dtype=np.float)     # float avoids integer overflow

    # sometimes the image is all zeros, just return
    if p == 0:
        return int(h/2), int(v/2), 0, 0, 0

    # find the centroid
    hh = np.arange(h, dtype=np.float)      # float avoids integer overflow
    vv = np.arange(v, dtype=np.float)      # ditto
    xc = int(np.sum(np.dot(image, hh))/p)
    yc = int(np.sum(np.dot(image.T, vv))/p)

    # find the variances
    hs = hh-xc
    vs = vv-yc
    xx = np.sum(np.dot(image, hs**2))/p
    xy = np.dot(np.dot(image.T, vs), hs)/p
    yy = np.sum(np.dot(image.T, vs**2))/p

    # the ISO measures
    diff = xx-yy
    summ = xx+yy

    # Ensure that the case xx==yy is handled correctly
    if diff:
        disc = np.sign(diff)*np.sqrt(diff**2 + 4*xy**2)
    else:
        disc = 2*xy

    dx = 2.0*np.sqrt(2)*np.sqrt(summ+disc)
    dy = 2.0*np.sqrt(2)*np.sqrt(summ-disc)

    # negative because top of matrix is zero
    if diff:
        phi = -0.5 * np.arctan2(2*xy, diff)
    else:
        phi = -np.sign(xy) * np.pi/4

    return xc, yc, dx, dy, phi


def elliptical_mask(image, xc, yc, dx, dy, phi):
    """
    Return a boolean mask for a rotated elliptical disk.

    The returned mask is the same size as `image`.

    Parameters
    ----------
    image: 2D array
    xc: float
        horizontal center of beam
    yc: int
        vertical center of beam
    dx: float
        horizontal diameter of beam
    dy: float
        vertical diameter of beam
    phi: float
        angle that elliptical beam is rotated (about center) from the horizontal axis in radians

    Returns
    -------
    mask: boolean 2D array
    """
    v, h = image.shape
    y, x = np.ogrid[:v, :h]

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    rx = dx/2
    ry = dy/2
    xx = x-xc
    yy = y-yc
    r2 = (xx*cosphi-yy*sinphi)**2/rx**2 + (xx*sinphi+yy*cosphi)**2/ry**2
    the_mask = r2 <= 1

    return the_mask


def corner_mask(image, corner_fraction=0.035):
    """
    Return mask of image with corners marked as True.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        new boolean image mask
    """
    v, h = image.shape
    n = int(v * corner_fraction)
    m = int(h * corner_fraction)

    the_mask = np.full_like(image, False, dtype=np.bool)
    the_mask[:n, :m] = True
    the_mask[:n, -m:] = True
    the_mask[-n:, :m] = True
    the_mask[-n:, -m:] = True
    return the_mask


def corner_background(image, corner_fraction=0.035):
    """
    Return mean and stdev of background in corners of image.

    The background is estimated using the average of the pixels in a
    n x m rectangle in each of the four corners of the image. Here n
    is the horizontal size multiplied by `corner_fraction`. Similar
    for m.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        average pixel value in corners
    """
    mask = corner_mask(image, corner_fraction)
    img = np.ma.masked_array(image, mask)
    mean = np.mean(img)
    stdev = np.std(img)
    return mean, stdev


def corner_subtract(image, corner_fraction=0.035, nT=3):
    """
    Return image with background subtracted.

    The background is estimated using the average of the pixels in a
    n x m rectangle in each of the four corners of the image. Here n
    is the horizontal size multiplied by `corner_fraction`. Similar
    for m.

    The new image will have `mean+nT*stdev` subtracted.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    ISO 11146-3 recommends from 2-4 for `nT`.

    Some care must be taken to ensure that any values in the image that are
    less than the background are set to zero.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        new image with background subtracted
    """
    back, sigma = corner_background(image, corner_fraction)
    offset = int(back + nT * sigma)
    return subtract_threshold(image, offset)


def rotated_rect_mask(image, xc, yc, dx, dy, phi):
    """
    Create ISO 1146-3 image mask for specified beam.

    The image is rotated about a centerpoint (x0, y0) and then
    cropped to the original size such that the centerpoint remains
    in the same location.

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        2D boolean array with appropriate mask
    """
    raw_mask = np.full_like(image, False, dtype=bool)

    dx *= 1.5
    dy *= 1.5
    vlo = int(yc-dy)
    vhi = int(yc+dy)
    hlo = int(xc-dx)
    hhi = int(xc+dx)

    raw_mask[vlo:vhi, hlo:hhi] = True
    rot_mask = rotate_image(raw_mask, xc, yc, phi)
    return rot_mask


def rotated_rect_arrays(xc, yc, dx, dy, phi):
    """
    Return x,y arrays to draw a rotated rectangle.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]

    Returns:
        x,y : two arrays for points on corners of rotated rectangle
    """
    dx *= 1.5
    dy *= 1.5

    # rectangle with center at (xc,yc)
    x = np.array([-dx, -dx, +dx, +dx, -dx]) + xc
    y = np.array([-dy, +dy, +dy, -dy, -dy]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return x_rot, y_rot


def axes_arrays(xc, yc, dx, dy, phi):
    """
    Return x,y arrays to draw a rotated rectangle.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]

    Returns:
        x,y : two arrays for points on corners of rotated rectangle
    """
    dx *= 1.5
    dy *= 1.5

    # major and minor ellipse axes with center at (xc,yc)
    x = np.array([-dx, dx, 0,   0,  0]) + xc
    y = np.array([  0,  0, 0, -dy, dy]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return x_rot, y_rot


def beam_size(image, mask_diameters=3, corner_fraction=0.035, nT=3, max_iter=10):
    """
    Determine beam parameters in an image with noise.

    The function first estimates the beam parameters by excluding all points
    that are less than 10% of the maximum value in the image.  These parameters
    are refined by masking all values more than three radii from the beam and
    recalculating.

    The returned parameters are::

        `xc`,`yc` is the center of the elliptical spot.

        `dx`,`dy` is the width and height of the elliptical spot.

        `phi` is tilt of the ellipse from the axis [radians]

    Parameters
    ----------
    image: 2D array
        should be a monochrome two-dimensional array

    threshold: float, optional
        used to eliminate points outside the beam that confound estimating
        the beam parameters

    mask_diameters: float, optional
        when masking the beam for the final estimation, this determines
        the size of the elliptical mask

    Returns
    -------
    array:
        [xc, yc, dx, dy, phi]
    """
    # remove any offset
    zero_background_image = corner_subtract(image, corner_fraction, nT)

    xc, yc, dx, dy, phi = basic_beam_size(zero_background_image)

    for _iteration in range(1, max_iter):

        xc2, yc2, dx2, dy2, _ = xc, yc, dx, dy, phi

        ddx = dx * mask_diameters/3
        ddy = dy * mask_diameters/3

        mask = rotated_rect_mask(image, xc, yc, ddx, ddy, phi)
        masked_image = np.copy(zero_background_image)
        masked_image[~mask] = 0       # zero all values outside mask

        xc, yc, dx, dy, phi = basic_beam_size(masked_image)
        if abs(xc-xc2) < 1 and abs(yc-yc2) < 1 and abs(dx-dx2) < 1 and abs(dy-dy2) < 1:
            break

    return xc, yc, dx, dy, phi


def beam_test_image(h, v, xc, yc, dx, dy, phi, offset=0, noise=0, max_value=255):
    """
    Create a test image.

    Create a v x h image with an elliptical beam with specified center and
    beam dimensions.  By default the values in the image will range from 0 to
    255. The default image will have no background and no noise.

    Args:
        h: horizontal size of image to generate
        v: vertical size of image to generate
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
        offset: background offset to be added to entire image
        noise: normally distributed pixel noise to add to image
        max_value: all values in image fall between 0 and `max_value`

    Returns:
        test_image: v x h pixels in size
    """
    rx = dx/2
    ry = dy/2

    image0 = np.zeros([v, h])

    y, x = np.ogrid[:v, :h]

    image0 = np.exp(-2*(x-xc)**2/rx**2 -2*(y-yc)**2/ry**2)

    image = rotate_image(image0, xc, yc, phi)

    scale = (max_value - 2 * noise)/np.max(image)
    image *= scale

    if noise > 0:
        image += np.random.normal(offset, noise, size=(v, h))

        # after adding noise, the signal may exceed the range 0 to max_value
        np.place(image, image > max_value, max_value)
        np.place(image, image < 0, 0)

    iimage = image.astype(int)
    return iimage


def ellipse_arrays(xc, yc, dx, dy, phi, npoints=200):
    """
    Return x,y arrays to draw a rotated ellipse.

    Parameters
    ----------
    xc: float
        horizontal center of beam
    yc: int
        vertical center of beam
    dx: float
        horizontal diameter of beam
    dy: float
        vertical diameter of beam
    phi: float
        angle that elliptical beam is rotated [radians]

    Returns
    -------
    x,y : two arrays of points on the ellipse
    """
    t = np.linspace(0, 2*np.pi, npoints)
    a = dx/2*np.cos(t)
    b = dy/2*np.sin(t)
    xp = xc + a*np.cos(phi) - b*np.sin(phi)
    yp = yc - a*np.sin(phi) - b*np.cos(phi)
    return xp, yp


def plot_image_and_ellipse(image, xc, yc, dx, dy, phi, scale=1):
    """
    Draw the image, an ellipse, and center lines.

    Parameters
    ----------
    image: 2D array
        image with beam spot
    xc: float
        horizontal center of beam
    yc: int
        vertical center of beam
    dx: float
        horizontal diameter of beam
    dy: float
        vertical diameter of beam
    phi: float
        angle that elliptical beam is rotated [radians]
    scale: float
        factor to increase/decrease ellipse size
    """
    v, h = image.shape
    xp, yp = ellipse_arrays(xc, yc, dx, dy, phi)
    xp *= scale
    yp *= scale
    xcc = xc * scale
    ycc = yc * scale
    dxx = dx * scale
    dyy = dy * scale
    ph = phi * 180/np.pi

    # show the beam image with actual dimensions on the axes
    plt.imshow(image, extent=[0, h*scale, v*scale, 0], cmap='gray')
    plt.plot(xp, yp, ':y')
    plt.plot([xcc, xcc], [0, v*scale], ':y')
    plt.plot([0, h*scale], [ycc, ycc], ':y')
    plt.title('c=(%.0f,%.0f), (dx,dy)=(%.1f,%.1f), $\phi$=%.1f°' %
              (xcc, ycc, dxx, dyy, ph))
    plt.xlim(0, h*scale)
    plt.ylim(v*scale, 0)
    plt.colorbar()


def basic_beam_size_naive(image):
    """
    Slow but simple implementation of ISO 1146 beam standard.

    This is the obvious way to calculate the moments.  It is slow.

    The returned parameters are::

        `xc`,`yc` is the center of the elliptical spot.

        `dx`,`dy` is the width and height of the elliptical spot.

        `phi` is tilt of the ellipse from the axis [radians]

    Parameters
    ----------
    image: (2D array)
        image with beam spot in it

    Returns
    -------
    array:
        [xc, yc, dx, dy, phi]
    """
    v, h = image.shape

    # locate the center just like ndimage.center_of_mass(image)
    p = 0.0
    xc = 0.0
    yc = 0.0
    for i in range(v):
        for j in range(h):
            p += image[i, j]
            xc += image[i, j]*j
            yc += image[i, j]*i
    xc = int(xc/p)
    yc = int(yc/p)

    # calculate variances
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for i in range(v):
        for j in range(h):
            xx += image[i, j]*(j-xc)**2
            xy += image[i, j]*(j-xc)*(i-yc)
            yy += image[i, j]*(i-yc)**2
    xx /= p
    xy /= p
    yy /= p

    # compute major and minor axes as well as rotation angle
    dx = 2*np.sqrt(2)*np.sqrt(xx+yy+np.sign(xx-yy)*np.sqrt((xx-yy)**2+4*xy**2))
    dy = 2*np.sqrt(2)*np.sqrt(xx+yy-np.sign(xx-yy)*np.sqrt((xx-yy)**2+4*xy**2))
    phi = 2 * np.arctan2(2*xy, xx-yy)

    return xc, yc, dx, dy, phi


def draw_beam_figure():
    """
    Draw a simple astigmatic beam.

    A super confusing thing is that python designates the top left corner as
    (0,0).  This is usually not a problem, but one has to be careful drawing
    rotated ellipses.  Also, if the aspect ratio is not set to be equal then
    the major and minor radii are not orthogonal to each other!
    """
    theta = np.radians(30)
    xc = 0
    yc = 0
    dx = 50
    dy = 25

    plt.subplots(1, 1, figsize=(6, 6))
    plt.axes().set_aspect('equal')

    xp, yp = ellipse_arrays(xc, yc, dx, dy, theta)
    plt.plot(xp, yp, 'k', lw=2)

    xp, yp = rotated_rect_arrays(xc, yc, 3*dx, 3*dy, theta)
    plt.plot(xp, yp, ':b', lw=2)

    sint = np.sin(theta)/2
    cost = np.cos(theta)/2
    plt.plot([xc-dx*cost, xc+dx*cost], [yc+dx*sint, yc-dx*sint], ':b')
    plt.plot([xc+dy*sint, xc-dy*sint], [yc+dy*cost, yc-dy*cost], ':r')

    # draw axes
    plt.annotate("x'", xy=(-25, 0), xytext=(25, 0),
                 arrowprops=dict(arrowstyle="<-"), va='center', fontsize=16)

    plt.annotate("y'", xy=(0, 25), xytext=(0, -25),
                 arrowprops=dict(arrowstyle="<-"), ha='center', fontsize=16)

    plt.annotate(r'$\phi$', xy=(13, -2.5), fontsize=16)
    plt.annotate('', xy=(15.5, 0), xytext=(
        14, -8.0), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=-0.2"))

    plt.annotate(r'$d_x$', xy=(-17, 7), color='blue', fontsize=16)
    plt.annotate(r'$d_y$', xy=(-4, -8), color='red', fontsize=16)

    plt.xlim(-30, 30)
    plt.ylim(30, -30)  # inverted to match image coordinates!
    plt.axis('off')
    plt.show()
