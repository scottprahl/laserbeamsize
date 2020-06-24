# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
"""
A module for finding the beam size in an monochrome image.

Simple and fast calculation of beam sizes from a single monochrome image based
on the ISO 11146 method of variances.  Some effort has been made to make
the algorithm less sensitive to background offset and noise.

Finding the center and dimensions of a monochrome image of a beam is simple::

    import imageio
    import numpy as np
    import laserbeamsize as lbs

    beam = imageio.imread("t-hene.pgm")
    x, y, dx, dy, phi = lbs.beam_size(beam)

    print("The center of the beam ellipse is at (%.0f, %.0f)" % (x,y))
    print("The ellipse diameter (closest to horizontal) is %.0f pixels" % dx)
    print("The ellipse diameter (closest to   vertical) is %.0f pixels" % dy)
    print("The ellipse is rotated %.0f° ccw from the horizontal" % (phi*180/3.1416))

    ellipticity = dy/dx
    if ellipticity>0.87:
        print()
        print("The beam is circular because the ellipticity=%.2f > 0.87")
        print("The circular beam diameter is %.0f pixels" % np.sqrt((dx**2+dy**2)/2))
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
           'major_axis_arrays',
           'minor_axis_arrays',
           'visual_report',
           'plot_beam_fit',
           )


def rotate_points(x, y, x0, y0, phi):
    """
    Rotate x and y around designated center (x0,y0).

    Args:
        x: x-values of point or array of points to be rotated
        y: y-values of point or array of points to be rotated
        x0: horizontal center of rotation
        y0: vertical center of rotation
        phi: angle to rotate (+ is ccw) in radians

    Returns:
        x,y locations of rotated points
    """
    xp = x-x0
    yp = y-y0

    s = np.sin(-phi)
    c = np.cos(-phi)

    xf = xp * c - yp * s
    yf = xp * s + yp * c

    xf += x0
    yf += y0

    return xf, yf


def values_along_line(image, x0, y0, x1, y1, N=100):
    """
    Return x, y, z, and distance values between (x0,y0) and (x1,y1).

    This creates four arrays::
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x,y positions
        s: distance from start of minor axis to x,y position

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
        diameters:
    Returns:
        x, y, z, and s arrays
    """
    d = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    s = np.linspace(0, 1, N)

    x = x0 + s * (x1-x0)
    y = y0 + s * (y1-y0)

    xx = x.astype(int)
    yy = y.astype(int)

    zz = image[yy, xx]

    return xx, yy, zz, (s-0.5)*d


def major_axis_arrays(image, xc, yc, dx, dy, phi, diameters=3):
    """
    Return x, y, z, and distance values along semi-major axis.

    This creates four arrays::
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x,y positions
        s: distance from start of minor axis to x,y position

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
        diameters:
    Returns:
        x, y, z, and s arrays
    """
    v, h = image.shape

    if dx > dy:
        rx = diameters * dx / 2
        left = max(xc-rx, 0)
        right = min(xc+rx, h-1)
        x = np.array([left, right])
        y = np.array([yc, yc])
        xr, yr = rotate_points(x, y, xc, yc, phi)
    else:
        ry = diameters * dy / 2
        top = max(yc-ry, 0)
        bottom = min(yc+ry, v-1)
        x = np.array([xc, xc])
        y = np.array([top, bottom])
        xr, yr = rotate_points(x, y, xc, yc, phi)

    return values_along_line(image, xr[0], yr[0], xr[1], yr[1])


def minor_axis_arrays(image, xc, yc, dx, dy, phi, diameters=3):
    """
    Return x, y, z, and distance values along semi-minor axis.

    This creates four arrays::
        x: index of horizontal pixel values along line
        y: index of vertical pixel values along line
        z: image values at each of the x,y positions
        s: distance from start of minor axis to x,y position

    Args:
        image: the image to work with
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]
        diameters:
    Returns:
        x, y, z, and s arrays
    """
    v, h = image.shape

    if dx <= dy:
        rx = diameters * dx / 2
        left = max(xc-rx, 0)
        right = min(xc+rx, h-1)
        x = np.array([left, right])
        y = np.array([yc, yc])
        xr, yr = rotate_points(x, y, xc, yc, phi)
    else:
        ry = diameters * dy / 2
        top = max(yc-ry, 0)
        bottom = min(yc+ry, v-1)
        x = np.array([xc, xc])
        y = np.array([top, bottom])
        xr, yr = rotate_points(x, y, xc, yc, phi)

    return values_along_line(image, xr[0], yr[0], xr[1], yr[1])


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
    cy, cx = (np.array(original.shape)-1)/2.0

    # rotate image using defaults mode='constant' and cval=0.0
    rotated = scipy.ndimage.rotate(original, np.degrees(phi), order=1)

    # center of rotated image, defaults mode='constant' and cval=0.0
    ry, rx = (np.array(rotated.shape)-1)/2.0

    # position of (x0,y0) in rotated image
    new_x0, new_y0 = rotate_points(x0, y0, cx, cy, phi)
    new_x0 += rx - cx
    new_y0 += ry - cy

    voff = int(new_y0-y0)
    hoff = int(new_x0-x0)

    # crop so center remains in same location as original
    ov, oh = original.shape
    rv, rh = rotated.shape

    rv1 = max(voff, 0)
    sv1 = max(-voff, 0)
    vlen = min(voff+ov, rv) - rv1

    rh1 = max(hoff, 0)
    sh1 = max(-hoff, 0)
    hlen = min(hoff+oh, rh) - rh1

    # move values into zero-padded array
    s = np.full_like(original, 0)
    s[sv1:sv1+vlen, sh1:sh1+hlen] = rotated[rv1:rv1+vlen, rh1:rh1+hlen]
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

        `dx`,`dy` is the semi-major/minor diameters of the elliptical spot.

        `phi` is tilt of the ellipse from the axis [radians]

    Args:
        image: 2D array of image with beam spot
    Returns:
        beam parameters [xc, yc, dx, dy, phi]
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

    # Ensure that the case xx==yy is handled correctly
    if xx == yy:
        disc = 2*xy
        phi = np.sign(xy) * np.pi/4
    else:
        diff = xx-yy
        disc = np.sign(diff)*np.sqrt(diff**2 + 4*xy**2)
        phi = 0.5 * np.arctan(2*xy/diff)

    # finally, the major and minor diameters
    dx = np.sqrt(8*(xx+yy+disc))
    dy = np.sqrt(8*(xx+yy-disc))

    # phi is negative because image is inverted
    phi *= -1

    return xc, yc, dx, dy, phi


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
        boolean 2D array with ellipse marked True
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
    Create boolean mask for image with corners marked as True.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    Args:
        image : the image to work with
        corner_fraction: the fractional size of corner rectangles
    Returns:
        boolean 2D array with corners marked True
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
    if corner_fraction == 0:
        return 0, 0
    mask = corner_mask(image, corner_fraction)
    img = np.ma.masked_array(image, ~mask)
    mean = np.mean(img)
    stdev = np.std(img)
    return mean, stdev


def corner_subtract(image, corner_fraction=0.035, nT=3):
    """
    Return image with background subtracted.

    The background is estimated using the values in the cornes of the
    image.  The new image will have a constant (`mean+nT*stdev`) subtracted.

    ISO 11146-3 recommends values from 2-5% for `corner_fraction`.

    ISO 11146-3 recommends from 2-4 for `nT`.

    Some care has been taken to ensure that any values in the image that are
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


def rotated_rect_mask(image, xc, yc, dx, dy, phi, mask_diameters=3):
    """
    Create ISO 11146-3 rectangular mask for specified beam.

    The rectangular mask is `mask_diameters' times the pixel diameters
    of the ellipse.  ISO 11146 states that `mask_diameterd=3`.

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
        2D boolean array with appropriate mask
    """
    raw_mask = np.full_like(image, 0, dtype=float)
    v, h = image.shape
    rx = mask_diameters * dx / 2
    ry = mask_diameters * dy / 2
    vlo = max(0, int(yc-ry))
    vhi = min(v, int(yc+ry))
    hlo = max(0, int(xc-rx))
    hhi = min(h, int(xc+rx))

    raw_mask[vlo:vhi, hlo:hhi] = 1
    rot_mask = rotate_image(raw_mask, xc, yc, phi)
    return rot_mask


def rotated_rect_arrays(xc, yc, dx, dy, phi, mask_diameters=3):
    """
    Return x,y arrays to draw a rotated rectangle.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]

    Returns:
        x,y : two arrays for points on corners of rotated rectangle
    """
    rx = mask_diameters * dx / 2
    ry = mask_diameters * dy / 2

    # rectangle with center at (xc,yc)
    x = np.array([-rx, -rx, +rx, +rx, -rx]) + xc
    y = np.array([-ry, +ry, +ry, -ry, -ry]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return np.array([x_rot, y_rot])


def axes_arrays(xc, yc, dx, dy, phi, mask_diameters=3):
    """
    Return x,y arrays needed to draw semi-axes of ellipse.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: ellipse diameter for axis closest to horizontal
        dy: ellipse diameter for axis closest to vertical
        phi: angle that elliptical beam is rotated [radians]

    Returns:
        x,y arrays needed to draw semi-axes of ellipse
    """
    rx = mask_diameters * dx / 2
    ry = mask_diameters * dy / 2

    # major and minor ellipse axes with center at (xc,yc)
    x = np.array([-rx, rx, 0, 0, 0]) + xc
    y = np.array([0, 0, 0, -ry, ry]) + yc

    x_rot, y_rot = rotate_points(x, y, xc, yc, phi)

    return np.array([x_rot, y_rot])


def beam_size(image, mask_diameters=3, corner_fraction=0.035, nT=3, max_iter=25):
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
    using the values in the cornes of the image as `mean+nT*stdev`. ISO 11146
    states that `2<nT<4`.  The default value works fine.

    `max_iter` is the maximum number of iterations done before giving up.

    The returned parameters are::

        `xc`,`yc` is the center of the elliptical spot.

        `dx`,`dy` are the diameters of the elliptical spot.

        `phi` is tilt of the ellipse from the axis [radians]

    Args:
        image: 2D array of image of beam
        mask_diameters: the size of the integration rectangle in diameters
        corner_fraction: the fractional size of the corners
        nT: the multiple of background noise to remove
        max_iter: maximum number of iterations.
    Returns:
        elliptical beam parameters [xc, yc, dx, dy, phi]
    """
    # remove any offset
    zero_background_image = corner_subtract(image, corner_fraction, nT)

#    zero_background_image = np.copy(image)
    xc, yc, dx, dy, phi = basic_beam_size(zero_background_image)

    for _iteration in range(1, max_iter):

        xc2, yc2, dx2, dy2, _ = xc, yc, dx, dy, phi

        mask = rotated_rect_mask(image, xc, yc, dx, dy, phi, mask_diameters)
        masked_image = np.copy(zero_background_image)
        masked_image[mask < 1] = 0       # zero all values outside mask

        xc, yc, dx, dy, phi = basic_beam_size(masked_image)
        if abs(xc-xc2) < 1 and abs(yc-yc2) < 1 and abs(dx-dx2) < 1 and abs(dy-dy2) < 1:
            break

    return xc, yc, dx, dy, phi


def beam_test_image(h, v, xc, yc, dx, dy, phi, noise=0, max_value=255):
    """
    Create a test image.

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
        phi: angle that elliptical beam is rotated [radians]
        noise: normally distributed pixel noise to add to image
        max_value: all values in image fall between 0 and `max_value`
    Returns:
        2D image of astigmatic spot is v x h pixels in size
    """
    rx = dx/2
    ry = dy/2

    image0 = np.zeros([v, h])

    y, x = np.ogrid[:v, :h]

    scale = max_value - 3 * noise
    image0 = scale * np.exp(-2*(x-xc)**2/rx**2 - 2*(y-yc)**2/ry**2)

    image1 = rotate_image(image0, xc, yc, phi)

    if noise > 0:
        image1 += np.random.poisson(noise, size=(v, h))

        # after adding noise, the signal may exceed the range 0 to max_value
        np.place(image1, image1 > max_value, max_value)
        np.place(image1, image1 < 0, 0)

    if max_value < 256:
        return image1.astype(np.uint8)
    if max_value < 65536:
        return image1.astype(np.uint16)
    return image1


def ellipse_arrays(xc, yc, dx, dy, phi, npoints=200):
    """
    Return x,y arrays to draw a rotated ellipse.

    Args:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    Returns:
        x,y : two arrays of points on the ellipse
    """
    t = np.linspace(0, 2*np.pi, npoints)
    a = dx/2*np.cos(t)
    b = dy/2*np.sin(t)
    xp = xc + a*np.cos(phi) - b*np.sin(phi)
    yp = yc - a*np.sin(phi) - b*np.cos(phi)
    return np.array([xp, yp])


def plot_image_and_ellipse(image, xc, yc, dx, dy, phi, scale=1):
    """
    Plot the image, an ellipse, and center lines.

    Args:
        image: 2D array of image with beam spot
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
        scale: factor to increase/decrease ellipse size
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
    plt.title(r'c=(%.0f,%.0f), (dx,dy)=(%.1f,%.1f), $\phi$=%.1f°' %
              (xcc, ycc, dxx, dyy, ph))
    plt.xlim(0, h*scale)
    plt.ylim(v*scale, 0)
    plt.colorbar()


def basic_beam_size_naive(image):
    """
    Slow but simple implementation of ISO 11146 beam standard.

    This is identical to `basic_beam_size()` and is the obvious way to
    program the calculation of the necessary moments.  It is slow.

    Args:
        image: 2D array of image with beam spot in it
    Returns:
        beam parameters [xc, yc, dx, dy, phi]
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
    """Draw a simple astigmatic beam ellipse with labels."""
    theta = np.radians(30)
    xc = 0
    yc = 0
    dx = 50
    dy = 25

    plt.subplots(1, 1, figsize=(6, 6))

    # If the aspect ratio is not `equal` then the major and minor radii
    # do not appear to be orthogonal to each other!
    plt.axes().set_aspect('equal')

    xp, yp = ellipse_arrays(xc, yc, dx, dy, theta)
    plt.plot(xp, yp, 'k', lw=2)

    xp, yp = rotated_rect_arrays(xc, yc, dx, dy, theta)
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


def visual_report(image, title='Original', pixel_size=None, units='µm', **kwargs):
    """
    Create a visual report for image fitting.

    Args:
        image: 2D image of laser beam
        title: optional title for upper left plot
    Returns:
        nothing
    """
    if pixel_size is None:
        scale = 1
        unit_str = ''
        label = 'Pixels'
        units = 'pixels'
        label2 = 'Pixels from Center'
    else:
        scale = pixel_size
        unit_str = '[%s]' % units
        label = 'Position %s' % unit_str
        label2 = 'Distance from Center %s' % unit_str

    beamsize_keys = ['mask_diameters', 'corner_fraction', 'nT', 'max_iter']
    bs_args = dict((k, kwargs[k]) for k in beamsize_keys if k in kwargs)

    corner_keys = ['corner_fraction', 'nT']
    c_args = dict((k, kwargs[k]) for k in corner_keys if k in kwargs)

    xc, yc, dx, dy, phi = beam_size(image, **bs_args)
    working_image = corner_subtract(image, **c_args)
    background, _ = corner_background(image, **c_args)
    rx = dx/2
    ry = dy/2

    min_ = image.min()
    max_ = image.max()
    vv, hh = image.shape

    # scale all the dimensions
    v_s = vv * scale
    h_s = hh * scale
    xc_s = xc * scale
    yc_s = yc * scale
    dx_s = dx * scale
    dy_s = dy * scale
    
    plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(right=1.0)

    # original image
    plt.subplot(2, 2, 1)
    im = plt.imshow(image, extent=[0, h_s, v_s, 0], cmap='gist_ncar')
    plt.colorbar(im, fraction=0.046*v_s/h_s, pad=0.04) 
    plt.clim(min_, max_)
    plt.xlim(0, h_s)
    plt.ylim(v_s, 0)
    plt.xlabel(label)
    plt.ylabel(label)
    plt.title(title)

    # working image
    plt.subplot(2, 2, 2)
    extent=np.array([-xc_s, h_s-xc_s, v_s-yc_s, -yc_s])
    im = plt.imshow(working_image, extent=extent, cmap='gist_ncar')
    xp, yp = ellipse_arrays(xc, yc, dx, dy, phi) * scale
    plt.plot(xp-xc_s, yp-yc_s, 'w')
    xp, yp = axes_arrays(xc, yc, dx, dy, phi) * scale
    plt.plot(xp-xc_s, yp-yc_s, ':w')
    xp, yp = rotated_rect_arrays(xc, yc, dx, dy, phi) * scale
    plt.plot(xp-xc_s, yp-yc_s, ':w')
    plt.colorbar(im, fraction=0.046*v_s/h_s, pad=0.04) 
    plt.clim(min_, max_)
    plt.xlim(-xc_s, h_s-xc_s)
    plt.ylim(v_s-yc_s, -yc_s)
    plt.xlabel(label2)
    plt.ylabel(label2)
    plt.title('Image w/o background, center at (%.0f, %.0f) %s' % (xc_s, yc_s, units))

    # plot of values along semi-major axis
    _, _, z, s = major_axis_arrays(image, xc, yc, dx, dy, phi)
    a = np.sqrt(2/np.pi)/rx * np.sum(z-background)*abs(s[1]-s[0])
    baseline = a*np.exp(-2)+background

    plt.subplot(2, 2, 3)
    plt.plot(s*scale, z, 'b')
    plt.plot(s*scale, a*np.exp(-2*(s/rx)**2)+background, ':k')
    plt.annotate('', (-rx*scale, baseline), (rx*scale, baseline), arrowprops=dict(arrowstyle="<->"))
    plt.text(0, 1.1*baseline, 'dx=%.0f %s' % (dx_s,units), va='bottom', ha='center')
    plt.text(0, a, '  Gaussian')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Major Axis')
    plt.title('Semi-Major Axis')
    plt.gca().set_ylim(bottom=0)

    # plot of values along semi-minor axis
    _, _, z, s = minor_axis_arrays(image, xc, yc, dx, dy, phi)
    a = np.sqrt(2/np.pi)/ry * np.sum(z-background)*abs(s[1]-s[0])
    baseline = a*np.exp(-2)+background

    plt.subplot(2, 2, 4)
    plt.plot(s*scale, z, 'b')
    plt.plot(s*scale, a*np.exp(-2*(s/ry)**2)+background, ':k', label='fitted')
    plt.annotate('', (-ry*scale, baseline), (ry*scale, baseline), arrowprops=dict(arrowstyle="<->"))
    plt.text(0, 1.1*baseline, 'dy=%.0f %s' % (dy_s,units), va='bottom', ha='center')
    plt.text(0, a, '  Gaussian')
    plt.xlabel('Distance from Center [%s]' % units)
    plt.ylabel('Pixel Intensity Along Semi-Minor Axis')
    plt.title('Semi-Minor Axis')
    plt.gca().set_ylim(bottom=0)
    
    # add more horizontal space between plots
    plt.subplots_adjust(wspace=0.3)


def plot_beam_fit(image, pixel_size=None, vmax=None, units='µm'):
    """
    Plot the image, fitted ellipse, integration area, and center lines.

    Args:
        image: 2D array of image with beam spot
        pixel_size: (optional) size of pixels
        vmax: (optional) maximum value for colorbar
        units: (optional) string used for units used on axes
    Returns:
        xc: horizontal center of beam
        yc: vertical center of beam
        dx: horizontal diameter of beam
        dy: vertical diameter of beam
        phi: angle that elliptical beam is rotated [radians]
    """
    v,h = image.shape
    xc, yc, dx, dy, phi = beam_size(image)
    if vmax is None:
        vmax = image.max()

    if pixel_size is None:
        scale = 1
        label = 'Pixels'
    else:
        scale = pixel_size
        label = 'Position (%s)' % units

    extent = np.array([-xc,h-xc,v-yc,-yc])*scale
    plt.imshow(image, extent=extent, cmap='gist_ncar', vmax=vmax)
    plt.xlabel(label)
    plt.ylabel(label)

    xp,yp = axes_arrays(xc, yc, dx, dy, phi)
    plt.plot((xp-xc)*scale, (yp-yc)*scale, ':w')

    # show ellipse around beam
    xp,yp = ellipse_arrays(xc, yc, dx, dy, phi)
    plt.plot((xp-xc)*scale, (yp-yc)*scale, ':w')

    # show integration area around beam
    xp,yp = rotated_rect_arrays(xc, yc, dx, dy, phi)
    plt.plot((xp-xc)*scale, (yp-yc)*scale, ':w')

    plt.xlim(-xc*scale, (h-xc)*scale)
    plt.ylim((v-yc)*scale, -yc*scale)
    
    return xc, yc, dx, dy, phi