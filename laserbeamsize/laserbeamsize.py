import numpy as np
import matplotlib.pyplot as plt

def basic_beam_size(image):
    """ 
    Determines laser beam center, diameters, and tilt of beam according to ISO 11146 standard.
    
    The function does nothing to eliminate noise.  It just finds the first and second order moments
    and returns the beam parameters.  This implementation is roughly 800X faster than one that finds
    the moments using for loops.
    
    Parameters
    ----------
    image : array_like
        should be a monochrome two-dimensional array
        
    Returns
    -------
    xc : int
        horizontal center of beam
    yc : int 
        vertical center of beam
    dx : float
        horizontal diameter of beam
    dy : float
        vertical diameter of beam
    phi: float
        angle that beam is rotated (about center) from the horizontal axis

    Notes
    -----
    Uses the ISO 11146 definitions
    
    Examples
    --------
    Maybe one day there will be an example
    """

    
    #vertical and horizontal dimensions of beam
    v,h = image.shape
    
    # total of all pixels
    p  = np.sum(image,dtype=np.float)     # float avoids integer overflow

    # find the centroid
    hh = np.arange(h,dtype=np.float)      # float avoids integer overflow
    vv = np.arange(v,dtype=np.float)      # ditto
    xc = int(np.sum(np.dot(image,hh))/p)
    yc = int(np.sum(np.dot(image.T,vv))/p)
      
    # find the variances
    hs = hh-xc
    vs = vv-yc
    xx = np.sum(np.dot(image,hs**2))/p
    xy = np.dot(np.dot(image.T,vs),hs)/p
    yy = np.sum(np.dot(image.T,vs**2))/p

    # the ISO measures
    diff = xx-yy
    summ = xx+yy

    # Ensure that the case xx==yy is handled correctly
    if diff :
    	disc = np.sign(diff)*np.sqrt(diff**2 + 4*xy**2)
    else :
    	disc = np.abs(xy)
    dx = 2.0*np.sqrt(2)*np.sqrt(summ+disc)
    dy = 2.0*np.sqrt(2)*np.sqrt(summ-disc)

    phi = -0.5 * np.arctan2(2*xy,diff) # negative because top of matrix is zero
    
    return xc, yc, dx, dy, phi


def elliptical_mask(image, xc, yc, dx, dy, phi):
    """
    Return a boolean mask for a rotated elliptical disk. 

    Parameters
    ----------
    image : array_like
        should be a two-dimensional array
    xc : int
        horizontal center of beam
    yc : int 
        vertical center of beam
    dx : float
        horizontal diameter of beam
    dy : float
        vertical diameter of beam
    phi: float
        angle that beam is rotated (about center) from the horizontal axis
 
    Returns
    -------
    mask : array_like (boolean)
        same size as image
    """
    
    v,h = image.shape
    y,x = np.ogrid[:v,:h]

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    rx=dx/2
    ry=dy/2
    r2 = ((x-xc)*cosphi-(y-yc)*sinphi)**2/rx**2 + ((x-xc)*sinphi+(y-yc)*cosphi)**2/ry**2
    elliptical_mask = r2 <= 1

    return elliptical_mask


def beam_size(image, threshold=0.1, mask_diameters=2):
    """ 
    Determines laser beam center, diameters, and tilt of beam according to ISO 11146 standard.
    
    The function first estimates the beam parameters by excluding all points that are less
    than 10% of the maximum value in the image.  These parameters are refined by masking all 
    values more than two radii from the beam and recalculating.

    Parameters
    ----------
    image : array_like
        should be a monochrome two-dimensional array

    threshold : float, optional
        used to eliminate points outside the beam that confound estimating the beam parameters
        
    mask_diameters: float, optional
        when masking the beam for the final estimation, this determines the size of the elliptical mask
        
    Returns
    -------
    xc : int
        horizontal center of beam
    yc : int 
        vertical center of beam
    dx : float
        horizontal diameter of beam
    dy : float
        vertical diameter of beam
    phi: float
        angle that beam is rotated (about center) from the horizontal axis

    Notes
    -----
    Uses the ISO 11146 definitions
    
    Examples
    --------
    Maybe one day there will be an example
    """
    
    # use a 10% threshold to get rough idea of beam parameters
    thresholded_image = np.copy(image) 
    
    # remove possible offset
    minn = thresholded_image.min() # remove any offset        
    thresholded_image -= minn
    
    # remove all values less than threshold*max
    maxx = thresholded_image.max() 
    minn = int(maxx*threshold)
    np.place(thresholded_image, thresholded_image<minn, 0)

    # estimate the beam values
    xc, yc, dx, dy, phi = basic_beam_size(thresholded_image)

    # create a that is twice the estimated beam size 
    mask = elliptical_mask(image,xc,yc,mask_diameters*dx,mask_diameters*dy, phi)
    masked_image=np.copy(image)
    
    # find the minimum in the region of the mask
    maxx = masked_image.max()
    masked_image[~mask]=maxx    # exclude max values
    
    minn = masked_image.min()   # remove offset everywhere
    masked_image -= minn
    
    masked_image[~mask]=0       # zero all masked values

    return basic_beam_size(masked_image)


def beam_test_image(h,v,xc,yc,dx,dy,phi, offset=0, noise=0, max_value=256):
    """
    Create a v x h image with an elliptical beam centered at (xc,yc) with diameters dx and dy 
    that is rotated around the center by an angle phi from the horizontal
    """
        
    rx=dx/2
    ry=dy/2

    image = np.zeros([v,h])
    
    y,x = np.ogrid[:v,:h]
    
    # translate center of ellipse
    y -= yc
    x -= xc
    
    # needed to rotate ellipse
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    r2 = ((x*cosphi-y*sinphi)/rx)**2 + ((-x*sinphi-y*cosphi)/ry)**2
    image = np.exp(-2*r2)            

    scale = max_value/np.max(image)
    image *= scale
    
    if noise > 0 :
        image += np.random.normal(offset,noise,size=(v,h))
    
        # after adding noise, the signal may exceed the range 0 to max_value
        np.place(image,image>max_value,max_value)
        np.place(image,image<0,0)
    
    return image


def ellipse_arrays(xc,yc,dx,dy,phi, npoints=200):
    """
    Returns two arrays containing points that correspond the ellipse
    rotated about its center.  The center is at (xc,yc), the diameters 
    are dx and dy, and the rotation angle is phi
    """
    t = np.linspace(0,2*np.pi,npoints)
    a = dx/2*np.cos(t)
    b = dy/2*np.sin(t)
    xp = xc + a*np.cos(phi) - b*np.sin(phi)         
    yp = yc - a*np.sin(phi) - b*np.cos(phi)
    return xp, yp


def plot_image_and_ellipse(image,xc,yc,dx,dy,phi, scale=1):
    """
    Draws the image, an ellipse, and center lines
    """
    v,h = image.shape
    xp,yp = ellipse_arrays(xc,yc,dx,dy,phi)
    xp *= scale
    yp *= scale
    xcc = xc * scale
    ycc = yc * scale
    dxx = dx * scale
    dyy = dy * scale
    ph = phi * 180/np.pi
    
    # show the beam image with actual dimensions on the axes
    plt.imshow(image, extent=[0,h*scale,v*scale,0],cmap='gray')
    plt.plot(xp,yp,':y')
    plt.plot([xcc,xcc],[0,v*scale],':y')
    plt.plot([0,h*scale],[ycc,ycc],':y')
    plt.title('c=(%.0f,%.0f), (dx,dy)=(%.1f,%.1f), $\phi$=%.1f°'%(xcc,ycc,dxx,dyy,ph))
    plt.xlim(0,h*scale)
    plt.ylim(v*scale,0)
    plt.colorbar()


def basic_beam_size_naive(image):
    """
    Slow but simple implementation of ISO 1146 beam standard
    """
    v,h = image.shape
    
    # locate the center just like ndimage.center_of_mass(image)
    p = 0.0
    xc = 0.0
    yc = 0.0
    for i in range(v):
        for j in range(h):
            p  += image[i,j]
            xc += image[i,j]*j
            yc += image[i,j]*i
    xc = int(xc/p)
    yc = int(yc/p)

    # calculate variances
    xx=0.0
    yy=0.0
    xy=0.0
    for i in range(v):
        for j in range(h):
            xx += image[i,j]*(j-xc)**2
            xy += image[i,j]*(j-xc)*(i-yc)
            yy += image[i,j]*(i-yc)**2
    xx /= p
    xy /= p
    yy /= p

    # compute major and minor axes as well as rotation angle
    dx = 2*np.sqrt(2)*np.sqrt(xx+yy+np.sign(xx-yy)*np.sqrt((xx-yy)**2+4*xy**2))
    dy = 2*np.sqrt(2)*np.sqrt(xx+yy-np.sign(xx-yy)*np.sqrt((xx-yy)**2+4*xy**2))
    phi = 2 * np.arctan2(2*xy,xx-yy)
    
    return xc, yc, dx, dy, phi


def draw_beam_figure():
    """
    Draw a simple astigmatic beam.  A super confusing thing is that python designates the 
    top left corner as (0,0).  This is usually not a problem, but one has to be careful drawing
    rotated ellipses.
    """
    theta = 30*np.pi/180
    xc=0
    yc=0
    dx=50
    dy=20
    xp,yp = ellipse_arrays(xc,yc,dx,dy,theta)

    plt.plot(xp,yp,color='black',lw=2)
    sint = np.sin(theta)/2
    cost = np.cos(theta)/2
    plt.plot([xc-dx*cost,xc+dx*cost],[yc+dx*sint,yc-dx*sint],':b')
    plt.plot([xc+dy*sint,xc-dy*sint],[yc+dy*cost,yc-dy*cost],':g')

    plt.annotate(r'$x$', xy=(-25,0), xytext=(25,0), arrowprops=dict(arrowstyle="<-"),va='center', fontsize=16)
    plt.annotate(r'$y$', xy=(0,25), xytext=(0,-25), arrowprops=dict(arrowstyle="<-"), ha='center', fontsize=16)

    plt.annotate(r'$\phi$',xy=(13,-2.5),fontsize=16)
    plt.annotate('', xy=(15.5,0), xytext=(14, -8.0), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=-0.2"))
    plt.annotate(r'$d_x$',xy=(-17,7),color='blue',fontsize=16)
    plt.annotate(r'$d_y$',xy=(-3,-6),color='green',fontsize=16)

    plt.title("Simple Astigmatic Beam")
    plt.xlim(-25,25)
    plt.ylim(25,-30)  #inverted to match image coordinates!
    plt.axis('off')

    plt.show()
