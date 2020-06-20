# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
"""
A module for finding M2 values for a laser beam.

Finding the center and dimensions of a monochrome image of a beam is simple::

    import numpy as np
    import laserbeamsize as lbs

    lambda0 = 632.8/1e6 # mm
    z = np.array([168, 210, 280, 348, 414, 480, 495, 510, 520, 580, 666, 770])
    d = np.array([0.5976914 , 0.57246158, 0.54747159, 0.55427816, 0.47916078,
           0.40394918, 0.41464084, 0.39929649, 0.3772103 , 0.39076051,
           0.32638856, 0.39693297])*2

    M2_analysis(z,d,lambda0)
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.ndimage

__all__ = ('z_rayleigh',
           'beam_radius',
           'focused_diameter',
           'M2_analysis',
           'M2_analysis2',
           )

def z_rayleigh(w0,lambda0):
    """
    Return the Rayleigh distance.
    Args:
        w0 : minimum beam radius [m]
        lambda0: wavelength of light [m]
    Returns:
        distance where irradiance drops by 1/2 [m]
    """
    return np.pi * w0**2/lambda0


def beam_radius(w0, lambda0, z, M2=1, z0=0, model='laboratory'):
    """
    Return the beam radius at an axial location.
  
    Args:
        w0 : minimum beam radius [m]
        z0 : axial location of beam waist [m]
        M2 : beam propagation factor [-]
        lambda0: wavelength of light [m]
        z : axial location of desired beam radius [m]
    Returns:
        Beam radius [m]
    """
    zz = (z-z0)/z_rayleigh(w0,lambda0)
    
    if model == 'illuminator' or model == 'constant waist':
        return w0*np.sqrt(1+(M2*zz)**2)

    if model == 'laboratory' or model == 'constant divergence':
        return w0*np.sqrt(M2**2+zz**2)

    return w0*M2*np.sqrt(1+zz**2)
    
def focused_diameter(f,lambda0,d, M2=1):
    """
    Return the diameter of diffraction-limited focused beam.
     Args:
        f : focal length of lens [m]
        lambda0: wavelength of light [m]
        d : diameter of limiting aperture [m]
        M2: beam propagation factor
    Returns:
        Beam diameter [m]
    """
    return 2 * lambda0 * M2 * f / d



def _params_from_abc(a, b, c, lambda0):

    z0 = -b/(2*c)
    M2 = np.pi/8/lambda0*np.sqrt(4*a*c-b*b)
    Theta0 = np.sqrt(c)/2
    w0 = np.sqrt((4*a*c-b*b)/(4*c))/2
    zR = np.sqrt((4*a*c-b*b)/(4*c*c))

    return M2, w0, Theta0, z0, zR

def M2_analysis(z,d,lambda0):

    a, b, c = poly.polyfit(z, d, 2)
    M2, w0, Theta0, z0, zR = _params_from_abc(a,b,c,lambda0)

    print("M2    = %.2f" % M2)
    print("w0    = %.2f mm"% w0)
    print("Theta = %.2f milliradians" % (1000*Theta0))
    print("zR    = %.0f mm"% zR)
    print("z0    = %.0f mm" % z0)

    zz = np.linspace(min(z)*0.9, max(z)*1.1, 100)
    ffit = a + b * zz + c * zz**2
    plt.plot(zz, ffit/2,':k')
    plt.plot(zz, -ffit/2,':k')

    plt.plot(z,d/2,'ob',markersize=2)
    plt.plot(z,-d/2,'ob',markersize=2)

    plt.axvline(z0)
    plt.axvline(z0-zR)

    plt.axhline(w0)
    plt.axhline(-w0)

    plt.show()
    return M2, w0, Theta0, z0, zR


def M2_analysis2(z,dx,dy,lambda0,f):

    ax, bx, cx = poly.polyfit(z, d, 2)
    M2x, w0x, Theta0x, z0x, zRx = _params_from_abc(ax,bx,cx,lambda0)

    a, b, c = poly.polyfit(z, d, 2)
    M2y, w0y, Theta0y, z0y, zRy = _params_from_abc(ay,by,cx,lambda0)
    
    w0 = np.sqrt((w0x**2 + w0y**2)/2)
    delta_z0 = abs(z0x - z0y)
    
    return M2, w0, Theta0, z0, zR


