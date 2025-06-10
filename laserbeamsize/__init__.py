"""
A package to facilitate analysis of laser beam images.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

The `laserbeamsize` module contains functions for finding the size of an beam
using a single monochrome image. Details can be shown using::

    help(laserbeamsize.analysis)
    help(laserbeamsize.background)
    help(laserbeamsize.display)
    help(laserbeamsize.image_tools)
    help(laserbeamsize.masks)

Another module, `laserbeamsize.gaussian`, contains functions that find properties
of a propagating Gaussian beam::

    help(laserbeamsize.gaussian)

The last module, `laserbeamsize.m2`, contains functions that find the M² value
and other beam parameters from a sequence of images::

    help(laserbeamsize.m2_fit)
    help(laserbeamsize.m2_display)
"""

__version__ = "2.1.0"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2017-25, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/laserbeamsize"

from .masks import *
from .image_tools import *
from .background import *
from .analysis import *
from .display import *
from .gaussian import *
from .m2_fit import *
from .m2_display import *
