"""
A package to facilitate analysis of laser beam images.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

The `laserbeamsize` module contains functions for finding the size of an beam
using a single monochrome image. Details can be shown using::

    help(laserbeamsize.analysis)
    help(laserbeamsize.background)
    help(laserbeamsize.display)
    help(laserbeamsize.image_tools)

Another module, `laserbeamsize.gaussian`, contains functions that find properties
of a propagating Gaussian beam::

    help(laserbeamsize.gaussian)

The M² analysis functionality is split into two modules.
`laserbeamsize.m2_fit` contains the fitting functions, and
`laserbeamsize.m2_display` contains the plotting/reporting functions::

    help(laserbeamsize.m2_fit)
    help(laserbeamsize.m2_display)
"""

__version__ = "2.4.1"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2017-2026, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/laserbeamsize"

from .image_tools import *
from .background import *
from .analysis import *
from .display import *
from .gaussian import *
from .m2_fit import *
from .m2_display import *
