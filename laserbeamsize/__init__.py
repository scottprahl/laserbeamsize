"""
A package to facilitate analysis of laser beam images.

Full documentation is available at <https://laserbeamsize.readthedocs.io>

Local help is available for the two modules that comprise the
laserbeamsize package. laserbeamsize.laserbeamsize has functions
for finding the size of an beam from as single image. This
information can be found using

help(laserbeamsize.laserbeamsize)

The other module has the functions that find the M² value and other
beam parameters from a sequence of images.  This information can be
found using

help(laserbeamsize.m2)
"""
__version__ = '1.9.2'
__author__ = 'Scott Prahl'
__email__ = 'scott.prahl@oit.edu'
__copyright__ = 'Copyright 2017-22, Scott Prahl'
__license__ = 'MIT'
__url__ = 'https://github.com/scottprahl/laserbeamsize.git'

from .laserbeamsize import *
from .m2 import *
