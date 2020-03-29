laserbeamsize
=============

Simple and fast calculation of beam sizes from a single monochrome image based
on the ISO 11146 method of variances.  Some effort has been made to make the 
algorithm less sensitive to background offset and noise.

Installation
------------

   pip install laserbeamsize

Usage
-----

Finding the center and dimensions of a good beam image::

import imageio
import laserbeamsize as lbs

beam = imageio.imread("t-hene.pgm")
x, y, dx, dy, phi = lbs.beam_size(beam)

print("The image center is at (%g, %g)" % (x,y))
print("The horizontal width is %.1f pixels" % dx)
print("The  vertical height is %.1f pixels" % dy)
print("The beam oval is rotated is %.1f°" % (phi*180/3.1416))

There are many examples at <https://laserbeamsize.readthedocs.io>

Source code repository
----------------------

    <https://github.com/scottprahl/laserbeamsize>

License
--------

laserbeamsize is licensed under the terms of the MIT license.
