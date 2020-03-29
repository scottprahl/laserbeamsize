from setuptools import setup

# long_description =
#     Simple and fast calculation of beam sizes from a single monochrome image based
#     on the ISO 11146 method of variances.  Some effort has been made to make
#     the algorithm less sensitive to background offset and noise.

# use README.rst as the long description
# make sure to use the syntax that works in both ReST and markdown
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type='text/x-rst'
)
