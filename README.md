# laserbeamsize

Simple and fast calculation of beam sizes from a single monochrome image based
on the ISO 11146 method of variances.  Some effort has been made to make the 
algorithm less sensitive to background offset and noise.
	
## Usage
For examples and use cases, see test folder

## Installation

### First install this python module

One way is to use pip

    pip install laserbeamsize

Alternatively you can install from github

    git clone https://github.com/scottprahl/laserbeamsize.git

Test by changing the iadpython directory and doing

    nosetests laserbeamsize/test_laserbeamsize.py

Then, add the iadpython directory to your PYTHONPATH or somehow


### Dependencies

For installation: setuptools

Required Python modules: numpy, matplotlib


### License

laserbeamsize is licensed under the terms of the MIT license.