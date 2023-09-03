"""
A basic test case.
"""

import imageio
import laserbeamsize as lbs
import unittest


class TestBeamSize(unittest.TestCase):

    def test_k_200mm(self):
        beam = imageio.v2.imread("./docs/k-200mm.png")
        x, y, dx, dy, phi = lbs.beam_size(beam)
        self.assertLess((x - 580.9918940222431)**2, 0.0001)
        self.assertLess((y - 388.11071998805414)**2, 0.0001)
        self.assertLess((dx - 187.54830372731638)**2, 0.0001)
        self.assertLess((dy - 153.1901738600395)**2, 0.0001)
        self.assertLess((phi + 0.4898844182586432)**2, 0.0001)


if __name__ == '__main__':
    unittest.main()
