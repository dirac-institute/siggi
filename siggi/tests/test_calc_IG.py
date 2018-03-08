import unittest
import numpy as np
from siggi import filters, spectra, calcIG


class testSiggi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.f = filters()
        s = spectra()
        cls.red_spec = s.get_red_spectrum()
        cls.blue_spec = s.get_blue_spectrum()

    def test_calcIG(self):

        trap_dict = self.f.trap_filters([[800., 120, 60], [800., 120, 60]])
        sed_probs = [0.5, 0.5]
        test_c = calcIG(trap_dict, [self.red_spec, self.blue_spec],
                        sed_probs, snr=5.)
        ig = test_c.calc_IG()
        self.assertAlmostEqual(ig, 0., places=2)

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()