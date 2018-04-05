import sys
sys.path.append('..')
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

    def test_calc_h(self):

        trap_dict = self.f.trap_filters([[800., 120, 60], [800., 120, 60]])
        sed_list = [self.red_spec, self.blue_spec]
        sed_probs = [0.5, 0.5]
        test_c = calcIG(trap_dict, sed_list, sed_probs)

        self.assertEqual(-np.log2(0.5), test_c.calc_h())

        sed_probs2 = [0.7, 0.3]
        test_c2 = calcIG(trap_dict, sed_list, sed_probs2)

        self.assertEqual(-0.3*np.log2(.3) - 0.7*np.log2(0.7), test_c2.calc_h())

        sed_probs3 = [0.25, 0.25, 0.25, 0.25]
        sed_list3 = [self.red_spec]*4
        test_c3 = calcIG(trap_dict, sed_list3, sed_probs3)

        self.assertEqual(-np.log2(0.25), test_c3.calc_h())

    def test_calcIG(self):

        trap_dict = self.f.trap_filters([[800., 120, 60], [800., 120, 60]])
        sed_probs = [0.5, 0.5]
        test_c = calcIG(trap_dict, [self.red_spec, self.blue_spec],
                        sed_probs, snr=5.)
        ig = test_c.calc_IG()
        self.assertAlmostEqual(ig, 0., delta=0.01)

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
