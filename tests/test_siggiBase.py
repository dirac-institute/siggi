import sys
sys.path.append('..')
import os
import unittest
from siggi import _siggiBase, filters, spectra, calcIG
from siggi.lsst_utils import Bandpass, BandpassDict
import numpy as np


class testSiggi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.f = filters()
        s = spectra()
        cls.red_spec = s.get_red_spectrum()
        cls.blue_spec = s.get_blue_spectrum()

        cls.frozen_dict = BandpassDict.loadTotalBandpassesFromFiles(
                        bandpassNames=['u', 'g'],
                        bandpassDir=os.path.join(os.path.dirname(__file__),
                                                 '../data',
                                                 'lsst_baseline_throughputs'))

        return

    def test_find_filt_centers(self):

        test_sb = _siggiBase()

        t_f_4 = test_sb.find_filt_centers([400., 500., 600., 700.])
        self.assertAlmostEqual(t_f_4, [550.])

        # We calculate the center by finding the point where half of the
        # area under the transmission curve is to the left and half to
        # the right of the given point.
        t_f_5 = test_sb.find_filt_centers([[400., 500., 600., 700.],
                                           [400., 400., 800., 800.],
                                           [400., 400., 400., 800.]])
        self.assertAlmostEqual(t_f_5,
                               [550., 600., 800 - 400/np.sqrt(2)])

        return

    def test_set_starting_points(self):

        test_sb = _siggiBase()

        test_start_0 = test_sb.set_starting_points(None, 2, 400., 700.)[1]

        self.assertEqual(len(test_start_0), 10)
        self.assertListEqual(test_start_0[0], [400., 450., 500., 550.,
                                               550., 600., 650., 700.])
        self.assertListEqual(test_start_0[1], [400., 425., 450., 475.,
                                               475., 500., 525., 550.])
        self.assertListEqual(test_start_0[2], [550., 575., 600., 625.,
                                               625., 650., 675., 700.])

        test_start_1 = test_sb.set_starting_points(None, 2, 400., 700.,
                                                   ratio=0.5)[1]

        self.assertEqual(len(test_start_1), 10)
        self.assertListEqual(test_start_1[0], [400., 550.,
                                               550., 700.])
        self.assertListEqual(test_start_1[1], [400., 475.,
                                               475., 550.])
        self.assertListEqual(test_start_1[2], [550., 625.,
                                               625., 700.])

        self.assertRaises(AssertionError,
                          test_sb.set_starting_points, [[400., 500.]], 2, 400.,
                          700.)

        self.assertRaises(AssertionError,
                          test_sb.set_starting_points, [[400., 500.]], 2, 400.,
                          700., 0.5)

        test_start_2 = test_sb.set_starting_points([[400., 500., 600., 700.]],
                                                   2, 400., 700., ratio=0.5)[1]

        self.assertEqual(len(test_start_2), 10)
        self.assertListEqual(test_start_2[0], [400., 500., 600., 700.])
        self.assertListEqual(test_start_2[1], [400., 550.,
                                               550., 700.])
        self.assertListEqual(test_start_2[2], [400., 475.,
                                               475., 550.])
        self.assertListEqual(test_start_2[3], [550., 625.,
                                               625., 700.])

        return

    def test_validate_filter_input(self):

        test_sb = _siggiBase()

        self.assertRaises(AssertionError,
                          test_sb.validate_filter_input,
                          [200.]*4, 200., 200., 2)

        self.assertRaises(AssertionError,
                          test_sb.validate_filter_input,
                          [200.]*8, 200., 200., 2, 0.5)

        # Test that higher center filter is not to left of lower
        test_input_0 = test_sb.validate_filter_input([400., 401., 402., 403.,
                                                      300., 301., 302., 303.],
                                                     300., 600., 2)
        self.assertFalse(test_input_0)

        # Test that proper filters get through
        test_input_1 = test_sb.validate_filter_input([300., 301., 302., 303.,
                                                      400., 401., 402., 403.],
                                                     300., 600., 2)
        self.assertTrue(test_input_1)

        # Test that filter cannot be less than min allowed wavelength
        test_input_2 = test_sb.validate_filter_input([300., 301., 302., 303.,
                                                      400., 401., 402., 403.],
                                                     301., 600., 2)
        self.assertFalse(test_input_2)

        # Test that filter cannot be more than max allowed wavelength
        test_input_3 = test_sb.validate_filter_input([300., 301., 302., 303.,
                                                      400., 401., 402., 403.],
                                                     300., 402., 2)
        self.assertFalse(test_input_3)

if __name__ == '__main__':
    unittest.main()
