import sys
sys.path.append('..')
import unittest
import numpy as np
from siggi import filters


class testFilters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        return

    def test_trap_filters(self):

        test_filters = filters(wavelen_min=99.,
                               wavelen_max=500.,
                               wavelen_step=0.5)

        self.assertEqual(test_filters.wavelen_min, 99.)
        self.assertEqual(test_filters.wavelen_max, 500.)
        self.assertEqual(test_filters.wavelen_step, 0.5)

        # First test 1 filter in list by itself.

        test_filt_1 = [100, 150, 250, 300]

        t_f_1 = test_filters.trap_filters(test_filt_1)

        np.testing.assert_array_almost_equal(t_f_1['filter_0'].wavelen,
                                             np.arange(99., 500.1, 0.5))
        np.testing.assert_array_almost_equal(t_f_1['filter_0'].sb[:2],
                                             np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(t_f_1['filter_0'].sb[2:103],
                                             np.linspace(0, 1, 101))
        np.testing.assert_array_almost_equal(t_f_1['filter_0'].sb[102:302],
                                             np.ones(200))
        np.testing.assert_array_almost_equal(t_f_1['filter_0'].sb[302:403],
                                             np.linspace(1, 0, 101))
        np.testing.assert_array_almost_equal(t_f_1['filter_0'].sb[402:],
                                             np.zeros(401))
        np.testing.assert_array_equal(t_f_1['filter_0'].wavelen[[2, 102,
                                                                 302, 402]],
                                      [100., 150., 250., 300.])

        # All the same tests but with 1 filter list in a list

        test_filt_2 = [[100, 150, 250, 300]]

        t_f_2 = test_filters.trap_filters(test_filt_2)

        np.testing.assert_array_almost_equal(t_f_2['filter_0'].wavelen,
                                             np.arange(99., 500.1, 0.5))
        np.testing.assert_array_almost_equal(t_f_2['filter_0'].sb[:2],
                                             np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(t_f_2['filter_0'].sb[2:103],
                                             np.linspace(0, 1, 101))
        np.testing.assert_array_almost_equal(t_f_2['filter_0'].sb[102:302],
                                             np.ones(200))
        np.testing.assert_array_almost_equal(t_f_2['filter_0'].sb[302:403],
                                             np.linspace(1, 0, 101))
        np.testing.assert_array_almost_equal(t_f_2['filter_0'].sb[402:],
                                             np.zeros(401))
        np.testing.assert_array_equal(t_f_2['filter_0'].wavelen[[2, 102,
                                                                 302, 402]],
                                      [100., 150., 250., 300.])

        # Now test two different filters in same filter dict and add one
        # with a different shape

        test_filt_3 = [[100, 150, 250, 300], [150, 200, 250, 350]]

        t_f_3 = test_filters.trap_filters(test_filt_3)

        self.assertListEqual(t_f_3.keys(), ['filter_0', 'filter_1'])

        np.testing.assert_array_almost_equal(t_f_3['filter_0'].wavelen,
                                             np.arange(99., 500.1, 0.5))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[:2],
                                             np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[2:103],
                                             np.linspace(0, 1, 101))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[102:302],
                                             np.ones(200))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[302:403],
                                             np.linspace(1, 0, 101))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[402:],
                                             np.zeros(401))
        np.testing.assert_array_equal(t_f_2['filter_0'].wavelen[[2, 102,
                                                                 302, 402]],
                                      [100., 150., 250., 300.])

        np.testing.assert_array_almost_equal(t_f_3['filter_1'].wavelen,
                                             np.arange(99., 500.1, 0.5))
        np.testing.assert_array_almost_equal(t_f_3['filter_1'].sb[:102],
                                             np.zeros(102))
        np.testing.assert_array_almost_equal(t_f_3['filter_1'].sb[102:203],
                                             np.linspace(0, 1, 101))
        np.testing.assert_array_almost_equal(t_f_3['filter_1'].sb[202:302],
                                             np.ones(100))
        np.testing.assert_array_almost_equal(t_f_3['filter_1'].sb[302:503],
                                             np.linspace(1, 0, 201))
        np.testing.assert_array_almost_equal(t_f_3['filter_1'].sb[502:],
                                             np.zeros(301))
        np.testing.assert_array_equal(t_f_3['filter_1'].wavelen[[102, 202,
                                                                 302, 502]],
                                      [150., 200., 250., 350.])

        return

    def test_find_filt_centers(self):

        test_filters = filters()

        t_f_4 = test_filters.find_filt_centers([400., 500., 600., 700.])
        self.assertAlmostEqual(t_f_4, [550.])

        # We calculate the center by finding the point where half of the
        # area under the transmission curve is to the left and half to
        # the right of the given point.
        t_f_5 = test_filters.find_filt_centers([[400., 500., 600., 700.],
                                                [400., 400., 800., 800.],
                                                [400., 400., 400., 800.]])
        self.assertAlmostEqual(t_f_5,
                               [550., 600., 800 - 400/np.sqrt(2)])

        return

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
