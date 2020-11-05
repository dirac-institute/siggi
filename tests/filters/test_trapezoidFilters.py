import unittest
import numpy as np
from siggi.filters import trapezoidFilter


class testTrapezoidFilters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        return

    def test_trap_filters(self):

        test_filters = trapezoidFilter()

        # Test that error raised in wavelen_grid not set
        with self.assertRaises(ValueError):
            test_filters.create_filter_dict_from_corners([1, 2, 3, 4])

        test_filters.set_wavelen_grid(wavelen_min=99.,
                                      wavelen_max=500.,
                                      wavelen_step=0.5)

        self.assertEqual(test_filters.wavelen_min, 99.)
        self.assertEqual(test_filters.wavelen_max, 500.)
        self.assertEqual(test_filters.wavelen_step, 0.5)

        # First test 1 filter in list by itself.

        test_filt_1 = [100, 150, 250, 300]

        t_f_1 = test_filters.create_filter_dict_from_corners(test_filt_1)

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

        t_f_2 = test_filters.create_filter_dict_from_corners(test_filt_2)

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

        t_f_3 = test_filters.create_filter_dict_from_corners(test_filt_3)

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
        np.testing.assert_array_equal(t_f_3['filter_0'].wavelen[[2, 102,
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

        # Test edge cases of triangular filters

        test_filters_2 = trapezoidFilter()
        test_filters_2.set_wavelen_grid()

        test_filt_4 = [[300.0, 300.0, 300.0, 1100.0]]

        t_f_4 = test_filters_2.create_filter_dict_from_corners(test_filt_4)

        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[:8000],
                                             np.linspace(1.0, 0.0, 8000))
        self.assertAlmostEqual(t_f_4['filter_0'].wavelen[8000], 1100.)

        test_filt_5 = [[300.0, 300.0, 300.0, 833.6770837679521]]

        t_f_5 = test_filters_2.create_filter_dict_from_corners(test_filt_5)

        max_idx = np.where(t_f_5['filter_0'].wavelen > 833.6770837679521)[0][0]

        np.testing.assert_array_almost_equal(t_f_5['filter_0'].sb[:max_idx],
                                             np.linspace(1.0, 0.0, max_idx))
        self.assertGreaterEqual(t_f_5['filter_0'].wavelen[max_idx],
                                833.6770837679521)

        test_filt_6 = [[400., 1200., 1200., 1200.]]

        t_f_6 = test_filters_2.create_filter_dict_from_corners(test_filt_6)

        np.testing.assert_array_almost_equal(t_f_6['filter_0'].sb[1000:],
                                             np.linspace(0.0, 1.0, 8001))
        self.assertAlmostEqual(t_f_6['filter_0'].wavelen[1000], 400.)

        return

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
