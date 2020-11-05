import unittest
import numpy as np
from siggi.filters import combFilter


class testCombFilters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        return

    def test_comb_filters(self):

        test_filters = combFilter()

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
        print(t_f_1)

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

        # Now test two different overlapping filters create one comb filter

        test_filt_3 = [[100, 150, 250, 300], [150, 200, 250, 350]]

        t_f_3 = test_filters.create_filter_dict_from_corners(test_filt_3)

        self.assertListEqual(t_f_3.keys(), ['filter_0'])

        np.testing.assert_array_almost_equal(t_f_3['filter_0'].wavelen,
                                             np.arange(99., 500.1, 0.5))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[:2],
                                             np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[2:103],
                                             np.linspace(0, 1, 101))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[102:352],
                                             np.ones(250))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[402:503],
                                             np.linspace(0.5, 0, 101))
        np.testing.assert_array_almost_equal(t_f_3['filter_0'].sb[502:],
                                             np.zeros(301))
        np.testing.assert_array_equal(t_f_3['filter_0'].wavelen[[2, 102,
                                                                 302, 402]],
                                      [100., 150., 250., 300.])
        # Test that 0 <= sb <= 1.
        self.assertEqual(np.max(t_f_3['filter_0'].sb), 1.0)
        self.assertEqual(np.min(t_f_3['filter_0'].sb), 0.0)

        # Now test two different non-overlapping filters create one comb filter

        test_filt_4 = [[100, 150, 200, 250], [300, 350, 400, 450]]

        t_f_4 = test_filters.create_filter_dict_from_corners(test_filt_4)

        self.assertListEqual(t_f_4.keys(), ['filter_0'])

        np.testing.assert_array_almost_equal(t_f_4['filter_0'].wavelen,
                                             np.arange(99., 500.1, 0.5))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[:2],
                                             np.zeros(2))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[2:103],
                                             np.linspace(0, 1, 101))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[102:202],
                                             np.ones(100))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[202:303],
                                             np.linspace(1, 0, 101))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[302:403],
                                             np.zeros(101))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[402:503],
                                             np.linspace(0, 1, 101))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[502:602],
                                             np.ones(100))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[602:703],
                                             np.linspace(1, 0, 101))
        np.testing.assert_array_almost_equal(t_f_4['filter_0'].sb[702:],
                                             np.zeros(101))
        np.testing.assert_array_equal(t_f_4['filter_0'].wavelen[[102, 202,
                                                                 302, 502]],
                                      [150., 200., 250., 350.])
        # Test that 0 <= sb <= 1.
        self.assertEqual(np.max(t_f_4['filter_0'].sb), 1.0)
        self.assertEqual(np.min(t_f_4['filter_0'].sb), 0.0)

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()