import unittest
import numpy as np
from siggi.filters import filterFactory


class testTrapezoidFilters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        return

    def test_find_filt_centers(self):

        test_f = filterFactory.create_filter_object('trap')
        test_f.set_wavelen_grid()
        test_f_dict = test_f.create_filter_dict_from_corners([400., 500.,
                                                              600., 700.])
        t_f_4 = test_f.find_filt_centers(test_f_dict)
        self.assertAlmostEqual(t_f_4[0], 550.)

        # We calculate the center by finding the point where half of the
        # area under the transmission curve is to the left and half to
        # the right of the given point.
        test_f_dict_2 = test_f.create_filter_dict_from_corners(
            [[400., 500., 600., 700.],
             [400., 400., 800., 800.],
             [400., 400., 400., 800.]]
        )
        t_f_5 = test_f.find_filt_centers(test_f_dict_2)
        np.testing.assert_array_almost_equal(np.array(t_f_5),
                                             np.array([550., 600.,
                                                       800 - 400/np.sqrt(2)]),
                                             decimal=1)

        return

    def test_calc_corners_from_shape_params(self):

        test_filters = [[350., 350., 650., 650.],
                        [500., 500., 800., 800.]]

        test_f = filterFactory.create_filter_object('trap')
        test_f.set_wavelen_grid()

        set_ratio = None
        set_width = None

        tf_1 = test_f.calc_corners_from_shape_params(
            set_ratio, set_width, np.array(test_filters).flatten()
        )

        np.testing.assert_array_equal(test_filters, tf_1)

        set_ratio = 1.0

        tf_2 = test_f.calc_corners_from_shape_params(
            set_ratio, set_width, np.array(test_filters).flatten()[::2]
        )

        np.testing.assert_array_equal(test_filters, tf_2)

        set_width = 300

        tf_3 = test_f.calc_corners_from_shape_params(
            set_ratio, set_width, np.array(test_filters).flatten()[::4]
        )

        np.testing.assert_array_equal(test_filters, tf_3)

        return

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
