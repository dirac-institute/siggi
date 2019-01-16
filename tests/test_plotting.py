import unittest
import numpy as np
from siggi import plotting, spectra


class testPlotting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        s = spectra()
        red_spec = s.get_red_spectrum()
        blue_spec = s.get_blue_spectrum()
        cls.spec_list = [red_spec, blue_spec]

        cls.best_point = [365.0, 485.0, 715.0, 835.0]

        return

    def test_plot_filters(self):

        sig_plot = plotting(self.spec_list, self.best_point,
                            frozen_filt_dict=None, set_ratio=0.5,
                            sed_mags=22.0)

        sig_plot.plot_filters()

        return

    def test_plot_color_color(self):

        sig_plot = plotting(self.spec_list, self.best_point,
                            frozen_filt_dict=None, set_ratio=0.5,
                            sed_mags=22.0)

        sig_plot.plot_color_color(['filter_0', 'filter_1',
                                   'filter_0', 'filter_1'],
                                  np.linspace(0.00, 0.0))

        return

    def test_plot_ig_space(self):

        sig_plot = plotting(self.spec_list, self.best_point,
                            frozen_filt_dict=None, set_ratio=0.5,
                            sed_mags=22.0)

        test_Xi = [[315.0, 435.0, 315.0, 435.0],
                   [315.0, 435.0, 340.0, 460.0],
                   [315.0, 435.0, 365.0, 485.0],
                   [340.0, 435.0, 315.0, 435.0],
                   [340.0, 435.0, 340.0, 460.0],
                   [340.0, 435.0, 365.0, 485.0]]

        test_yi = [0.20557157351949373,
                   0.2095352030315274,
                   0.2705995989606681,
                   0.20557157351949373,
                   0.2095352030315274,
                   0.2705995989606681]

        sig_plot.plot_ig_space(np.array(test_Xi), np.array(test_yi),
                               [0, 1])

        return

    def test_plot_color_distributions(self):

        sig_plot = plotting(self.spec_list, self.best_point,
                            frozen_filt_dict=None, set_ratio=0.5,
                            sed_mags=22.0)

        sig_plot.plot_color_distributions(['filter_0', 'filter_1'],
                                          np.linspace(0.00, 0.0))

        sig_plot.plot_color_distributions(['filter_0', 'filter_1'],
                                          np.linspace(0.00, 0.0),
                                          add_cbar=True)

        return

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
