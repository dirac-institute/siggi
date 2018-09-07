import sys
sys.path.append('..')
import os
import unittest
from siggi import siggi, filters, spectra, calcIG
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

    def test_siggi(self):

        # Basically just using a fast version of
        # the intro example notebook as a test

        def prior_z(z, z0=0.5):

            return ((z**2.)*np.exp(-(z/z0)**1.5) /
                    (np.sum((np.arange(0, 2.51, .05)**2.) *
                     np.exp(-(np.arange(0, 2.51, .05)/z0)**1.5))))

        sig_example = siggi([self.red_spec, self.blue_spec],
                            [0.5, 0.5], prior_z,
                            z_min=0.1, z_max=1.0, z_steps=20)

        random_state = np.random.RandomState(23)
        num_filters = 2
        set_ratio = 1.0
        t_1 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_min=300., filt_max=1100.,
                                           sed_mags=22.0,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=10,
                                           optimizer_verbosity=10,
                                           procs=4, 
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=random_state)

        t_2 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_min=300., filt_max=1100.,
                                           sed_mags=22.0,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=10,
                                           optimizer_verbosity=10,
                                           procs=4, 
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=23)

        np.testing.assert_array_equal(t_1.Xi, t_2.Xi)
        np.testing.assert_array_equal(t_1.yi, t_2.yi)
        np.testing.assert_almost_equal(np.max(np.abs(t_1.yi)), 2.992060199)

    @classmethod
    def tearDownClass(cls):

        return


if __name__ == '__main__':
    unittest.main()
