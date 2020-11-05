import os
import unittest
import pickle
from siggi import siggi, spectra
from siggi.lsst_utils import BandpassDict
from siggi.filters import filterFactory
import numpy as np


class testSiggi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.f = filterFactory()
        s = spectra()
        cls.red_spec = s.get_red_spectrum()
        cls.blue_spec = s.get_blue_spectrum()

        cls.frozen_dict = BandpassDict.loadTotalBandpassesFromFiles(
                        bandpassNames=['u', 'g'],
                        bandpassDir=os.path.join(os.path.dirname(__file__),
                                                 '../siggi/data',
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
                            z_min=0.0, z_max=1.0, z_steps=21)

        random_state = np.random.RandomState(23)
        num_filters = 2
        set_ratio = 1.0
        set_width = 100
        t_1 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_type='trap',
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=14,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=random_state)

        t_2 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_type='trap',
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=14,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=23)

        np.testing.assert_array_equal(t_1.Xi, t_2.Xi)
        np.testing.assert_array_equal(t_1.yi, t_2.yi)
        np.testing.assert_almost_equal(np.max(np.abs(t_1.yi[:10])),
                                       1.2962127)
        self.assertGreaterEqual(np.max(np.abs(t_1.yi)), 1.2962127)

        # Test pickling of optimization can replicate results

        t_3 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=10,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           save_optimizer='test.pkl',
                                           rand_state=23)

        f = open('test.pkl', 'rb')
        test_opt = pickle.load(f)
        f.close()

        t_4 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=4,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           load_optimizer=test_opt,
                                           rand_state=23)

        np.testing.assert_array_equal(t_1.Xi, t_4.Xi)
        np.testing.assert_array_equal(t_3.Xi, t_4.Xi[:10])
        np.testing.assert_array_equal(t_1.yi, t_4.yi)
        np.testing.assert_array_equal(t_3.yi, t_4.yi[:10])

        # Test random point assignment after timeout

        t_5 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=14,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=2354,
                                           max_search_factor=0)

        t_6 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           system_wavelen_max=1200.,
                                           n_opt_points=10,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=2354,
                                           max_search_factor=0)

        self.assertGreater(len(t_5.Xi), 14)
        self.assertGreaterEqual(18, len(t_5.Xi))
        np.testing.assert_array_equal(t_6.Xi[:10], t_5.Xi[:10])
        np.testing.assert_array_equal(t_6.yi[:10], t_5.yi[:10])
        np.testing.assert_equal(len(t_5.yi), len(t_5.Xi))

        # Tests for set_width

        t_7 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           set_width=set_width,
                                           system_wavelen_max=1200.,
                                           n_opt_points=14,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=random_state)

        self.assertGreaterEqual(len(t_7.Xi), 14)
        np.testing.assert_equal(np.shape(t_7.Xi)[1], num_filters)
        np.testing.assert_equal(len(t_7.Xi), len(t_7.yi))

        # Test comb filters

        random_state = np.random.RandomState(23)
        num_filters = 1
        set_ratio = 1.0
        set_width = 100
        t_8 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_type='comb',
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           set_width=set_width,
                                           system_wavelen_max=1200.,
                                           n_opt_points=14,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=random_state)

        t_9 = sig_example.optimize_filters(num_filters=num_filters,
                                           filt_type='trap',
                                           filt_min=300., filt_max=1100.,
                                           set_ratio=set_ratio,
                                           set_width=set_width,
                                           system_wavelen_max=1200.,
                                           n_opt_points=14,
                                           optimizer_verbosity=10,
                                           procs=4,
                                           acq_func_kwargs_dict={'kappa': 3},
                                           frozen_filt_dict=self.frozen_dict,
                                           frozen_filt_eff_wavelen=[365, 477],
                                           starting_points=None,
                                           rand_state=23)

        np.testing.assert_array_equal(t_8.Xi, t_9.Xi)
        np.testing.assert_array_equal(t_8.yi, t_9.yi)

    @classmethod
    def tearDownClass(cls):

        os.remove('test.pkl')
        os.remove('Xi.out')
        os.remove('yi.out')

        return


if __name__ == '__main__':
    unittest.main()
