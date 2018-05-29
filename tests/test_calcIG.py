import sys
sys.path.append('..')
import unittest
import numpy as np
from siggi import filters, spectra, calcIG
from siggi import Sed, Bandpass, BandpassDict
from siggi.lsst_utils import PhotometricParameters, calcMagError_sed
from copy import deepcopy


class testSiggi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.f = filters()
        s = spectra()
        cls.red_spec = s.get_red_spectrum()
        cls.blue_spec = s.get_blue_spectrum()
        cls.sky_spec = s.get_dark_sky_spectrum()

        cls.red_spec_z_1 = deepcopy(cls.red_spec)
        cls.red_spec_z_1.redshiftSED(1.0)
        cls.blue_spec_z_1 = deepcopy(cls.blue_spec)
        cls.blue_spec_z_1.redshiftSED(1.0)

        cls.imsimBand = Bandpass()
        cls.imsimBand.imsimBandpass()

        cls.phot_params = PhotometricParameters()

    def test_calc_colors(self):

        trap_dict = self.f.trap_filters([[740., 770., 830., 860.],
                                         [740., 770., 830., 860.]])
        filt_dict, atmos_filt_dict = \
            BandpassDict.addSystemBandpass(trap_dict)

        sed_list = [self.red_spec, self.red_spec]
        sed_probs = [0.5, 0.5]
        test_c = calcIG(trap_dict, sed_list, sed_probs,
                        sed_mags=np.random.uniform(21.0, 23.0))

        colors, errors = test_c.calc_colors()

        np.testing.assert_equal(colors, np.zeros(np.shape(colors)))

        sky_fn = self.sky_spec.calcFluxNorm(19.0, self.imsimBand)
        self.sky_spec.multiplyFluxNorm(sky_fn)

        test_error = calcMagError_sed(test_c._sed_list[0],
                                      atmos_filt_dict['filter_0'],
                                      self.sky_spec, filt_dict['filter_0'],
                                      self.phot_params, 1.0)

        np.testing.assert_almost_equal(errors, [[test_error*np.sqrt(2)],
                                                [test_error*np.sqrt(2)]])

        trap_dict_2 = self.f.trap_filters([[420., 435., 465., 480.],
                                           [770., 785., 815., 830.],
                                           [970., 985., 1015., 1030.]])
        filt_dict_2, atmos_filt_dict_2 = \
            BandpassDict.addSystemBandpass(trap_dict_2)

        sky_fn2 = self.sky_spec.calcFluxNorm(19.0, filt_dict_2['filter_0'])
        self.sky_spec.multiplyFluxNorm(sky_fn2)
        sky_mags = filt_dict_2.magListForSed(self.sky_spec)

        sed_1 = Sed()
        sed_1.setSED(wavelen=np.linspace(200., 1500., 1301),
                     flambda=np.ones(1301))
        imsim_f_norm = sed_1.calcFluxNorm(19.0, self.imsimBand)
        sed_1.multiplyFluxNorm(imsim_f_norm)
        f_norm_1_0 = sed_1.calcFluxNorm(15.0, filt_dict_2['filter_0'])
        f_norm_1_1 = sed_1.calcFluxNorm(14.0, filt_dict_2['filter_1'])
        f_norm_1_2 = sed_1.calcFluxNorm(13.0, filt_dict_2['filter_2'])
        flambda_1 = sed_1.flambda
        flambda_1[:500] *= f_norm_1_0
        flambda_1[500:700] *= f_norm_1_1
        flambda_1[700:] *= f_norm_1_2
        sed_1.flambda = flambda_1

        test_c2 = calcIG(trap_dict_2, [sed_1], [1.0], sed_mags=15.0,
                         ref_filter=filt_dict_2['filter_0'])

        colors2, errors2, snr2, mags2, sky_m2 = test_c2.calc_colors(
                                                        return_all=True)

        np.testing.assert_almost_equal(sky_mags, sky_m2)

        np.testing.assert_almost_equal([[15.0, 14.0, 13.0]], mags2, 5)

        np.testing.assert_almost_equal([[1.0, 1.0]], colors2, 5)

    def test_calc_h(self):

        trap_dict = self.f.trap_filters([[740., 770., 830., 860.],
                                         [740., 770., 830., 860.]])
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

    def test_calc_hyx(self):

        trap_dict = self.f.trap_filters([[740., 770., 830., 860.],
                                         [740., 770., 830., 860.]])
        sed_probs = [0.5, 0.5]
        test_c = calcIG(trap_dict, [self.red_spec, self.red_spec],
                        sed_probs)
        hy = test_c.calc_h()
        colors, errors = test_c.calc_colors()
        hyx = test_c.calc_hyx(colors, errors)

        self.assertAlmostEqual(hy, hyx, delta=0.01)

    def test_calcIG(self):

        # With same filter should be zero information gain. All colors = 0
        trap_dict = self.f.trap_filters([[740., 770., 830., 860.],
                                         [740., 770., 830., 860.],
                                         [740., 770., 830., 860.],
                                         [740., 770., 830., 860.],
                                         [740., 770., 830., 860.],
                                         [740., 770., 830., 860.]])
        sed_probs = [0.25, 0.25, 0.25, 0.25]

        test_c = calcIG(trap_dict, [self.red_spec, self.red_spec,
                                    self.red_spec, self.red_spec],
                        sed_probs, sed_mags=23.)

        ig = test_c.calc_IG()
        self.assertAlmostEqual(ig, 0., delta=0.01)

        # At very high signal to noise information gain should be perfect
        trap_dict_2 = self.f.trap_filters([[340., 370., 430., 460.],
                                           [540., 570., 630., 660.],
                                           [740., 770., 830., 860.]])

        sed_probs_2 = [0.25, 0.25, 0.25, 0.25]
        test_c_2 = calcIG(trap_dict_2, [self.red_spec, self.red_spec_z_1,
                                        self.blue_spec, self.blue_spec_z_1],
                          sed_probs_2, sed_mags=10.0, sky_mag=20.0)
        ig_2 = test_c_2.calc_IG()
        self.assertAlmostEqual(ig_2, 2.0, delta=0.01)

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
