import unittest
import numpy as np
from siggi import filters, spectra, calcIG
from siggi import Sed, Bandpass, BandpassDict
from siggi.lsst_utils import PhotometricParameters, calcMagError_sed
from copy import deepcopy


class testCalcIG(unittest.TestCase):

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
        hardware_filt_dict, total_filt_dict = \
            BandpassDict.addSystemBandpass(trap_dict)

        sed_list = [self.red_spec, self.red_spec]
        sed_probs = [0.5, 0.5]

        normed_sed_list = []
        norm_mag = np.random.uniform(21., 23.)
        for sed_obj in sed_list:
            sed_copy = deepcopy(sed_obj)
            f_norm = sed_copy.calcFluxNorm(norm_mag, 
                                           self.imsimBand)
            sed_copy.multiplyFluxNorm(f_norm)
            normed_sed_list.append(sed_copy)

        test_c = calcIG(trap_dict, normed_sed_list, sed_probs)

        colors, errors = test_c.calc_colors()

        np.testing.assert_equal(colors, np.zeros(np.shape(colors)))

        sky_fn = self.sky_spec.calcFluxNorm(21.2, self.imsimBand)
        self.sky_spec.multiplyFluxNorm(sky_fn)

        test_error = calcMagError_sed(test_c._sed_list[0],
                                      total_filt_dict['filter_0'],
                                      self.sky_spec,
                                      hardware_filt_dict['filter_0'],
                                      self.phot_params, 1.0)

        np.testing.assert_almost_equal(errors, [[test_error*np.sqrt(2)],
                                                [test_error*np.sqrt(2)]])

        trap_dict_2 = self.f.trap_filters([[420., 435., 465., 480.],
                                           [770., 785., 815., 830.],
                                           [970., 985., 1015., 1030.]])
        hardware_filt_dict_2, total_filt_dict_2 = \
            BandpassDict.addSystemBandpass(trap_dict_2)

        sky_fn2 = self.sky_spec.calcFluxNorm(21.2,
                                             total_filt_dict_2['filter_0'])
        self.sky_spec.multiplyFluxNorm(sky_fn2)
        sky_mags = total_filt_dict_2.magListForSed(self.sky_spec)

        sed_1 = Sed()
        sed_1.setSED(wavelen=np.linspace(200., 1500., 1301),
                     flambda=np.ones(1301))
        imsim_f_norm = sed_1.calcFluxNorm(21.2, self.imsimBand)
        sed_1.multiplyFluxNorm(imsim_f_norm)
        f_norm_1_0 = sed_1.calcFluxNorm(15.0, total_filt_dict_2['filter_0'])
        f_norm_1_1 = sed_1.calcFluxNorm(14.0, total_filt_dict_2['filter_1'])
        f_norm_1_2 = sed_1.calcFluxNorm(13.0, total_filt_dict_2['filter_2'])
        flambda_1 = sed_1.flambda
        flambda_1[:500] *= f_norm_1_0
        flambda_1[500:700] *= f_norm_1_1
        flambda_1[700:] *= f_norm_1_2
        sed_1.flambda = flambda_1

        test_c2 = calcIG(trap_dict_2, [sed_1], [1.0],
                         ref_filter=total_filt_dict_2['filter_0'])

        colors2, errors2, snr2, mags2, sky_m2 = test_c2.calc_colors(
                                                        return_all=True)

        np.testing.assert_almost_equal(sky_mags, sky_m2)

        np.testing.assert_almost_equal([[15.0, 14.0, 13.0]], mags2, 5)

        np.testing.assert_almost_equal([[1.0, 1.0]], colors2, 5)

    def test_calc_hyx(self):

        trap_dict = self.f.trap_filters([[740., 770., 830., 860.],
                                         [740., 770., 830., 860.]])
        sed_probs = [0.5, 0.5]
        test_c = calcIG(trap_dict, [self.red_spec, self.red_spec],
                        sed_probs)
        hy = test_c.calc_h(sed_probs)
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

        red_copy = deepcopy(self.red_spec)
        f_norm = red_copy.calcFluxNorm(23., self.imsimBand)
        red_copy.multiplyFluxNorm(f_norm)
        sed_list = [red_copy]*4

        test_c = calcIG(trap_dict, sed_list,
                        sed_probs)

        ig = test_c.calc_IG()
        self.assertAlmostEqual(ig, 0., delta=0.01)

        # At very high signal to noise information gain should be perfect
        trap_dict_2 = self.f.trap_filters([[340., 370., 430., 460.],
                                           [540., 570., 630., 660.],
                                           [740., 770., 830., 860.]])

        sed_probs_2 = [0.25, 0.25, 0.25, 0.25]

        sed_list = [deepcopy(self.red_spec), deepcopy(self.red_spec_z_1),
                    deepcopy(self.blue_spec), deepcopy(self.blue_spec_z_1)]

        for sed_obj in sed_list:
            f_norm = sed_obj.calcFluxNorm(10.0, self.imsimBand)
            sed_obj.multiplyFluxNorm(f_norm)

        test_c_2 = calcIG(trap_dict_2, [self.red_spec, self.red_spec_z_1,
                                        self.blue_spec, self.blue_spec_z_1],
                          sed_probs_2, sky_mag=20.0)
        ig_2 = test_c_2.calc_IG()
        self.assertAlmostEqual(ig_2, 2.0, delta=0.01)

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
