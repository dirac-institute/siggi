import unittest
import os
import numpy as np
from scipy.stats import norm, entropy, multivariate_normal
from siggi import spectra, calcIG
from siggi.filters import filterFactory
from siggi import Sed, Bandpass, BandpassDict
from siggi.lsst_utils import PhotometricParameters, calcMagError_sed
from copy import deepcopy


class testCalcIG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.f = filterFactory.create_filter_object('trap')
        cls.f.set_wavelen_grid()
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

        cls.frozen_dict = BandpassDict.loadTotalBandpassesFromFiles(
                        bandpassNames=['i'],
                        bandpassDir=os.path.join(os.path.dirname(__file__),
                                                 '../siggi/data',
                                                 'lsst_baseline_throughputs'))

        phot_params = {}
        for idx in range(6):
            phot_params['filter_%i' % idx] = \
                PhotometricParameters(nexp=2, sigmaSys=0.005,
                                      bandpass='filter_%i' % idx)

        cls.phot_params = phot_params

    def test_calc_colors(self):

        trap_dict = self.f.create_filter_dict_from_corners(
            [[740., 770., 830., 860.], [740., 770., 830., 860.]]
        )
        hardware_filt_dict, total_filt_dict = \
            BandpassDict.addSystemBandpass(trap_dict)

        sed_list = [self.red_spec, self.red_spec]
        sed_probs = [0.5, 0.5]

        normed_sed_list = []
        norm_mag = np.random.uniform(21., 23.)
        sed_labels = []
        i = 0
        for sed_obj in sed_list:
            sed_copy = deepcopy(sed_obj)
            f_norm = sed_copy.calcFluxNorm(norm_mag,
                                           self.imsimBand)
            sed_copy.multiplyFluxNorm(f_norm)
            normed_sed_list.append(sed_copy)
            sed_labels.append(i)
            i += 1

        test_c = calcIG(trap_dict, [normed_sed_list], [1.], [1.],
                        n_pts=40000,
                        sky_mag=21.2, phot_params=self.phot_params)

        colors, errors = test_c.calc_colors(normed_sed_list)

        np.testing.assert_equal(colors, np.zeros(np.shape(colors)))

        sky_fn = self.sky_spec.calcFluxNorm(21.2, self.frozen_dict['i'])
        self.sky_spec.multiplyFluxNorm(sky_fn)

        test_error = calcMagError_sed(test_c.sed_list[0][0],
                                      total_filt_dict['filter_0'],
                                      self.sky_spec,
                                      hardware_filt_dict['filter_0'],
                                      self.phot_params['filter_0'], 0.8)

        np.testing.assert_almost_equal(errors, [[test_error*np.sqrt(2)],
                                                [test_error*np.sqrt(2)]])

        trap_dict_2 = self.f.create_filter_dict_from_corners(
            [[420., 435., 465., 480.],
             [770., 785., 815., 830.],
             [970., 985., 1015., 1030.]]
        )
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

        test_c2 = calcIG(trap_dict_2, [[sed_1]], [1.0], [1.], [1.],
                         sky_mag=21.2,
                         ref_filter=total_filt_dict_2['filter_0'],
                         phot_params=self.phot_params)

        colors2, errors2, snr2, mags2, sky_m2 = test_c2.calc_colors(
                                                        [sed_1],
                                                        return_all=True)

        np.testing.assert_almost_equal(sky_mags, sky_m2)

        np.testing.assert_almost_equal([[15.0, 14.0, 13.0]], mags2, 5)

        np.testing.assert_almost_equal([[1.0, 1.0]], colors2, 5)

    # def test_calc_hyx(self):

    #     trap_dict = self.f.trap_filters([[740., 770., 830., 860.],
    #                                      [740., 770., 830., 860.]])
    #     sed_probs = [0.5, 0.5]
    #     test_c = calcIG(trap_dict, [[self.red_spec], [self.red_spec]],
    #                     sed_probs, [0.25, 0.75])
    #     hy = test_c.calc_h(sed_probs)
    #     colors, errors = test_c.calc_colors([])
    #     hyx = test_c.calc_hyx(colors, errors)

    #     self.assertAlmostEqual(hy, hyx, delta=0.01)

    def test_calcIG(self):

        # With same filter should be zero information gain. All colors = 0
        trap_dict = self.f.create_filter_dict_from_corners(
            [[740., 770., 830., 860.], [740., 770., 830., 860.],
             [740., 770., 830., 860.], [740., 770., 830., 860.],
             [740., 770., 830., 860.], [740., 770., 830., 860.]]
        )
        sed_probs = [0.0, 0.25, 0.25, 0.25, 0.25]

        red_copy = deepcopy(self.red_spec)
        f_norm = red_copy.calcFluxNorm(23., self.imsimBand)
        red_copy.multiplyFluxNorm(f_norm)
        sed_list = [[red_copy]]*4

        test_c = calcIG(trap_dict, sed_list,
                        sed_probs, [0.0, 1, 2, 3, 4], n_pts=40000)

        ig = test_c.calc_IG()
        self.assertAlmostEqual(ig, 0., delta=0.01)

        # At very high signal to noise information gain should be perfect
        trap_dict_2 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.],
             [540., 570., 630., 660.],
             [740., 770., 830., 860.]]
        )
        sed_probs_2 = [0., 0.25, 0.25, 0.25, 0.25]

        sed_list = [[deepcopy(self.red_spec)], [deepcopy(self.red_spec_z_1)],
                    [deepcopy(self.blue_spec)], [deepcopy(self.blue_spec_z_1)]]

        for sed_item in sed_list:
            sed_obj = sed_item[0]
            f_norm = sed_obj.calcFluxNorm(6.0, self.imsimBand)
            sed_obj.multiplyFluxNorm(f_norm)

        test_c_2 = calcIG(trap_dict_2, sed_list,
                          sed_probs_2, [0, 1, 2, 3, 4],
                          n_pts=40000, sky_mag=24.0)
        ig_2 = test_c_2.calc_IG()
        self.assertAlmostEqual(ig_2, 2.0, delta=0.01)

        # With one dimension should equal 1-d KL divergence
        trap_dict_3 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.], [440., 470., 530., 560.]]
        )
        sed_probs_3 = [0., 0.5, 0.5]

        red_copy = deepcopy(self.red_spec)
        f_norm = red_copy.calcFluxNorm(25.5, self.imsimBand)
        red_copy.multiplyFluxNorm(f_norm)
        red_copy_2 = deepcopy(self.blue_spec)
        f_norm_2 = red_copy_2.calcFluxNorm(25., self.imsimBand)
        red_copy_2.multiplyFluxNorm(f_norm_2)

        sed_list_3 = [[red_copy], [red_copy_2]]

        test_c_3 = calcIG(trap_dict_3, sed_list_3,
                          sed_probs_3, [0, 1, 2], n_pts=40000)

        ig = test_c_3.calc_IG(rand_state=np.random.RandomState(37))
        colors = []
        errors = []
        for sed_item in sed_list_3:
            c, e = test_c_3.calc_colors(sed_item)
            colors.append(c[0])
            errors.append(e[0])

        print(ig, errors)

        rv_1 = norm(loc=colors[0], scale=errors[0])
        p1 = rv_1.pdf(np.arange(-2, 3, 0.001))
        rv_2 = norm(loc=colors[1], scale=errors[1])
        p2 = rv_2.pdf(np.arange(-2., 3, 0.001))
        p3 = p1 + p2

        print(.5*entropy(p1, p3, base=2) + .5*entropy(p2, p3, base=2))

        kl_div = .5*entropy(p1, p3, base=2) + .5*entropy(p2, p3, base=2)

        self.assertAlmostEqual(ig, kl_div, delta=0.01)

        # # With one dimension should equal 1-d KL divergence
        trap_dict_4 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.], [440., 470., 530., 560.]]
        )
        sed_probs_4 = [0.0, 0.25, 0.25, 0.25, 0.25]

        red_copy_3 = deepcopy(self.red_spec)
        f_norm_3 = red_copy.calcFluxNorm(24.5, self.imsimBand)
        red_copy_3.multiplyFluxNorm(f_norm_3)
        red_copy_4 = deepcopy(self.blue_spec)
        f_norm_4 = red_copy_4.calcFluxNorm(25.2, self.imsimBand)
        red_copy_4.multiplyFluxNorm(f_norm_4)

        sed_list_4 = [[red_copy], [red_copy_2],
                      [red_copy_3], [red_copy_4]]

        test_c_4 = calcIG(trap_dict_4, sed_list_4,
                          sed_probs_4, [0, 1, 2, 3, 4], n_pts=40000)

        ig = test_c_4.calc_IG(rand_state=np.random.RandomState(17))
        colors = []
        errors = []
        for sed_item in sed_list_4:
            c, e = test_c_4.calc_colors(sed_item)
            colors.append(c[0])
            errors.append(e[0])

        print(ig, errors)

        rv_1 = norm(loc=colors[0], scale=errors[0])
        p1 = rv_1.pdf(np.arange(-2, 3, 0.001))
        rv_2 = norm(loc=colors[1], scale=errors[1])
        p2 = rv_2.pdf(np.arange(-2., 3, 0.001))
        rv_3 = norm(loc=colors[2], scale=errors[2])
        p3 = rv_3.pdf(np.arange(-2, 3, 0.001))
        rv_4 = norm(loc=colors[3], scale=errors[3])
        p4 = rv_4.pdf(np.arange(-2., 3, 0.001))
        p5 = p1 + p2 + p3 + p4

        print(.25*entropy(p1, p5, base=2) + .25*entropy(p2, p5, base=2) +
              .25*entropy(p3, p5, base=2) + .25*entropy(p4, p5, base=2))

        kl_div = .25*entropy(p1, p5, base=2) + .25*entropy(p2, p5, base=2) + \
            .25*entropy(p3, p5, base=2) + .25*entropy(p4, p5, base=2)

        self.assertAlmostEqual(ig, kl_div, delta=0.01)

        # Colors of 0 don't mean 0 IG if errors are different.
        trap_dict_5 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.], [340., 370., 430., 460.]]
        )
        sed_probs_5 = [0.0, 0.5, 0.5]

        sed_list_5 = [[red_copy], [red_copy_2]]

        test_c_5 = calcIG(trap_dict_5, sed_list_5,
                          sed_probs_5, [0, 1, 2], n_pts=40000)

        ig = test_c_5.calc_IG(rand_state=np.random.RandomState(17))
        colors = []
        errors = []
        for sed_item in sed_list_5:
            c, e = test_c_5.calc_colors(sed_item)
            colors.append(c[0])
            errors.append(e[0])

        print(ig)

        rv_1 = norm(loc=colors[0], scale=errors[0])
        p1 = rv_1.pdf(np.arange(-2, 3, 0.001))
        rv_2 = norm(loc=colors[1], scale=errors[1])
        p2 = rv_2.pdf(np.arange(-2., 3, 0.001))
        p3 = p1 + p2

        print(.5*entropy(p1, p3, base=2) + .5*entropy(p2, p3, base=2))

        kl_div = .5*entropy(p1, p3, base=2) + .5*entropy(p2, p3, base=2)

        self.assertAlmostEqual(ig, kl_div, delta=0.01)

        # Test 2-dimensional example
        trap_dict_6 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.],
             [340., 370., 430., 460.],
             [440., 470., 530., 560.]]
        )
        sed_probs_6 = [0.0, 0.5, 0.5]

        sed_list_6 = [[red_copy], [red_copy_2]]

        test_c_6 = calcIG(trap_dict_6, sed_list_6,
                          sed_probs_6, [0, 1, 2], n_pts=40000)

        ig = test_c_6.calc_IG(rand_state=np.random.RandomState(17))
        colors = []
        errors = []
        for sed_item in sed_list_6:
            c, e = test_c_6.calc_colors(sed_item)
            colors.append(c[0])
            errors.append(e[0])

        print(ig)

        x, y = np.mgrid[-2:4:0.01, -2:4:0.01]
        pos = np.dstack((x, y))
        p1 = multivariate_normal.pdf(pos,
                                     mean=colors[0],
                                     cov=np.diag(errors[0]**2.))
        p2 = multivariate_normal.pdf(pos,
                                     mean=colors[1],
                                     cov=np.diag(errors[1]**2.))
        p1 = p1.flatten()
        p2 = p2.flatten()
        p3 = p1 + p2

        print(.5*entropy(p1, p3, base=2) + .5*entropy(p2, p3, base=2))

        kl_div = .5*entropy(p1, p3, base=2) + .5*entropy(p2, p3, base=2)

        #self.assertAlmostEqual(ig, kl_div, delta=0.01)

        # Test > 2 values

        trap_dict_6 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.],
             [340., 370., 430., 460.],
             [440., 470., 530., 560.]]
        )
        sed_probs_6 = [0, 1./3., 1./3., 1./3.]

        sed_list_6 = [[red_copy], [red_copy_2], [red_copy_3]]

        test_c_6 = calcIG(trap_dict_6, sed_list_6,
                          sed_probs_6, [0, 1, 2, 3], n_pts=250000)

        ig = test_c_6.calc_IG(rand_state=np.random.RandomState(17))
        colors = []
        errors = []
        for sed_item in sed_list_6:
            c, e = test_c_6.calc_colors(sed_item)
            colors.append(c[0])
            errors.append(e[0])

        print(ig)

        x, y = np.mgrid[-2:4:0.01, -2:4:0.01]
        pos = np.dstack((x, y))
        p1 = multivariate_normal.pdf(pos,
                                     mean=colors[0],
                                     cov=np.diag(errors[0]**2.))
        p2 = multivariate_normal.pdf(pos,
                                     mean=colors[1],
                                     cov=np.diag(errors[1]**2.))
        p3 = multivariate_normal.pdf(pos,
                                     mean=colors[2],
                                     cov=np.diag(errors[2]**2.))
        p1 = p1.flatten()
        p2 = p2.flatten()
        p3 = p3.flatten()
        p4 = p1 + p2 + p3

        print((1./3.)*entropy(p1, p4, base=2) +
              (1./3.)*entropy(p2, p4, base=2) +
              (1./3.)*entropy(p3, p4, base=2))

        kl_div = (1./3.)*entropy(p1, p4, base=2) + \
            (1./3.)*entropy(p2, p4, base=2) + \
            (1./3.)*entropy(p3, p4, base=2)

        self.assertAlmostEqual(ig, kl_div, delta=0.01)

        # Test > 2 values

        trap_dict_6 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.],
             [340., 370., 430., 460.],
             [440., 470., 530., 560.]]
        )
        sed_probs_6 = [0, .5, .5]

        red_copy = deepcopy(self.red_spec)
        red_copy.redshiftSED(0.1)
        f_norm = red_copy.calcFluxNorm(25., self.imsimBand)
        red_copy.multiplyFluxNorm(f_norm)
        red_copy_2 = deepcopy(self.blue_spec)
        red_copy_2.redshiftSED(0.1)
        f_norm_2 = red_copy_2.calcFluxNorm(25., self.imsimBand)
        red_copy_2.multiplyFluxNorm(f_norm_2)
        red_copy_3 = deepcopy(self.red_spec)
        f_norm = red_copy_3.calcFluxNorm(24.2, self.imsimBand)
        red_copy_3.multiplyFluxNorm(f_norm)
        red_copy_4 = deepcopy(self.blue_spec)
        f_norm_2 = red_copy_4.calcFluxNorm(24.2, self.imsimBand)
        red_copy_4.multiplyFluxNorm(f_norm_2)

        sed_list_6 = [[red_copy, red_copy_3],
                      [red_copy_2, red_copy_4]]

        test_c_6 = calcIG(trap_dict_6, sed_list_6,
                          sed_probs_6, [0, 1, 2], n_pts=250000)

        ig = test_c_6.calc_IG(rand_state=np.random.RandomState(17))
        colors = []
        errors = []
        for sed_item in sed_list_6:
            c, e = test_c_6.calc_colors(sed_item)
            colors.append(c)
            errors.append(e)

        print(ig)

        x, y = np.mgrid[-2:3:0.005, -2:3:0.005]
        pos = np.dstack((x, y))
        p1 = multivariate_normal.pdf(pos,
                                     mean=colors[0][0],
                                     cov=np.diag(errors[0][0]**2.))
        p2 = multivariate_normal.pdf(pos,
                                     mean=colors[1][0],
                                     cov=np.diag(errors[1][0]**2.))
        p3 = multivariate_normal.pdf(pos,
                                     mean=colors[0][1],
                                     cov=np.diag(errors[0][1]**2.))
        p4 = multivariate_normal.pdf(pos,
                                     mean=colors[1][1],
                                     cov=np.diag(errors[1][1]**2.))
        p1 = p1.flatten()
        p2 = p2.flatten()
        p3 = p3.flatten()
        p4 = p4.flatten()
        p5 = p1 + p2 + p3 + p4

        p1p3_normed = (p1+p3)/np.sum(p1+p3)
        p2p4_normed = (p2+p4)/np.sum(p2+p4)
        p5_normed = p5/np.sum(p5)

        p1_normed = p1/np.sum(p1)
        p2_normed = p2/np.sum(p2)
        p3_normed = p3/np.sum(p3)
        p4_normed = p4/np.sum(p4)

        kl_1 = np.sum(p1p3_normed * np.log2(p1p3_normed/p5_normed))
        kl_2 = np.sum(p2p4_normed * np.log2(p2p4_normed/p5_normed))

        print(0.5*kl_1 + 0.5*kl_2)

        kl_div = 0.5*kl_1 + 0.5*kl_2

        self.assertAlmostEqual(ig, kl_div, delta=0.01)


        # Test > 2 values and non-equal prior

        trap_dict_6 = self.f.create_filter_dict_from_corners(
            [[340., 370., 430., 460.],
             [340., 370., 430., 460.],
             [440., 470., 530., 560.]]
        )
        sed_probs_6 = [0, .5, 2.0]

        red_copy = deepcopy(self.red_spec)
        red_copy.redshiftSED(0.1)
        f_norm = red_copy.calcFluxNorm(25., self.imsimBand)
        red_copy.multiplyFluxNorm(f_norm)
        red_copy_2 = deepcopy(self.blue_spec)
        red_copy_2.redshiftSED(0.1)
        f_norm_2 = red_copy_2.calcFluxNorm(25., self.imsimBand)
        red_copy_2.multiplyFluxNorm(f_norm_2)
        red_copy_3 = deepcopy(self.red_spec)
        f_norm = red_copy_3.calcFluxNorm(24.2, self.imsimBand)
        red_copy_3.multiplyFluxNorm(f_norm)
        red_copy_4 = deepcopy(self.blue_spec)
        f_norm_2 = red_copy_4.calcFluxNorm(24.2, self.imsimBand)
        red_copy_4.multiplyFluxNorm(f_norm_2)

        sed_list_6 = [[red_copy, red_copy_3],
                      [red_copy_2, red_copy_4]]

        test_c_6 = calcIG(trap_dict_6, sed_list_6,
                          sed_probs_6, [0, 1, 2], n_pts=250000)

        ig = test_c_6.calc_IG(rand_state=np.random.RandomState(17))
        colors = []
        errors = []
        for sed_item in sed_list_6:
            c, e = test_c_6.calc_colors(sed_item)
            colors.append(c)
            errors.append(e)

        print(ig)

        x, y = np.mgrid[-2:3:0.005, -2:3:0.005]
        pos = np.dstack((x, y))
        p1 = multivariate_normal.pdf(pos,
                                     mean=colors[0][0],
                                     cov=np.diag(errors[0][0]**2.))
        p2 = multivariate_normal.pdf(pos,
                                     mean=colors[1][0],
                                     cov=np.diag(errors[1][0]**2.))
        p3 = multivariate_normal.pdf(pos,
                                     mean=colors[0][1],
                                     cov=np.diag(errors[0][1]**2.))
        p4 = multivariate_normal.pdf(pos,
                                     mean=colors[1][1],
                                     cov=np.diag(errors[1][1]**2.))
        p1 = .1*p1.flatten()
        p2 = .4*p2.flatten()
        p3 = .1*p3.flatten()
        p4 = .4*p4.flatten()
        p5 = (p1 + p2) + (p3 + p4)

        p1p3_normed = (p1+p3)/np.sum(p1+p3)
        p2p4_normed = (p2+p4)/np.sum(p2+p4)
        p5_normed = p5/np.sum(p5)

        p1_normed = p1/np.sum(p1)
        p2_normed = p2/np.sum(p2)
        p3_normed = p3/np.sum(p3)
        p4_normed = p4/np.sum(p4)

        kl_1 = np.sum(p1p3_normed * np.log2(p1p3_normed/p5_normed))
        kl_2 = np.sum(p2p4_normed * np.log2(p2p4_normed/p5_normed))

        kl_div = 0.2*kl_1 + 0.8*kl_2

        self.assertAlmostEqual(ig, kl_div, delta=0.01)

    @classmethod
    def tearDownClass(cls):
        return


if __name__ == '__main__':
    unittest.main()
