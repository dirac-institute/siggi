from __future__ import division

import numpy as np
from sklearn.neighbors.kde import KernelDensity
from . import Sed, Bandpass
__all__ = ["calcIG"]


class calcIG(object):

    """
    This class will take a set of SEDs and a set of filters
    and calculate the information gain.
    """

    def __init__(self, filter_dict, sed_list, prior_func,
                 y_min, y_max, y_step):

        self._filter_dict = filter_dict
        self._sed_list = []
        for sed_obj in sed_list:
            sed_copy = Sed()
            sed_copy.setSED(wavelen=sed_obj.wavelen,
                            flambda=sed_obj.flambda)
            sed_copy.resampleSED(wavelen_match=filter_dict.values()[0].wavelen)
            sed_copy.flambda[np.where(np.isnan(sed_copy.flambda))] = 0.
            imsimBand = Bandpass()
            imsimBand.imsimBandpass()
            f_norm = sed_copy.calcFluxNorm(-10.0, imsimBand)
            sed_copy.multiplyFluxNorm(f_norm)
            self._sed_list.append(sed_copy)
        self._flat_sed = Sed()
        self._flat_sed.setFlatSED(wavelen_min=99., wavelen_max=1500.)
        f_norm = self._flat_sed.calcFluxNorm(-15.0, imsimBand)
        self._flat_sed.multiplyFluxNorm(f_norm)
        self._flat_sed.resampleSED(wavelen_match=self._sed_list[0].wavelen)

        self.prior = prior_func

        self.y_min = y_min
        self.y_max = y_max
        self.y_step = y_step

        return

    def draw_colors(self, num_points, random_seed=17):

        np.random.seed(random_seed)

        mags_list = []
        color_list = []
        true_sed_list = []

        sed_on = 0
        sed_copy_1 = Sed()

        y_range = np.arange(self.y_min,
                            self.y_max + 0.5*self.y_step,
                            self.y_step)
        sed_probs = self.prior(y_range)
        sed_choices = np.random.choice(y_range, p=sed_probs, 
                                       size=num_points)
        bins = np.arange(self.y_min - 0.5*self.y_step,
                         self.y_max + 1.5*self.y_step,
                         self.y_step)
        bin_counts, _ = np.histogram(sed_choices, bins=bins)

        for sed_obj in self._sed_list:

            sed_copy_1.setSED(wavelen=sed_obj.wavelen, 
                              flambda=sed_obj.flambda)

            for i in range(bin_counts[sed_on]):
                #  TODO: Add realistic errors

                err_val = np.random.normal(loc=0.0, 
                                           scale=np.sqrt(sed_obj.flambda))
                sed_copy_1.flambda = sed_obj.flambda + .4*err_val
                sed_copy_1.flambda += self._flat_sed.flambda
                sed_copy_1.flambda[sed_copy_1.flambda <= 0.] = 0.
                flux_with_errors = self._filter_dict.fluxListForSed(sed_copy_1)
                mags = [sed_copy_1.magFromFlux(f) for f in flux_with_errors]
                mags_list.append(mags)
                color_list.append([mags[i] - mags[i-1] 
                                   for i in range(len(mags)-1)])
                true_sed_list.append(sed_on)
            sed_on += 1

        return np.array(mags_list), np.array(color_list), \
            np.array(true_sed_list)

    def calc_density(self, colors, truth_vals, bandwidth=None):

        if bandwidth is None:
            bandwidth = 0.05  # TODO: Fix this to adjust based on color range

        kde_total = KernelDensity(kernel='gaussian',
                                  bandwidth=bandwidth)
        kde_total.fit(colors)

        kde_list = []

        for y_i in np.unique(truth_vals):
            kde_1 = KernelDensity(kernel='gaussian',
                                  bandwidth=bandwidth)
            kde_1.fit(colors[np.where(truth_vals == y_i)])
            kde_list.append(kde_1)

        kde_lists = [kde_total, kde_list]

        return kde_lists

    def calc_h(self):

        y_range = np.arange(self.y_min + 0.5*self.y_step,
                            self.y_max + 0.75*self.y_step,
                            self.y_step)
        sed_probs = self.prior(y_range)

        h_sum = 0
        total_y = len(sed_probs)

        for py_i in sed_probs:
            h_sum += -(py_i)*np.log2(py_i)

        return h_sum

    def calc_hyx(self, kde_lists):

        hyx_sum = 0
        total_kde, y_kde_list = kde_lists
        num_y = len(y_kde_list)
        dens_step = 0.01
        dens_range = np.arange(-2., 2., dens_step).reshape(-1,1)
        total_dens = np.exp(total_kde.score_samples(dens_range))

        for idx in range(num_y):
            y_dens = np.exp(y_kde_list[idx].score_samples(dens_range))
            for x_val, yx_val in zip(total_dens, y_dens):
                p_x_y = yx_val*dens_step/num_y
                p_x = x_val*dens_step
                if ((p_x_y > 0.) & (p_x > 0.)):
                    hyx_sum += p_x_y*np.log2(p_x_y/p_x)

        return -1.*hyx_sum

    def calc_IG(self):

        mags, colors, truth = self.draw_colors(150*len(self._sed_list))
        k = self.calc_density(colors, truth)
        hy_sum = self.calc_h()
        hyx_sum = self.calc_hyx(k)

        info_gain = hy_sum - hyx_sum

        return info_gain
