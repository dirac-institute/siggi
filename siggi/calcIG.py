from __future__ import division

import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy import stats
from . import Sed, Bandpass, PhotometricParameters
__all__ = ["calcIG"]


class calcIG(object):

    """
    This class will take a set of SEDs and a set of filters
    and calculate the information gain.
    """

    def __init__(self, filter_dict, sed_list, prior_func,
                 y_min, y_max, y_step, snr=5.):

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
            f_norm = sed_copy.calcFluxNorm(18.0, imsimBand)
            sed_copy.multiplyFluxNorm(f_norm)
            self._sed_list.append(sed_copy)
        self._flat_sed = Sed()
        self._flat_sed.setFlatSED(wavelen_min=99., wavelen_max=1500.)
        f_norm = self._flat_sed.calcFluxNorm(18.0, imsimBand)
        self._flat_sed.multiplyFluxNorm(f_norm)
        self._flat_sed.resampleSED(wavelen_match=self._sed_list[0].wavelen)

        self.prior = prior_func

        self.y_min = y_min
        self.y_max = y_max
        self.y_step = y_step

        self.snr = snr

        return

    def calc_colors(self):

        sed_colors = []
        color_errors = []

        mag_error = 1.0/self.snr
        color_error = np.sqrt(2)*mag_error

        for sed_obj in self._sed_list:

            sed_mags = self._filter_dict.magListForSed(sed_obj)
            if np.isnan(sed_mags[0]):
                print(sed_mags)
            sed_colors.append([sed_mags[i] - sed_mags[i-1]
                               for i in range(len(sed_mags) - 1)])
            color_errors.append([color_error for i in range(len(sed_mags)-1)])

        return np.array(sed_colors), np.array(color_errors)

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

    def calc_hyx(self, colors, errors):

        y_range = np.arange(self.y_min,
                            self.y_max + 0.5*self.y_step,
                            self.y_step)
        sed_probs = self.prior(y_range)

        hyx_sum = 0
        c_max = np.max(colors) + 5*np.max(errors)
        c_min = np.min(colors) - 5*np.max(errors)
        dens_step = (c_max - c_min) / 250.
        num_colors = np.shape(colors)[1]

        dens_range = np.arange(c_min, c_max, dens_step)
        d_range = [dens_range for dim in range(num_colors)]
        dens_range = np.meshgrid(*d_range)
        dens_range = np.transpose(dens_range)
        dens_range = np.reshape(dens_range, (len(dens_range)**num_colors,
                                             num_colors))

        rv = stats.multivariate_normal

        x_total = np.zeros(len(dens_range))
        y_vals = []

        for idx in range(len(y_range)):
            y_dens = sed_probs[idx]*rv.pdf(dens_range, mean=colors[idx],
                                           cov=np.diagflat(errors[idx]))
            x_total += y_dens
            y_vals.append(y_dens)

        y_vals = np.array(y_vals)

        for idx in range(len(dens_range)):
            hyx_i = np.nansum(dens_step*y_vals[:, idx]*np.log2(y_vals[:, idx] /
                                                               x_total[idx]))
            hyx_sum += hyx_i

        return -1.*hyx_sum

    def calc_IG(self):

        colors, errors = self.calc_colors()
        hy_sum = self.calc_h()
        hyx_sum = self.calc_hyx(colors, errors)

        info_gain = hy_sum - hyx_sum

        return info_gain
