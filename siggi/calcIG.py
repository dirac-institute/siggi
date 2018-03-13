from __future__ import division

import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.spatial.distance import cdist
from scipy import stats
from . import Sed, Bandpass

__all__ = ["calcIG"]


class calcIG(object):

    """
    This class will take a set of SEDs and a set of filters
    and calculate the information gain.
    """

    def __init__(self, filter_dict, sed_list, sed_probs,
                 snr=5.):

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

        self.sed_probs = np.array(sed_probs)

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

        sed_probs = self.sed_probs

        h_sum = 0
        total_y = len(sed_probs)

        for py_i in sed_probs:
            h_sum += -(py_i)*np.log2(py_i)

        return h_sum

    def calc_hyx(self, colors, errors):

        sed_probs = self.sed_probs

        hyx_sum = 0

        num_seds, num_colors = np.shape(colors)

        rv = stats.multivariate_normal

        y_vals = []
        y_distances = []
        num_points = 1000
        x_total = np.zeros((num_seds*num_points, num_colors))

        for idx in range(num_seds):

            y_samples = rv.rvs(mean=colors[idx],
                               cov=np.diagflat(errors[idx]),
                               size=(num_points+1, num_colors))

            y_samples = np.sort(y_samples)
            y_dist = [y_samples[x+1] - y_samples[x] for x in range(num_points)]

            x_total[idx*num_points:(idx+1)*num_points] = \
                y_samples[1:].reshape(num_points, num_colors)

            y_vals.append(y_samples[1:])
            y_distances.append(y_dist)

        y_vals = np.array(y_vals)

        x_dens = np.zeros(len(x_total))

        for idx in range(num_seds):
            y_dens = sed_probs[idx]*rv.pdf(x_total, mean=colors[idx],
                                           cov=np.diagflat(errors[idx]))
            x_dens += y_dens

        for idx in range(num_seds):

            y_samp = x_total[idx*num_points:(idx+1)*num_points]
            y_dens = sed_probs[idx]*rv.pdf(y_samp, mean=colors[idx],
                                           cov=np.diagflat(errors[idx]))
            
            # norm_factor = ((errors[0][0]*10)**num_colors)/num_points

            hyx_i = np.nansum(y_distances[idx] * (y_dens * np.log2(y_dens /
                                             x_dens[idx*num_points:(idx+1) *
                                                    num_points])))

            # hyx_i = norm_factor * np.nansum((y_dens * np.log2(y_dens /
            #                                 x_dens[idx*num_points:(idx+1) *
            #                                        num_points])))

            hyx_sum += hyx_i

        return -1.*hyx_sum

    def calc_IG(self):

        colors, errors = self.calc_colors()
        hy_sum = self.calc_h()
        hyx_sum = self.calc_hyx(colors, errors)

        info_gain = hy_sum - hyx_sum

        return info_gain
