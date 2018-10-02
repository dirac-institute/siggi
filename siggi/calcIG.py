from __future__ import division

import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.special import gamma
from .mathUtils import integrationUtils
from . import Sed, Bandpass, BandpassDict, spectra
from .lsst_utils import calcMagError_sed, calcSNR_sed
from .lsst_utils import PhotometricParameters

__all__ = ["calcIG"]


class calcIG(integrationUtils):

    """
    This class will take a set of SEDs and a set of filters
    and calculate the information gain.
    """

    def __init__(self, filter_dict, sed_list, sed_probs, sed_mags=22.0,
                 sky_mag=19.0, ref_filter=None, phot_params=None,
                 fwhm_eff=1.0):

        self._sed_list = []

        if ref_filter is None:
            ref_filter = Bandpass()
            ref_filter.imsimBandpass()

        for sed_obj in sed_list:

            sed_copy = Sed()
            sed_copy.setSED(wavelen=sed_obj.wavelen,
                            flambda=sed_obj.flambda)
            sed_copy.resampleSED(wavelen_match=filter_dict.values()[0].wavelen)
            sed_copy.flambda[np.where(np.isnan(sed_copy.flambda))] = 0.

            f_norm = sed_copy.calcFluxNorm(sed_mags, ref_filter)
            sed_copy.multiplyFluxNorm(f_norm)
            self._sed_list.append(sed_copy)

        self._hardware_filt_dict, self._total_filt_dict = \
            BandpassDict.addSystemBandpass(filter_dict)

        self.sky_spec = spectra().get_dark_sky_spectrum()
        sky_fn = self.sky_spec.calcFluxNorm(sky_mag, ref_filter)
        self.sky_spec.multiplyFluxNorm(sky_fn)

        if phot_params is None:
            # Use Default LSST parameters
            self.phot_params = PhotometricParameters()
        else:
            self.phot_params = phot_params

        self.sed_probs = np.array(sed_probs)/np.sum(sed_probs)
        self.fwhm_eff = fwhm_eff

        return

    def calc_colors(self, return_all=False):

        """
        Calculate the colors and errors in color measurement
        of the SEDs in the given filters.
        """

        sed_colors = []
        color_errors = []
        snr_values = []
        sed_mag_list = []

        sky_mags = self._total_filt_dict.magListForSed(self.sky_spec)

        for sed_obj in self._sed_list:

            sed_mags = self._total_filt_dict.magListForSed(sed_obj)

            mag_errors = [calcMagError_sed(sed_obj, filt_tot,
                                           self.sky_spec, filt_hw,
                                           self.phot_params, self.fwhm_eff) for
                          filt_tot, filt_hw in zip(
                                self._total_filt_dict.values(),
                                self._hardware_filt_dict.values())]

            if np.isnan(sed_mags[0]):
                print(sed_mags)
            sed_colors.append([sed_mags[i] - sed_mags[i+1]
                               for i in range(len(sed_mags) - 1)])
            color_errors.append([np.sqrt(mag_errors[i]**2. +
                                         mag_errors[i+1]**2.) for i
                                 in range(len(mag_errors) - 1)])

            if return_all is True:
                snr_value = [calcSNR_sed(sed_obj, filt_tot,
                                         self.sky_spec, filt_hw,
                                         self.phot_params, self.fwhm_eff) for
                             filt_tot, filt_hw in zip(
                                self._total_filt_dict.values(),
                                self._hardware_filt_dict.values())]
                snr_values.append(snr_value)
                sed_mag_list.append(sed_mags)

        if return_all is True:
            return np.array(sed_colors), np.array(color_errors), snr_values,\
                   sed_mag_list, sky_mags

        return np.array(sed_colors), np.array(color_errors)

    def calc_h(self):

        """
        Calculates the total entropy of the set of SEDs and redshifts.
        """

        sed_probs = self.sed_probs

        h_sum = 0
        total_y = len(sed_probs)

        for py_i in sed_probs:
            h_sum += -(py_i)*np.log2(py_i)

        return h_sum

    def calc_hyx(self, colors, errors, rand_state=None):

        """
        Calculate the conditional entropy.
        """

        if rand_state is None:
            rand_state = np.random.RandomState()

        sed_probs = self.sed_probs

        hyx_sum = 0

        num_seds, num_colors = np.shape(colors)

        rv = rand_state.multivariate_normal

        y_vals = []
        y_distances = []
        num_points = 10000
        x_total = np.zeros((num_seds*num_points, num_colors))

        for idx in range(num_seds):

            y_samples = rv(mean=colors[idx],
                           cov=np.diagflat(errors[idx]**2.),
                           size=num_points)

            y_samples = y_samples.reshape(num_points, num_colors)

            inv_cov = np.linalg.inv(np.diagflat(errors[idx]**2.))

            y_dist = cdist(y_samples, [colors[idx]], metric='mahalanobis',
                           VI=inv_cov).flatten()
            y_sort = np.argsort(y_dist)
            y_dist = y_dist[y_sort]
            y_samples = y_samples[y_sort]

            x_total[idx*num_points:(idx+1)*num_points] = \
                y_samples

            y_vals.append(y_samples)
            y_distances.append(y_dist)

        y_vals = np.array(y_vals)

        x_dens = np.zeros(len(x_total))

        for idx in range(num_seds):
            pdf_dist = stats.multivariate_normal
            y_dens = sed_probs[idx] * \
                pdf_dist.pdf(x_total, mean=colors[idx],
                             cov=np.diagflat(errors[idx])**2.)
            x_dens += y_dens

        for idx in range(num_seds):

            y_samp = x_total[idx*num_points:(idx+1)*num_points]
            y_dens = sed_probs[idx] * \
                pdf_dist.pdf(y_samp, mean=colors[idx],
                             cov=np.diagflat(errors[idx])**2.)

            norm_factor = self.calc_integral_scaling(y_distances[idx],
                                                     num_colors, errors[idx])

            hyx_i = np.nansum(norm_factor * (y_dens * np.log2(y_dens /
                                             x_dens[idx*num_points:(idx+1) *
                                                    num_points])))
            hyx_sum += hyx_i

        return -1.*hyx_sum

    def calc_IG(self, rand_state=None):

        """
        Use the entropy and conditional entropy methods and then subtract
        to get information gain.
        """

        colors, errors = self.calc_colors()
        hy_sum = self.calc_h()
        hyx_sum = self.calc_hyx(colors, errors, rand_state)

        info_gain = hy_sum - hyx_sum

        return info_gain
