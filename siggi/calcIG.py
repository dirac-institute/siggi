from __future__ import division

import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist
from scipy import stats
from .mathUtils import mathUtils
from . import Sed, Bandpass, BandpassDict, spectra
from .lsst_utils import calcMagError_sed, calcSNR_sed
from .lsst_utils import PhotometricParameters
from copy import deepcopy
from sklearn.neighbors import BallTree, KernelDensity, NearestNeighbors

__all__ = ["calcIG"]


class calcIG(mathUtils):

    """
    This class will take a set of SEDs and a set of filters
    and calculate the information gain.

    Inputs
    ------

    filter_dict, dict

        Dictionary with the bandpass objects for the filters to test.

    sed_list, list

        List with the template SEDs at the different redshifts and already
        normalized in flux

    sed_probs, list

        List with the probabilities for each SED at the given redshift
        in the same order as sed_list
    """

    def __init__(self, filter_dict, sed_list, y_probs, y_vals,
                 n_pts=150000,
                 sky_mag=20.47, ref_filter=None, phot_params=None,
                 fwhm_eff=0.8):

        if ref_filter is None:
            ref_filter = Bandpass()
            ref_filter.imsimBandpass()

        self.sed_list = sed_list
        self.n_pts = n_pts

        self._hardware_filt_dict, self._total_filt_dict = \
            BandpassDict.addSystemBandpass(filter_dict)

        self.sky_spec = spectra().get_dark_sky_spectrum()
        sky_fn = self.sky_spec.calcFluxNorm(sky_mag, ref_filter)
        self.sky_spec.multiplyFluxNorm(sky_fn)

        if phot_params is None:
            # Use Default LSST parameters + Same for new filters
            self.phot_params = {}
            for filt_name in filter_dict.keys():
                self.phot_params[filt_name] = PhotometricParameters(
                    bandpass=filt_name)
        else:
            self.phot_params = phot_params

        self.y_probs = y_probs
        self.y_vals = y_vals
        self.fwhm_eff = fwhm_eff

        self.n_colors = len(self._total_filt_dict.keys()) - 1
        self.n_seds = len(sed_list[0])
        self.y_steps = len(y_vals) - 1

        return

    def calc_colors(self, sed_list, return_all=False):

        """
        Calculate the colors and errors in color measurement
        of the SEDs in the given filters.
        """

        sed_colors = []
        color_errors = []
        snr_values = []
        sed_mag_list = []

        sky_mags = self._total_filt_dict.magListForSed(self.sky_spec)

        for sed_item in sed_list:

            sed_obj = deepcopy(sed_item)

            sed_mags = self._total_filt_dict.magListForSed(sed_obj)

            mag_errors = [calcMagError_sed(sed_obj, filt_tot,
                                           self.sky_spec, filt_hw,
                                           self.phot_params[filt_name],
                                           self.fwhm_eff) for
                          filt_tot, filt_hw, filt_name in zip(
                                self._total_filt_dict.values(),
                                self._hardware_filt_dict.values(),
                                self._total_filt_dict.keys())]

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
                                         self.phot_params[filt_name],
                                         self.fwhm_eff) for
                             filt_tot, filt_hw, filt_name in zip(
                                self._total_filt_dict.values(),
                                self._hardware_filt_dict.values(),
                                self._total_filt_dict.keys())]
                snr_values.append(snr_value)
                sed_mag_list.append(sed_mags)

        if return_all is True:
            return np.array(sed_colors), np.array(color_errors), snr_values,\
                   sed_mag_list, sky_mags

        return np.array(sed_colors), np.array(color_errors)

    def nearest_neighbors_density(self, x, radius, normalize=True):

        x = np.asarray(x)

        bt = BallTree(x)

        counts = bt.query_radius(x, radius, count_only=True)
        counts = counts.astype(float)

        if normalize:
            counts /= x.shape[0] * (radius ** x.shape[1])

        return counts.astype(float)

    def knn_density(self, x, n_neighbors):

        x = np.asarray(x)

        if n_neighbors > len(x)*.9:
            n_neighbors = int(len(x)*.9)

        bt = NearestNeighbors(n_neighbors=n_neighbors).fit(x)

        dist, indices = bt.kneighbors()
        dist = dist.astype(float)

        return dist

    def kernel_estimate_density(self, x, h):

        x = np.asarray(x)

        kde = KernelDensity(kernel='epanechnikov', bandwidth=h, rtol=1e-4).fit(x)

        dens_est = kde.score_samples(x)

        return dens_est.astype(float)

    def calc_gain(self, rand_state=None):

        """
        Calculate the information gain.
        """

        if rand_state is None:
            rand_state = np.random.RandomState()

        # Set up draws of the prior distribution
        y_probs = deepcopy(self.y_probs)
        # Make sure probability function starts at (0., 0.)
        if y_probs[0] != 0.:
            raise ValueError('Make sure probability function' +
                             ' starts at (0., 0.)')

        y_probs /= np.sum(y_probs)
        y_cum = y_probs.cumsum()
        n_pts = self.n_pts

        if len(self.y_vals) > 3:
            fy = interpolate.splrep(y_cum, self.y_vals)
        else:
            fy = interpolate.splrep(y_cum, self.y_vals,
                                    k=len(self.y_vals)-1)
        samp_y = rand_state.uniform(size=n_pts)
        fy_samp = interpolate.splev(samp_y, fy)

        # Assign to redshift bin
        i_sort = fy_samp.argsort()
        fy_sorted = fy_samp[i_sort]

        # Find indices of z bins
        y_bin_idx = fy_sorted.searchsorted(self.y_vals)
        bin_counts = np.diff(y_bin_idx)

        # Calc Total Entropy
        py = bin_counts / n_pts
        hy = self.calc_h(py)

        # Calc Color distributions
        colors = []
        errors = []
        for bin_idx in range(self.y_steps):
            bin_colors, bin_errors = self.calc_colors(self.sed_list[bin_idx])
            colors.append(bin_colors)
            errors.append(bin_errors)

        # Store distributions as functions
        color_funcs = []
        for y_idx in range(self.y_steps):
            color_funcs.append([])
            for sed_num in range(self.n_seds):
                color_funcs[-1].append(
                    stats.multivariate_normal(mean=colors[y_idx][sed_num],
                                              cov=np.diag(errors[y_idx][sed_num])**2.))

        # Draw samples from color distributions
        x_sample = np.zeros((n_pts, self.n_colors))
        dx_sample = np.zeros((n_pts))

        x_idx = 0
        for idx in range(self.y_steps):
            func_choose = rand_state.randint(self.n_seds, size=bin_counts[idx])
            func_count = np.bincount(func_choose)
            for func_num in range(len(func_count)):
                slc = slice(x_idx, x_idx + func_count[func_num])
                func_x_samp = color_funcs[idx][func_num].rvs(size=func_count[func_num],
                                                             random_state=rand_state)
                x_sample[slc] = func_x_samp.reshape(func_count[func_num], self.n_colors)
                dx_sample[slc] = np.linalg.norm(errors[idx][func_num])
                x_idx += func_count[func_num]

        # Calc pxy from density
        n_neighbors = 150
        pxy = np.zeros(n_pts)
        dist = np.zeros(n_pts)
        for idx in range(self.y_steps):
            slc = slice(y_bin_idx[idx], y_bin_idx[idx+1])
            # pxy[slc] = self.nearest_neighbors_density(x_sample[slc],
            #                                           dx_sample[slc],
            #                                           normalize=True)
            # pxy[slc] = self.kernel_estimate_density(x_sample[slc], 0.08)#np.mean(dx_sample[slc]))
            dist_slc = self.knn_density(x_sample[slc], n_neighbors)
            pxy[slc] = n_neighbors# / (dist_slc[:, -1] ** x_sample.shape[1])
            dist[slc] = dist_slc[:, -1]

        # Calc px from density
        px = self.nearest_neighbors_density(x_sample, dist,
                                            normalize=False)
        # px = px / (dist ** x_sample.shape[1])
        # px = self.knn_density(x_sample, 3)
        # px = self.kernel_estimate_density(x_sample, 0.08)#np.mean(dx_sample))

        # pxy = np.exp(pxy)
        # px = np.exp(px)

        print(pxy)
        print(px)
        # print(dx_sample)

        # ig = (1./n_pts) * np.log2(pxy / px).sum()
        ig = (1./n_pts) * np.log2((px) / pxy).sum()

        return hy - ig, hy

    def calc_IG(self, rand_state=None):

        """
        Use the entropy and conditional entropy methods and then subtract
        to get information gain.
        """

        info_gain, h_y = self.calc_gain(rand_state)

        print(info_gain, h_y)

        return info_gain
