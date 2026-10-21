from __future__ import division

import os
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
    Calculate the information gain for a set of SEDs and filters.

    This class takes a set of template SEDs (across a grid of the quantity
    of interest, e.g. redshift) and a set of filters, simulates the observed
    colors with realistic photometric noise, and estimates the information
    gain -- the mutual information, in bits, between the colors and the
    quantity of interest.

    Parameters
    ----------
    filter_dict : BandpassDict
        Dictionary with the bandpass objects for the filters to test.
    sed_list : list
        Nested list of template ``Sed`` objects: one inner list per ``y``
        bin, each already redshifted and flux-normalized.
    y_probs : array_like
        Probability (prior) weight for each value in ``y_vals``, in the same
        order. The first entry must be ``0`` (see :meth:`calc_gain`).
    y_vals : array_like
        The grid of the quantity of interest (e.g. redshift bin edges).
    n_pts : int, optional
        Number of Monte-Carlo samples used in the entropy estimate.
        Default 1_000_000.
    sky_mag : float, optional
        Magnitude of the sky background in ``ref_filter``. Default 20.47.
    ref_filter : Bandpass, optional
        Reference filter used to normalize the sky brightness. Defaults to
        LSST ``i``.
    phot_params : dict, optional
        Mapping of filter name to ``PhotometricParameters``. If ``None``,
        default LSST parameters are built for each filter.
    fwhm_eff : float, optional
        Effective FWHM of the seeing, in arcseconds, used in the photometric
        error calculation. Default 0.8.
    """

    def __init__(self, filter_dict, sed_list, y_probs, y_vals,
                 n_pts=1000000,
                 sky_mag=20.47, ref_filter=None, phot_params=None,
                 fwhm_eff=0.8):

        bp_dict_folder = os.path.join(os.path.dirname(__file__),
                                      'data',
                                      'lsst_baseline_throughputs')
        bp_dict = BandpassDict.loadTotalBandpassesFromFiles(
            bandpassDir=bp_dict_folder)

        if ref_filter is None:
            ref_filter = bp_dict['i']

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
        Calculate the colors and color errors of the SEDs in the filters.

        For each SED the magnitudes through the filters are computed and
        differenced into adjacent-filter colors; the color errors are
        propagated from the per-filter magnitude errors (which include the
        sky background and photometric parameters set on construction).

        Parameters
        ----------
        sed_list : list
            List of ``Sed`` objects to measure.
        return_all : bool, optional
            If ``True``, also return the per-filter signal-to-noise ratios,
            magnitudes, magnitude errors, and sky magnitudes. Default
            ``False``.

        Returns
        -------
        sed_colors : numpy.ndarray
            Array of adjacent-filter colors, one row per SED.
        color_errors : numpy.ndarray
            Array of 1-sigma color errors matching ``sed_colors``.
        snr_values : list
            Per-filter signal-to-noise ratios. Only returned when
            ``return_all`` is ``True``.
        sed_mag_list : list
            Per-filter magnitudes for each SED. Only returned when
            ``return_all`` is ``True``.
        mag_errors : list
            Per-filter magnitude errors for the last SED. Only returned when
            ``return_all`` is ``True``.
        sky_mags : list
            Per-filter sky magnitudes. Only returned when ``return_all`` is
            ``True``.
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
                   sed_mag_list, mag_errors, sky_mags

        return np.array(sed_colors), np.array(color_errors)

    def calc_gain(self, rand_state=None):

        """
        Estimate the information gain and total entropy by Monte Carlo.

        Samples the prior over ``y``, builds a multivariate-normal color
        distribution per ``y`` bin and SED, draws color samples, and
        estimates the total entropy ``H(y)`` and the conditional entropy
        ``H(y | colors)`` to return the information gain ``H(y) - H(y|x)``.

        Parameters
        ----------
        rand_state : numpy.random.RandomState, optional
            Random state for reproducible sampling. If ``None``, a fresh
            ``RandomState`` is created.

        Returns
        -------
        info_gain : float
            The information gain, ``H(y) - H(y | colors)``, in bits.
        hy : float
            The total entropy ``H(y)`` of the prior, in bits.

        Raises
        ------
        ValueError
            If the first entry of ``y_probs`` is not ``0`` (the probability
            function must start at ``(0., 0.)``).
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

        if len(self.y_vals) >= 10:
            fy = interpolate.splrep(y_cum, self.y_vals)
        else:
            fy = interpolate.splrep(y_cum, self.y_vals,
                                    k=1)
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

        x_idx = 0
        for idx in range(self.y_steps):
            func_choose = rand_state.randint(self.n_seds, size=bin_counts[idx])
            func_count = np.bincount(func_choose)
            for func_num in range(len(func_count)):
                slc = slice(x_idx, x_idx + func_count[func_num])
                func_x_samp = color_funcs[idx][func_num].rvs(size=func_count[func_num],
                                                             random_state=rand_state)
                x_sample[slc] = func_x_samp.reshape(func_count[func_num], self.n_colors)
                x_idx += func_count[func_num]

        pxy = np.zeros(n_pts)
        px = np.zeros(n_pts)
        x_idx = 0
        for idx in range(self.y_steps):
            slc = slice(x_idx, x_idx + bin_counts[idx])
            for func_num in range(self.n_seds):
                func_px_samp = color_funcs[idx][func_num].pdf(x_sample)
                pxy[slc] += func_px_samp[slc] * (1/(self.n_seds)) * (bin_counts[idx]/n_pts)
                px += func_px_samp * (1/self.n_seds) * (bin_counts[idx]/(n_pts))
            x_idx += bin_counts[idx]

        hyx = -(1./n_pts) * np.log2(pxy / px).sum()

        return hy - hyx, hy

    def calc_IG(self, rand_state=None):

        """
        Compute the information gain.

        Thin wrapper around :meth:`calc_gain` that returns only the
        information gain (the difference of the total and conditional
        entropies).

        Parameters
        ----------
        rand_state : numpy.random.RandomState, optional
            Random state for reproducible sampling. If ``None``, a fresh
            ``RandomState`` is created.

        Returns
        -------
        float
            The information gain, in bits.
        """

        info_gain, h_y = self.calc_gain(rand_state)

        # print(info_gain, h_y)

        return info_gain
