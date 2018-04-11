import numpy as np
import multiprocessing as mp
from skopt import gp_minimize
from copy import deepcopy
from functools import reduce
from . import filters, spectra, calcIG
from .lsst_utils import BandpassDict

__all__ = ["siggi"]


def unwrap_self_f(arg, **kwarg):
    return siggi.grid_results(*arg, **kwarg)


class siggi(object):

    """
    Class to run a complete siggi maximization run.
    """

    def __init__(self, spec_list, spec_weights, z_prior,
                 z_min=0.05, z_max=2.5, z_steps=50):

        self.shift_seds = []
        self.z_probs = []

        for spec, weight in zip(spec_list, spec_weights):
            for z_val in np.linspace(z_min, z_max, z_steps):
                spec_copy = deepcopy(spec)
                spec_copy.redshiftSED(z_val)
                self.shift_seds.append(spec_copy)
                self.z_probs.append(z_prior(z_val)*weight)

        bp_dict_folder = '../data/lsst_baseline_throughputs'
        bp_dict = BandpassDict.loadTotalBandpassesFromFiles(bandpassDir=
                                                            bp_dict_folder)

        self.calib_filter = bp_dict['r']

    def optimize_filters(self, filt_min=300., filt_max=1200.,
                         sky_mag=19.0, sed_mags=22.0, num_filters=6, 
                         filter_type='trap',
                         default_width=120., default_ratio=0.5,
                         adjust_widths=False, width_min=30., width_max=120.,
                         adjust_width_ratio=False, 
                         ratio_min=0.5, ratio_max=0.9,
                         procs=4, n_opt_points=100, skopt_kwargs_dict=None):

        self.adjust_widths = adjust_widths
        self.adjust_ratios = adjust_width_ratio
        self.default_width = default_width
        self.default_ratio = default_ratio
        self.width_max = width_max
        self.sky_mag = sky_mag
        self.sed_mags = sed_mags
        self.filt_min = filt_min
        self.filt_max = filt_max

        dim_list = [(filt_min, filt_max) for n in range(num_filters)]
        x0 = list(np.linspace(filt_min, filt_max, num_filters))

        if adjust_widths is True:
            dim_list.insert(0, (width_min, width_max))
            x0.insert(0, self.default_width)
        if adjust_width_ratio is True:
            dim_list.insert(0, (ratio_min, ratio_max))
            x0.insert(0, self.default_ratio)

        print(dim_list)

        skopt_kwargs = {'n_jobs': procs,
                        'x0': x0,
                        'n_calls': n_opt_points}
        if skopt_kwargs_dict is not None:
            for key, val in skopt_kwargs_dict.items():
                skopt_kwargs[key] = val
        
        res = gp_minimize(self.grid_results, dim_list, **skopt_kwargs)

        return res

    def grid_results(self, filt_params):

        if ((self.adjust_widths is False) and (self.adjust_ratios is False)):
            filt_centers = filt_params
        elif ((self.adjust_ratios is False) | (self.adjust_widths is False)):
            filt_centers = filt_params[1:]
        else:
            filt_centers = filt_params[2:]

        filt_diffs = [filt_centers[idx] - filt_centers[idx-1] 
                      for idx in range(1, len(filt_centers))]
        filt_diffs = np.array(filt_diffs, dtype=np.int)

        if np.min(filt_diffs) <= 0:
            return 0.

        f = filters(self.filt_min - self.width_max/2.,
                    self.filt_max + self.width_max/2.)

        if ((self.adjust_widths is False) and (self.adjust_ratios is False)):
            filt_dict = f.trap_filters([[filt_loc, self.default_width,
                                         self.default_ratio*self.default_width]
                                        for filt_loc in filt_centers])
        elif self.adjust_ratios is False:
            filt_dict = f.trap_filters([[filt_loc, 
                                         filt_params[0],
                                         self.default_ratio *
                                         filt_params[0]]
                                        for filt_loc in filt_centers])
        elif self.adjust_widths is False:
            filt_dict = f.trap_filters([[filt_loc, self.default_width,
                                         filt_params[0] *
                                         self.default_width]
                                        for filt_loc in filt_centers])
        else:
            filt_dict = f.trap_filters([[filt_loc, 
                                         filt_params[1],
                                        filt_params[0] *
                                        filt_params[1]]
                                        for filt_loc in filt_centers])

        c = calcIG(filt_dict, self.shift_seds, self.z_probs,
                   sky_mag=self.sky_mag, sed_mags=self.sed_mags,
                   ref_filter=self.calib_filter)
        step_result = c.calc_IG()
        print(filt_params, step_result)

        return -1.*step_result
