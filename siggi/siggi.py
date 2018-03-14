import numpy as np
import multiprocessing as mp
from skopt import gbrt_minimize
from copy import deepcopy
from functools import reduce
from . import filters, spectra, calcIG

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

    def optimize_filters(self, filt_min=300., filt_max=1200., filt_steps=10,
                         snr_level=5., num_filters=6, filter_type='trap',
                         default_width=120., default_ratio=0.5,
                         adjust_widths=False, width_min=30., width_max=120.,
                         width_steps=10, adjust_width_ratio=False, 
                         ratio_min=0.5, ratio_max=0.9, ratio_steps=10,
                         procs=4):

        self.filt_wave_range = np.linspace(filt_min, filt_max, filt_steps)

        dim_list = [len(self.filt_wave_range) for i in range(num_filters)]

        self.width_list = None
        self.ratio_list = None
        self.default_width = default_width
        self.default_ratio = default_ratio
        self.width_max = width_max
        self.snr_level = snr_level

        dim_list = [(300., 1200.) for n in range(num_filters)]

        if adjust_widths is True:
            dim_list.insert(0, (width_min, width_max))
        if adjust_width_ratio is True:
            dim_list.insert(0, (ratio_min, ratio_max))

        print(dim_list)

        res = gbrt_minimize(self.grid_results, dim_list, n_jobs=-1)

        # result_grid = np.reshape(pool_res, dim_list)

        return res

    def grid_results(self, filt_params):


        # step_indices = np.unravel_index(idx, dim_list)

        # if idx % reduce((lambda x, y: x*y), dim_list[-2:]) == 0:
        print(filt_params)

        if ((self.width_list is None) and (self.ratio_list is None)):
            filt_centers = filt_params
        elif ((self.ratio_list is None) | (self.width_list is None)):
            filt_centers = filt_params[1:]
        else:
            filt_centers = filt_params[2:]

        filt_diffs = [filt_centers[idx] - filt_centers[idx-1] 
                      for idx in range(1, len(filt_centers))]
        filt_diffs = np.array(filt_diffs, dtype=np.int)

        if np.min(filt_diffs) <= 0:
            return 0.

        f = filters(self.filt_wave_range[0] - self.width_max,
                    self.filt_wave_range[-1] + self.width_max)

        if ((self.width_list is None) and (self.ratio_list is None)):
            filt_dict = f.trap_filters([[filt_loc, self.default_width,
                                         self.default_ratio*self.default_width]
                                        for filt_loc in filt_centers])
        elif self.ratio_list is None:
            filt_dict = f.trap_filters([[filt_loc, 
                                         self.width_list[filt_params[0]],
                                         self.default_ratio *
                                         self.width_list[filt_params[0]]]
                                        for filt_loc in filt_centers])
        elif self.width_list is None:
            filt_dict = f.trap_filters([[filt_loc, self.default_width,
                                         self.ratio_list[filt_params[0]] *
                                         self.default_width]
                                        for filt_loc in filt_centers])
        else:
            filt_dict = f.trap_filters([[filt_loc, 
                                         self.width_list[filt_params[1]],
                                        self.ratio_list[filt_params[0]] *
                                        self.width_list[filt_params[1]]]
                                        for filt_loc in filt_centers])

        c = calcIG(filt_dict, self.shift_seds, self.z_probs,
                   snr=self.snr_level)
        step_result = c.calc_IG()

        return -1.*step_result
