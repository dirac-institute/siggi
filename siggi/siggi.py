import numpy as np
import multiprocessing as mp
from copy import deepcopy
from functools import reduce
from . import filters, spectra, calcIG

def unwrap_self_f(arg, **kwarg):
    return siggi.grid_results(*arg, **kwarg)


class siggi(object):

    """
    Class to run a complete siggi maximization run.
    """

    def __init__(self, spec_list, z_prior, z_min=0., z_max=2.5, z_steps=51):

        self.shift_seds = []

        for spec in spec_list:
            for z_val in np.linspace(z_min, z_max, z_steps):
                spec_copy = deepcopy(spec)
                spec_copy.redshiftSED(z_val)
                self.shift_seds.append(spec_copy)

        self.z_prior = z_prior
        self.z_min = z_min
        self.z_max = z_max
        self.z_steps = z_steps

    def optimize_filters(self, filt_min=300., filt_max=1200., filt_steps=10,
                         snr_level=5., num_filters=6, filter_type='trap',
                         adjust_widths=False, width_min=30., width_max=120.,
                         width_steps=10, adjust_width_ratio=False, 
                         ratio_min=0.1, ratio_max=1.0, ratio_steps=10):

        self.filt_wave_range = np.linspace(filt_min, filt_max, filt_steps)

        dim_list = [len(self.filt_wave_range) for i in range(num_filters)]
        if adjust_widths is True:
            dim_list.insert(0, width_steps)
        if adjust_width_ratio is True:
            dim_list.insert(0, ratio_steps)

        num_points = reduce((lambda x, y: x*y), dim_list)

        step_on = 0

        processes = []

        pool = mp.Pool(processes=4)

        pool_res = pool.map(unwrap_self_f, zip([self]*num_points, 
                                               [[idx, dim_list, width_max,
                                                 snr_level]
                                                for idx in range(num_points)]))

        result_grid = np.reshape(pool_res, dim_list)

        return result_grid

    def grid_results(self, arg_list):

        idx = arg_list[0]
        dim_list = arg_list[1]
        width_max = arg_list[2]
        snr_level = arg_list[3]

        step_indices = np.unravel_index(idx, dim_list)

        filt_centers = [self.filt_wave_range[filt_idx] 
                        for filt_idx in step_indices]

        filt_diffs = [filt_centers[idx] - filt_centers[idx-1] 
                    for idx in range(1, len(filt_centers))]
        filt_diffs = np.array(filt_diffs, dtype=np.int)
        
        if np.min(filt_diffs) < 0:
            return 0.

        f = filters(self.filt_wave_range[0] - width_max,
                    self.filt_wave_range[-1] + width_max)

        filt_dict = f.trap_filters([[filt_loc, 120, 60]
                                    for filt_loc in filt_centers])

        c = calcIG(filt_dict, self.shift_seds, self.z_prior, self.z_min,
                   self.z_max, self.z_steps, snr=snr_level)
        step_result = c.calc_IG()

        return step_result
