import numpy as np
from copy import deepcopy
from functools import reduce
from . import filters, spectra, calcIG


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

        filt_wave_range = np.linspace(filt_min, filt_max, filt_steps)

        dim_list = [len(filt_wave_range) for i in range(num_filters)]
        if adjust_widths is True:
            dim_list.insert(0, width_steps)
        if adjust_width_ratio is True:
            dim_list.insert(0, ratio_steps)
        result_grid = np.zeros(dim_list)
        num_gridpoints = reduce((lambda x, y: x*y), dim_list)

        f = filters(filt_min-width_max, filt_max+width_max)

        step_on = 0

        for idx in range(num_gridpoints):

            step_indices = np.unravel_index(idx, dim_list)
            if step_indices[0] == step_on:
                print(step_on)
                step_on += 1

            filt_centers = [filt_wave_range[filt_idx] 
                            for filt_idx in step_indices]
            filt_dict = f.trap_filters([[filt_loc, 120, 60]
                                        for filt_loc in filt_centers])

            c = calcIG(filt_dict, self.shift_seds, self.z_prior, self.z_min,
                       self.z_max, self.z_steps, snr=snr_level)
            result_grid[step_indices] = c.calc_IG()

#print(step_indices, filt_centers, result_grid[step_indices])

        return result_grid
