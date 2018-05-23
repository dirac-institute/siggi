import numpy as np
import multiprocessing as mp
from skopt import gp_minimize, Optimizer
from skopt.space import Real
from sklearn.externals.joblib import Parallel, delayed
from copy import deepcopy
from functools import reduce
from . import filters, spectra, calcIG
from .lsst_utils import BandpassDict

__all__ = ["siggi"]


def unwrap_self_f(arg, arg2, **kwarg):
    return siggi.calc_results(arg, arg2, **kwarg)


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
        bp_dict = BandpassDict.loadTotalBandpassesFromFiles(
            bandpassDir=bp_dict_folder)

        self.calib_filter = bp_dict['r']

    def optimize_filters(self, filt_min=300., filt_max=1100.,
                         sky_mag=19.0, sed_mags=22.0, num_filters=6,
                         filter_type='trap', frozen_filt_dict=None,
                         frozen_filt_eff_wavelen=None,
                         default_ratio=0.5,
                         top_width=None, bottom_width=None,
                         width_min=30., width_max=120.,
                         adjust_width_ratio=True,
                         ratio_min=0.5, ratio_max=0.9, starting_points=None,
                         system_wavelen_min=300., system_wavelen_max=1150.,
                         procs=1, n_opt_points=100, acq_func_kwargs_dict=None,
                         acq_opt_kwargs_dict=None, checkpointing=True,
                         optimizer_verbosity=100,
                         parallel_backend="multiprocessing"):

        self.num_filters = num_filters
        self.top_width = top_width
        self.bottom_width = bottom_width
        self.adjust_ratios = adjust_width_ratio
        self.width_max = width_max
        self.sky_mag = sky_mag
        self.sed_mags = sed_mags
        self.filt_min = filt_min
        self.filt_max = filt_max
        self.system_wavelen_min = system_wavelen_min
        self.system_wavelen_max = system_wavelen_max
        self.frozen_filt_dict = frozen_filt_dict
        self.frozen_eff_lambda = frozen_filt_eff_wavelen

        # dim_list, x0 = self.set_dimensions(width_min, width_max,
        #                                    ratio_min, ratio_max)

        dim_list, x0 = self.set_dimensions()
        print(dim_list, x0)

        if starting_points is not None:
            x0 = starting_points

        i = 0

        opt = Optimizer(dimensions=[Real(x1, x2) for x1, x2 in dim_list],
                        random_state=1,
                        acq_func_kwargs=acq_func_kwargs_dict,
                        acq_optimizer_kwargs=acq_opt_kwargs_dict)

        with Parallel(n_jobs=procs, batch_size=1, backend=parallel_backend,
                      verbose=optimizer_verbosity) as parallel:
            while i < n_opt_points:
                if i == 0:
                    x = x0
                else:
                    x = []
                    while len(x) < procs:
                        x_pot = opt.ask(n_points=1)
                        filt_input = self.validate_filter_input(x_pot[0])
                        if filt_input is True:
                            x.append(x_pot[0])
                        else:
                            opt.tell(x_pot[0], 0)

                y = parallel(delayed(unwrap_self_f)(arg1, val) for
                             arg1, val in zip([self]*len(x), x))

                opt.tell(x, y)

                non_zero = np.where(np.array(y) != 0)[0]
                i += len(non_zero)

                if checkpointing is True:
                    keep_rows = np.where(np.array(opt.yi) != 0)
                    np.savetxt('yi.out', np.array(opt.yi)[keep_rows])
                    np.savetxt('Xi.out', np.array(opt.Xi)[keep_rows])

                print(min(opt.yi), i)

        return opt

    def set_dimensions(self):

        dim_list = [(self.filt_min,
                     self.filt_max) for n in range(4*self.num_filters)]
        x0 = [list(np.linspace(self.filt_min, self.filt_max,
                   4*self.num_filters))]

        # if self.top_width is not None:
        #     if self.bottom_width is not None:
        #         dim_list = [(self.filt_min,
        #                      self.filt_max) for n in range(self.num_filters)]
        #         left_edge = np.linspace(self.filt_min, self.filt_max,
        #                                 self.num_filters+1)[:-1]
        #         x0 = [list(le)]

        # if self.adjust_widths is True:
        #     if self.adjust_ind is True:
        #         for i in range(self.num_filters):
        #             dim_list.insert(0, (width_min,
        #                                 width_max))
        #             x0.insert(0, self.default_width)
        #     else:
        #         dim_list.insert(0, (width_min,
        #                             width_max))
        #         x0.insert(0, self.default_width)

        # if self.adjust_ratios is True:
        #     if self.adjust_ind is True:
        #         for i in range(self.num_filters):
        #             dim_list.insert(0, (ratio_min,
        #                                 ratio_max))
        #             x0.insert(0, self.default_ratio)
        #     else:
        #         dim_list.insert(0, (ratio_min,
        #                             ratio_max))
        #         x0.insert(0, self.default_ratio)

        # print(dim_list)

        return dim_list, x0

    def validate_filter_input(self, filt_edges):

        if filt_edges[0] < self.filt_min:
            return False
        elif filt_edges[-1] > self.filt_max:
            return False

        filt_input = [filt_edges[4*i:4*(i+1)]
                      for i in range(self.num_filters)]

        print(filt_input)

        for filt_list in filt_input:
            filt_diffs = [filt_list[idx] - filt_list[idx-1]
                          for idx in range(1, len(filt_list))]
            filt_diffs = np.array(filt_diffs, dtype=np.int)
            if np.min(filt_diffs) < 0:
                return False

        return True

    def set_filters(self, filt_params):

        # if ((self.adjust_widths is False) and (self.adjust_ratios is False)):
        #     filt_centers = filt_params
        # elif (((self.adjust_ratios is False) |
        #        (self.adjust_widths is False)) and
        #       (self.adjust_ind is True)):
        #     filt_centers = filt_params[self.num_filters:]
        # elif (((self.adjust_ratios is False) |
        #        (self.adjust_widths is False)) and
        #       (self.adjust_ind is False)):
        #     filt_centers = filt_params[1:]
        # elif self.adjust_ind is True:
        #     filt_centers = filt_params[self.num_filters*2:]
        # else:
        #     filt_centers = filt_params[2:]

        filt_input = [filt_params[4*i:4*(i+1)]
                      for i in range(self.num_filters)]

        f = filters(self.system_wavelen_min,
                    self.system_wavelen_max)

        # if ((self.adjust_widths is False) and (self.adjust_ratios is False)):

        filt_dict = f.trap_filters(filt_input)

        return filt_dict

        # elif self.adjust_ratios is False:

        #     filt_widths = np.array(filt_params[:self.num_filters])

        #     left_edge = np.array(filt_centers) - \
        #         filt_widths/2.
        #     right_edge = np.array(filt_centers) + \
        #         filt_widths/2.

        #     if np.min(left_edge) < self.filt_min:
        #         return 0
        #     elif np.max(right_edge) > self.filt_max:
        #         return 0

        #     filt_dict = f.trap_filters([[filt_loc,
        #                                  filt_width,
        #                                  self.default_ratio *
        #                                  filt_width]
        #                                 for filt_loc, filt_width in
        #                                 zip(filt_centers, filt_widths)])

        # elif self.adjust_widths is False:

        #     if filt_centers[0] - self.default_width/2. < self.filt_min:
        #         return 0
        #     elif filt_centers[-1] + self.default_width/2. > self.filt_max:
        #         return 0

        #     filt_ratios = np.array(filt_params[:self.num_filters])

        #     filt_dict = f.trap_filters([[filt_loc, self.default_width,
        #                                  filt_ratio * self.default_width]
        #                                 for filt_loc, filt_ratio in
        #                                 zip(filt_centers, filt_ratios)])

        # else:

        #     filt_widths = np.array(filt_params[self.num_filters:
        #                                        self.num_filters*2])
        #     filt_ratios = np.array(filt_params[:self.num_filters])

        #     left_edge = np.array(filt_centers) - \
        #         np.array(filt_widths)/2.
        #     right_edge = np.array(filt_centers) + \
        #         np.array(filt_widths)/2.

        #     if np.min(left_edge) < self.filt_min:
        #         return 0
        #     elif np.max(right_edge) > self.filt_max:
        #         return 0

        #     filt_dict = f.trap_filters([[f_loc,
        #                                  f_width,
        #                                  f_width * f_ratio]
        #                                 for f_loc, f_width, f_ratio in
        #                                 zip(filt_centers, filt_widths,
        #                                     filt_ratios)])

        # if self.frozen_filt_dict is None:
        #     return filt_dict
        # else:
        #     filter_wavelengths = self.frozen_eff_lambda + filt_centers
        #     filter_names_unsort = self.frozen_filt_dict.keys() + \
        #         filt_dict.keys()
        #     filter_list_unsort = self.frozen_filt_dict.values() + \
        #         filt_dict.values()
        #     if len(filter_wavelengths) != (self.num_filters +
        #                                    len(self.frozen_eff_lambda)):
        #         raise ValueError("Make sure frozen_filt_eff_wavelen is a list")
        #     sort_idx = np.argsort(filter_wavelengths)
        #     filter_names = [filter_names_unsort[idx] for idx in sort_idx]
        #     filter_list = [filter_list_unsort[idx] for idx in sort_idx]

        #     return BandpassDict(filter_list, filter_names)

    def calc_results(self, filt_params):

        filt_dict = self.set_filters(filt_params)

        if filt_dict == 0:
            return 0

        c = calcIG(filt_dict, self.shift_seds, self.z_probs,
                   sky_mag=self.sky_mag, sed_mags=self.sed_mags,
                   ref_filter=self.calib_filter)
        step_result = c.calc_IG()
        print(filt_params, step_result)

        return -1.*step_result
