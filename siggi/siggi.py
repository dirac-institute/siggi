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
                         filter_type='trap',
                         default_width=120., default_ratio=0.5,
                         adjust_widths=False, width_min=30., width_max=120.,
                         adjust_width_ratio=False, adjust_independently=False,
                         ratio_min=0.5, ratio_max=0.9,
                         system_wavelen_min=300., system_wavelen_max=1150.,
                         procs=4, n_opt_points=100, skopt_kwargs_dict=None,
                         optimize_parallel=False):

        self.num_filters = num_filters
        self.adjust_widths = adjust_widths
        self.adjust_ratios = adjust_width_ratio
        self.default_width = default_width
        self.default_ratio = default_ratio
        self.width_max = width_max
        self.sky_mag = sky_mag
        self.sed_mags = sed_mags
        self.filt_min = filt_min
        self.filt_max = filt_max
        self.system_wavelen_min = system_wavelen_min
        self.system_wavelen_max = system_wavelen_max
        self.adjust_ind = adjust_independently

        dim_list, x0 = self.set_dimensions(width_min, width_max,
                                           ratio_min, ratio_max)

        skopt_kwargs = {'x0': x0,
                        'n_calls': n_opt_points}
        if skopt_kwargs_dict is not None:
            for key, val in skopt_kwargs_dict.items():
                skopt_kwargs[key] = val

        if optimize_parallel is False:
            opt = gp_minimize(self.calc_results, dim_list, **skopt_kwargs)
        else:
            i = 0
            opt = Optimizer(dimensions=[Real(x1, x2) for x1, x2 in dim_list],
                            random_state=1)
            with Parallel(n_jobs=procs, backend="threading",
                          batch_size=1, verbose=100) as parallel:
                while i < 10:
                    if i == 0:
                        x = skopt_kwargs['x0']
                    else:
                        x = opt.ask(n_points=procs*2)

                    y = parallel(delayed(unwrap_self_f)(arg1, val) for
                                 arg1, val in zip([self]*len(x), x))

                    opt.tell(x, y)

                    non_zero = np.where(np.array(y) != 0)[0]
                    i += len(non_zero)

                    print(min(opt.yi), i)

        return opt

    def set_dimensions(self, width_min, width_max, ratio_min, ratio_max):

        dim_list = [(self.filt_min,
                     self.filt_max) for n in range(self.num_filters)]
        x0 = list(np.linspace(self.filt_min, self.filt_max,
                              self.num_filters))

        if self.adjust_widths is True:
            if self.adjust_ind is True:
                for i in range(self.num_filters):
                    dim_list.insert(0, (width_min,
                                        width_max))
                    x0.insert(0, self.default_width)
            else:
                dim_list.insert(0, (width_min,
                                    width_max))
                x0.insert(0, self.default_width)

        if self.adjust_ratios is True:
            if self.adjust_ind is True:
                for i in range(self.num_filters):
                    dim_list.insert(0, (ratio_min,
                                        ratio_max))
                    x0.insert(0, self.default_ratio)
            else:
                dim_list.insert(0, (ratio_min,
                                    ratio_max))
                x0.insert(0, self.default_ratio)

        print(dim_list)

        return dim_list, x0

    def set_filters(self, filt_params):

        if ((self.adjust_widths is False) and (self.adjust_ratios is False)):
            filt_centers = filt_params
        elif (((self.adjust_ratios is False) |
               (self.adjust_widths is False)) and
              (self.adjust_ind is True)):
            filt_centers = filt_params[self.num_filters:]
        elif (((self.adjust_ratios is False) |
               (self.adjust_widths is False)) and
              (self.adjust_ind is False)):
            filt_centers = filt_params[1:]
        elif self.adjust_ind is True:
            filt_centers = filt_params[self.num_filters*2:]
        else:
            filt_centers = filt_params[2:]

        filt_diffs = [filt_centers[idx] - filt_centers[idx-1]
                      for idx in range(1, len(filt_centers))]
        filt_diffs = np.array(filt_diffs, dtype=np.int)

        if np.min(filt_diffs) <= 0:
            return 0

        f = filters(self.system_wavelen_min,
                    self.system_wavelen_max)

        if ((self.adjust_widths is False) and (self.adjust_ratios is False)):

            if filt_centers[0] - self.default_width/2. < self.filt_min:
                return 0
            elif filt_centers[-1] + self.default_width/2. > self.filt_max:
                return 0

            filt_dict = f.trap_filters([[filt_loc, self.default_width,
                                         self.default_ratio*self.default_width]
                                        for filt_loc in filt_centers])

        elif self.adjust_ratios is False:

            filt_widths = np.array(filt_params[:self.num_filters])

            left_edge = np.array(filt_centers) - \
                filt_widths/2.
            right_edge = np.array(filt_centers) + \
                filt_widths/2.

            if np.min(left_edge) < self.filt_min:
                return 0
            elif np.max(right_edge) > self.filt_max:
                return 0

            filt_dict = f.trap_filters([[filt_loc,
                                         filt_width,
                                         self.default_ratio *
                                         filt_width]
                                        for filt_loc, filt_width in
                                        zip(filt_centers, filt_widths)])

        elif self.adjust_widths is False:

            if filt_centers[0] - self.default_width/2. < self.filt_min:
                return 0
            elif filt_centers[-1] + self.default_width/2. > self.filt_max:
                return 0

            filt_ratios = np.array(filt_params[:self.num_filters])

            filt_dict = f.trap_filters([[filt_loc, self.default_width,
                                         filt_ratio * self.default_width]
                                        for filt_loc, filt_ratio in
                                        zip(filt_centers, filt_ratios)])

        else:

            filt_widths = np.array(filt_params[self.num_filters:
                                               self.num_filters*2])
            filt_ratios = np.array(filt_params[:self.num_filters])

            left_edge = np.array(filt_centers) - \
                np.array(filt_widths)/2.
            right_edge = np.array(filt_centers) + \
                np.array(filt_widths)/2.

            if np.min(left_edge) < self.filt_min:
                return 0
            elif np.max(right_edge) > self.filt_max:
                return 0

            filt_dict = f.trap_filters([[f_loc,
                                         f_width,
                                         f_width * f_ratio]
                                        for f_loc, f_width, f_ratio in
                                        zip(filt_centers, filt_widths,
                                            filt_ratios)])

        return filt_dict

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
