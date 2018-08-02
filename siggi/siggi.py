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
                 z_min=0.05, z_max=2.5, z_steps=50,
                 calib_filter=None):

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

        if calib_filter is None:
            self.calib_filter = bp_dict['r']
        else:
            self.calib_filter = calib_filter

    def optimize_filters(self, filt_min=300., filt_max=1100.,
                         sky_mag=19.0, sed_mags=22.0, num_filters=6,
                         filter_type='trap', frozen_filt_dict=None,
                         frozen_filt_eff_wavelen=None,
                         set_ratio=None,
                         width_min=30., width_max=120.,
                         starting_points=None,
                         system_wavelen_min=300., system_wavelen_max=1150.,
                         procs=1, n_opt_points=100, acq_func_kwargs_dict=None,
                         acq_opt_kwargs_dict=None, checkpointing=True,
                         optimizer_verbosity=100,
                         parallel_backend="multiprocessing",
                         rand_state=None):

        self.num_filters = num_filters
        self.ratio = set_ratio
        self.sky_mag = sky_mag
        self.sed_mags = sed_mags
        self.filt_min = filt_min
        self.filt_max = filt_max
        self.f = filters(system_wavelen_min,
                         system_wavelen_max)
        self.frozen_filt_dict = frozen_filt_dict
        self.frozen_eff_lambda = frozen_filt_eff_wavelen
        self.verbosity = optimizer_verbosity

        dim_list, x0 = self.set_dimensions(starting_points,
                                           rand_state=rand_state)
        if self.verbosity >= 10:
            print(dim_list, x0)

        i = 0

        opt = Optimizer(dimensions=[Real(x1, x2) for x1, x2 in dim_list],
                        random_state=rand_state,
                        acq_func_kwargs=acq_func_kwargs_dict,
                        acq_optimizer_kwargs=acq_opt_kwargs_dict)

        with Parallel(n_jobs=procs, batch_size=1, backend=parallel_backend,
                      verbose=self.verbosity) as parallel:
            while i < n_opt_points:
                if i == 0:
                    x = x0
                else:
                    x = []
                    pts_needed = procs
                    while len(x) < procs:
                        x_pot = opt.ask(n_points=pts_needed)
                        filt_input = self.validate_filter_input(x_pot[0])
                        if filt_input is True:
                            x.append(x_pot[0])
                            pts_needed -= 1
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

    def set_dimensions(self, x0, rand_state=None):

        if x0 is None:
            x0 = []
            add_pts = 7
        else:
            add_pts = 7 - len(x0)

        if rand_state is None:
            rand_state = np.random.RandomState()

        if self.ratio is not None:
            x0_len = 2*self.num_filters
            dim_list = [(self.filt_min, self.filt_max)
                        for n in range(x0_len)]
        else:
            x0_len = 4*self.num_filters
            dim_list = [(self.filt_min,
                         self.filt_max) for n in range(x0_len)]

        # Create multiple starting points
        x_full_space = list(np.linspace(self.filt_min, self.filt_max,
                                        x0_len))
        x0.append(x_full_space)
        space_length = self.filt_max - self.filt_min
        x_half_space_l = list(np.linspace(self.filt_min,
                                          self.filt_max - space_length/2.,
                                          x0_len))
        x0.append(x_half_space_l)
        x_half_space_r = list(np.linspace(self.filt_min + space_length/2.,
                                          self.filt_max,
                                          x0_len))
        x0.append(x_half_space_r)

        if add_pts > 0:
            for i in range(add_pts):
                x_random = rand_state.uniform(low=self.filt_min,
                                              high=self.filt_max,
                                              size=x0_len)
                x_random = list(np.sort(x_random))
                x0.append(x_random)

        return dim_list, x0

    def validate_filter_input(self, filt_edges):

        if filt_edges[0] < self.filt_min:
            return False
        elif filt_edges[-1] > self.filt_max:
            return False

        if self.ratio is not None:

            filt_input = []

            for i in range(self.num_filters):
                edges = np.array(filt_edges[2*i:2*(i+1)])
                bottom_len = edges[1] - edges[0]
                top_len = self.ratio*bottom_len
                center = edges[0] + bottom_len/2.
                top_left = center - top_len/2.
                top_right = center + top_len/2.
                filt_input.append([edges[0], top_left, top_right, edges[1]])
        else:
            filt_input = [filt_edges[4*i:4*(i+1)]
                          for i in range(self.num_filters)]

        for filt_list in filt_input:
            filt_diffs = [filt_list[idx] - filt_list[idx-1]
                          for idx in range(1, len(filt_list))]
            filt_diffs = np.array(filt_diffs, dtype=np.int)
            if np.min(filt_diffs) < 0:
                return False
            elif np.max(filt_diffs) <= 0:
                return False

        filt_centers = self.f.find_filt_centers(filt_input)
        print(filt_centers, filt_input)
        filt_diffs = [filt_centers[idx] - filt_centers[idx-1]
                      for idx in range(1, len(filt_centers))]
        filt_diffs = np.array(filt_diffs, dtype=np.int)
        if np.min(filt_diffs) < 0:
            return False

        return True

    def set_filters(self, filt_params):

        if self.ratio is not None:

            filt_input = []

            for i in range(self.num_filters):
                edges = np.array(filt_params[2*i:2*(i+1)])
                bottom_len = edges[1] - edges[0]
                top_len = self.ratio*bottom_len
                center = edges[0] + bottom_len/2.
                top_left = center - top_len/2.
                top_right = center + top_len/2.
                filt_input.append([edges[0], top_left, top_right, edges[1]])
        else:
            filt_input = [filt_params[4*i:4*(i+1)]
                          for i in range(self.num_filters)]

        filt_dict = self.f.trap_filters(filt_input)

        if self.frozen_filt_dict is None:
            return filt_dict
        else:
            filter_wavelengths = self.frozen_eff_lambda +\
                self.f.find_filt_centers(filt_input)
            filter_names_unsort = self.frozen_filt_dict.keys() +\
                filt_dict.keys()
            filter_list_unsort = self.frozen_filt_dict.values() +\
                filt_dict.values()
            if len(filter_wavelengths) != (self.num_filters +
                                           len(self.frozen_eff_lambda)):
                raise ValueError("Make sure frozen_filt_eff_wavelen is a list")
            sort_idx = np.argsort(filter_wavelengths)
            filter_names = [filter_names_unsort[idx] for idx in sort_idx]
            filter_list = [filter_list_unsort[idx] for idx in sort_idx]

            return BandpassDict(filter_list, filter_names)

    def calc_results(self, filt_params):

        filt_dict = self.set_filters(filt_params)

        if filt_dict == 0:
            return 0

        c = calcIG(filt_dict, self.shift_seds, self.z_probs,
                   sky_mag=self.sky_mag, sed_mags=self.sed_mags,
                   ref_filter=self.calib_filter)
        step_result = c.calc_IG(rand_state=np.random.RandomState(
                        np.int(np.sum(filt_params))))
        if self.verbosity >= 10:
            print(filt_params, step_result)

        return -1.*step_result
