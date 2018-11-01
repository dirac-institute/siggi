import os
import numpy as np
import multiprocessing as mp
import pickle
from skopt import gp_minimize, Optimizer
from skopt.space import Real
from sklearn.externals.joblib import Parallel, delayed
from copy import deepcopy
from functools import reduce
from . import filters, spectra, calcIG, _siggiBase
from .lsst_utils import BandpassDict

__all__ = ["siggi"]


def unwrap_self_f(arg, arg2, **kwarg):
    return siggi.calc_results(arg, arg2, **kwarg)


class siggi(_siggiBase):

    """
    Class to run a complete siggi maximization run.

    Input
    -----

    spec_list, list

        List of spectra you wish to use as templates
        already stored as LSST Sims SED Objects

    spec_weights, list

        The weight you wish to assign to each SED
        template, usually just 1./len(spec_list)

    z_prior, function

        The redshift prior function you wish to use.

    z_min, float

        The minimum redshift for each SED

    z_max, float

        The maximum redshift for each SED

    z_steps, int

        The number of steps in the redshift grid

    calib_filter, Bandpass Object, default = lsst 'r' band

        The filter you wish to use to set the sky brightness
        and brightness of the SEDs when optimizing.
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

        bp_dict_folder = os.path.join(os.path.dirname(__file__),
                                      'data',
                                      'lsst_baseline_throughputs')
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
                         system_wavelen_step=0.1,
                         procs=1, n_opt_points=100, acq_func_kwargs_dict=None,
                         acq_opt_kwargs_dict=None, checkpointing=True,
                         optimizer_verbosity=5,
                         parallel_backend="multiprocessing",
                         max_search_factor=50, rand_state=None,
                         load_optimizer=None,
                         save_optimizer=None):

        """
        Run the optimization for redshift estimation over the input filters
        and redshift grid. Define the brightness of the galaxies and the sky.

        Input
        -----

        filt_min, float, default = 300.

            The minimum wavelength in nm of the bandpasses.

        filt_max, float, default = 1100.

            The maximum wavelength in nm of the bandpasses.

        sky_mag, float, default = 19.0

            The default magnitude of the sky background in the calib
            filter specified when instantiating.

        sed_mags, float, default = 22.0

            The default magnitude of the SEDs in the calib filter
            specified when instantiating.

        num_filters, int, default = 6

            The number of filters to optimize

        filter_type, str, default = 'trap'

            The shape of the filters. Currently can only create
            trapezoidal shapes before the CCD response function
            is included.

        frozen_filt_dict, BandpassDict Object or None, default = None

            A set of bandpasses to include in the calculation of colors
            but not to be changed in the optimization. These will stay
            in place while other filters are optimized. This should be
            the bandpass files before the hardware and atmosphere effects
            are included in the throughput.

        frozen_filt_eff_wavelen, list of floats or None, default = None

            If a set of frozen filters is included then a set of
            effective wavelengths must also be added in order to sort their
            positions when calculating colors.

        set_ratio, float or None, default = None

            Use this to only optimize over the width of a set of trapezoidal
            filters with a set ratio between the top width and the bottom width
            specified here. For instance, 1.0 will mean a top hat filter.

        width_min, float, default = 30.

            If set_ratio is used then this is the minimum width the bottom
            of the filter must have.

        width_max, float, default = 120.

            If set_ratio is used then this is the maximum width the bottom
            of the filter can have.

        starting_points, list (n_pts, n_filters, 2 or 4) or None,
            default = None

            This is a list of starting point for the values of the filter
            corners. This is a list of lists where each list contains the s
            tarting positions of the filters.

            If set_ratio is not None then
            each of these lists have two floats to specify the bottom left
            and bottom right of a filter.

            If set_ratio is None then this is the same
            except that the lists are four numbers where the order is bottom
            left, top left, top right, bottom right corners of the filters.

        system_wavelen_min, float, default = 300.

            This is the minimum edge of the Bandpass Objects in nm.

        system_wavelen_max, float default = 1150.

            This is the maximum edge of the Bandpass Objects in nm.

        system_wavelen_step, float default = 0.1

            This is the wavelength step of the Bandpass Objects in nm.

        procs, int, default = 1

            This is the number of processors to use in optimization.

        n_opt_points, int, default = 100

            The minimum number of points to calculate the information gain
            in the optimization. If this is not a multiple of the
            number of processors used then the actual number will be
            greater than this.

        max_search_factor, int, default = 50

            This number multiplied by the number of processors used is the
            maximum number of points the optimizer will try to find
            allowable filter edges before a random allowable set of numbers
            will be chosen to fill in the remaining points in a search
            iteration. This happens since the optimizer will try to pick points
            where the filters are not in order of lowest filter center to
            highest at the beginning of the optimization
            and we want to restrict the parameter space to only where
            this is the case in order to better optimize in shorter time.

        rand_state, numpy RandomState, int or None, default = None

            Provide a RandomState Object or the integer seed for a RandomState
            in order to generate reproducible results.

        load_optimizer, None or scikit-optimizer optimizer object,
            default = None

            If this is a previously saved object then the optimizer will
            continue from this state for n_opt_points. This assumes the
            saved optimizer state ran for at least 10 points.

        save_optimizer, None or str, default = None

            If this is a string then it will save a copy of the state of the
            optimizer at the end to the filename given by the string.

        Returns
        -------

        opt, Optimizer object

            Contains the input and output values of the optimization. Use the
            absolute value of opt.yi for the information gain values. The
            corresponding filter inputs are in opt.Xi.
        """

        self.num_filters = num_filters
        self.ratio = set_ratio
        self.sky_mag = sky_mag
        self.sed_mags = sed_mags
        self.f = filters(system_wavelen_min,
                         system_wavelen_max,
                         system_wavelen_step)
        self.frozen_filt_dict = frozen_filt_dict
        self.frozen_eff_lambda = frozen_filt_eff_wavelen
        self.verbosity = optimizer_verbosity

        if type(rand_state) is int:
            rand_state = np.random.RandomState(rand_state)

        dim_list, x0 = self.set_starting_points(starting_points,
                                                self.num_filters,
                                                filt_min,
                                                filt_max,
                                                ratio=self.ratio,
                                                rand_state=rand_state)
        if self.verbosity >= 10:
            print(dim_list, x0)

        i = 0
        random_points_used = 0

        if load_optimizer is None:
            opt = Optimizer(dimensions=[Real(x1, x2) for x1, x2 in dim_list],
                            random_state=rand_state,
                            acq_func_kwargs=acq_func_kwargs_dict,
                            acq_optimizer_kwargs=acq_opt_kwargs_dict)
        else:
            opt = load_optimizer

        with Parallel(n_jobs=procs, batch_size=1, backend=parallel_backend,
                      verbose=self.verbosity) as parallel:
            while i < n_opt_points:
                if ((i == 0) and (load_optimizer is None)):
                    x = x0
                else:
                    x = []
                    pts_needed = procs
                    pts_tried = 0
                    while len(x) < procs:
                        x_pot = opt.ask(n_points=pts_needed, strategy='cl_max')
                        for point in x_pot:
                            filt_input = \
                              self.validate_filter_input(point,
                                                         filt_min,
                                                         filt_max,
                                                         self.num_filters,
                                                         self.ratio,
                                                         self.f.wavelen_step)
                            if filt_input is True:
                                x.append(point)
                                pts_needed -= 1
                            else:
                                opt.tell(point, 0)
                            pts_tried += 1

                        if ((pts_tried >= max_search_factor*procs) and
                           (pts_needed > 0)):
                            while pts_needed > 0:
                                best_xi = opt.Xi[np.argmin(opt.yi)]
                                next_pt = rand_state.normal(0.,
                                                            0.02*(filt_max -
                                                                  filt_min),
                                                            size=len(best_xi))
                                print(best_xi, next_pt)
                                next_pt += best_xi
                                filt_input = self.validate_filter_input(
                                    next_pt, filt_min, filt_max,
                                    self.num_filters, self.ratio,
                                    self.f.wavelen_step)
                                if filt_input is True:
                                    x.append(list(np.sort(next_pt)))
                                    random_points_used += 1
                                    pts_needed -= 1
                        print(pts_tried)
                    print("Random Points Used: %i" % random_points_used)

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

                # Add random point information
                opt.random_pts_used = random_points_used

        if save_optimizer is not None:
            f = open(save_optimizer, 'wb')
            pickle.dump(opt, f)
            f.close()

        return opt

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
                self.find_filt_centers(filt_input)
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
