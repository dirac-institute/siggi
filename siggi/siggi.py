import os
import numpy as np
import pickle
from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed
from copy import deepcopy
from . import calcIG, _siggiBase
from .lsst_utils import BandpassDict
from .filters import filterFactory

__all__ = ["siggi"]


def unwrap_self_f(arg, arg2, **kwarg):
    """Module-level trampoline to call :meth:`siggi.calc_results`.

    Lets a bound ``siggi.calc_results`` method be dispatched through joblib's
    parallel backend, which requires a top-level (picklable) callable.

    Parameters
    ----------
    arg : siggi
        The ``siggi`` instance providing ``calc_results``.
    arg2 : BandpassDict
        The filter dictionary passed through to ``calc_results``.
    **kwarg
        Additional keyword arguments forwarded to ``calc_results``.

    Returns
    -------
    float
        The result of ``arg.calc_results(arg2, **kwarg)``.
    """
    return siggi.calc_results(arg, arg2, **kwarg)


class siggi(_siggiBase):

    """
    Run a complete siggi filter-optimization.

    On construction the template SEDs are redshifted across a grid and
    flux-normalized to a calibration magnitude, ready for
    :meth:`optimize_filters` to search for the most informative filter set.

    Parameters
    ----------
    spec_list : list
        Template spectra to use, as LSST Sims ``Sed`` objects.
    spec_weights : list
        Weight assigned to each SED template, usually
        ``1. / len(spec_list)``.
    z_prior : callable
        Redshift prior function, evaluated on the redshift grid.
    z_min : float, optional
        Minimum redshift of the grid. Default 0.0.
    z_max : float, optional
        Maximum redshift of the grid. Default 2.5.
    z_steps : int, optional
        Number of steps in the redshift grid. Default 51.
    calib_filter : Bandpass, optional
        Filter used to set the sky brightness and the brightness of the SEDs
        when optimizing. Defaults to the LSST ``i`` band.
    calib_mag : float, optional
        Magnitude in ``calib_filter`` to which observed SEDs are normalized.
        Default 25.3.
    phot_params : dict, optional
        Mapping of filter name to ``PhotometricParameters``. If ``None``,
        default LSST parameters are used.
    """

    def __init__(self, spec_list, spec_weights, z_prior,
                 z_min=0.00, z_max=2.5, z_steps=51,
                 calib_filter=None, calib_mag=25.3,
                 phot_params=None):

        self.num_sed_types = len(spec_list)
        self.shift_seds = []
        self.z_probs = []

        bp_dict_folder = os.path.join(os.path.dirname(__file__),
                                      'data',
                                      'lsst_baseline_throughputs')
        bp_dict = BandpassDict.loadTotalBandpassesFromFiles(
            bandpassDir=bp_dict_folder)

        if calib_filter is None:
            self.calib_filter = bp_dict['i']
        else:
            self.calib_filter = calib_filter

        z_vals = np.linspace(z_min, z_max, z_steps)

        self.z_probs = z_prior(z_vals)
        self.z_vals = z_vals

        for z_val in z_vals:
            z_specs = []
            for spec in spec_list:
                spec_copy = deepcopy(spec)
                spec_copy.redshiftSED(z_val)
                f_norm = spec_copy.calcFluxNorm(calib_mag, self.calib_filter)
                spec_copy.multiplyFluxNorm(f_norm)
                z_specs.append(spec_copy)
            self.shift_seds.append(z_specs)

        self.phot_params = phot_params

    def optimize_filters(self, filt_min=300., filt_max=1100.,
                         sky_mag=20.47, num_filters=6,
                         filt_type='trap', frozen_filt_dict=None,
                         frozen_filt_eff_wavelen=None,
                         set_ratio=None,
                         set_width=None,
                         starting_points=None,
                         starting_vals=None,
                         system_wavelen_min=300., system_wavelen_max=1200.,
                         system_wavelen_step=0.1,
                         procs=1, n_opt_points=100, acq_func_kwargs_dict=None,
                         acq_opt_kwargs_dict=None, checkpointing=True,
                         optimizer_verbosity=5,
                         parallel_backend="multiprocessing",
                         max_search_factor=50, rand_state=None,
                         load_optimizer=None,
                         save_optimizer=None):

        """
        Optimize the filter set for information gain about redshift.

        Runs a Bayesian optimization (scikit-optimize) over the filter-edge
        parameters on the redshift grid built at construction, evaluating the
        information gain of each validated candidate filter set, optionally in
        parallel.

        Parameters
        ----------
        filt_min : float, optional
            Minimum wavelength of the bandpasses, in nm. Default 300.
        filt_max : float, optional
            Maximum wavelength of the bandpasses, in nm. Default 1100.
        sky_mag : float, optional
            Magnitude of the sky background in the calibration filter set at
            construction. Default 20.47.
        num_filters : int, optional
            Number of filters to optimize. Default 6.
        filt_type : {'trap', 'comb'}, optional
            Shape of the filters (before the CCD response is included):
            trapezoidal or comb. Default ``'trap'``.
        frozen_filt_dict : BandpassDict, optional
            Bandpasses to include when computing colors but hold fixed during
            optimization. These should be the throughputs before hardware and
            atmosphere effects are included. Requires
            ``frozen_filt_eff_wavelen``.
        frozen_filt_eff_wavelen : list of float, optional
            Effective wavelengths of the frozen filters, used to sort their
            positions when computing colors. Required with
            ``frozen_filt_dict``.
        set_ratio : float, optional
            Optimize only the width of trapezoidal filters with this fixed
            ratio of top width to bottom width. For example, ``1.0`` gives a
            top-hat filter.
        set_width : float, optional
            Fixed bottom width of the filters, in nm. Requires ``set_ratio``.
        starting_points : list, optional
            Starting configurations for the filter parameters, one list per
            point. With ``set_ratio`` set (and no ``set_width``) each list has
            two values per filter (bottom-left, bottom-right); with neither
            set, four values per filter (bottom-left, top-left, top-right,
            bottom-right).
        starting_vals : list, optional
            Pre-computed objective values corresponding to the leading entries
            of ``starting_points``, fed to the optimizer with ``tell``.
        system_wavelen_min : float, optional
            Minimum edge of the ``Bandpass`` objects, in nm. Default 300.
        system_wavelen_max : float, optional
            Maximum edge of the ``Bandpass`` objects, in nm. Default 1200.
        system_wavelen_step : float, optional
            Wavelength step of the ``Bandpass`` objects, in nm. Default 0.1.
        procs : int, optional
            Number of processors to use. Default 1.
        n_opt_points : int, optional
            Minimum number of points at which to evaluate the information
            gain. If not a multiple of ``procs`` the actual number will be
            larger. Default 100.
        acq_func_kwargs_dict : dict, optional
            Extra keyword arguments passed to the optimizer's acquisition
            function.
        acq_opt_kwargs_dict : dict, optional
            Extra keyword arguments passed to the optimizer's acquisition
            optimizer.
        checkpointing : bool, optional
            If ``True``, write the running ``Xi.out`` and ``yi.out`` arrays to
            the working directory each iteration. Default ``True``.
        optimizer_verbosity : int, optional
            Verbosity level for the optimizer and the parallel backend.
            Default 5.
        parallel_backend : str, optional
            joblib parallel backend. Default ``"multiprocessing"``.
        max_search_factor : int, optional
            Multiplied by ``procs``, the maximum number of points the
            optimizer will try in a search iteration before filling the rest
            with random allowable points. This bounds the time spent rejecting
            out-of-order filter proposals early in the run. Default 50.
        rand_state : numpy.random.RandomState or int, optional
            Random state object or integer seed for reproducible results.
        load_optimizer : skopt.Optimizer, optional
            A previously saved optimizer to continue from for ``n_opt_points``
            more points. Assumes the saved state ran for at least 10 points.
        save_optimizer : str, optional
            If given, pickle the final optimizer state to this filename.

        Returns
        -------
        skopt.Optimizer
            The optimizer holding the inputs and outputs of the run. The
            information-gain values are ``np.abs(opt.yi)`` and the
            corresponding filter parameters are in ``opt.Xi``.

        Raises
        ------
        AssertionError
            If ``frozen_filt_dict`` is given without
            ``frozen_filt_eff_wavelen``.
        """

        if frozen_filt_dict is not None:
            assert_str = str('If frozen_filt_dict specified must also ' +
                             'include corresponding frozen_filt_eff_wavelen')
            assert (frozen_filt_eff_wavelen is not None), assert_str

        self.num_filters = num_filters
        self.ratio = set_ratio
        self.sky_mag = sky_mag
        self.width = set_width

        self.f = filterFactory.create_filter_object(filt_type)
        self.f.set_wavelen_grid(system_wavelen_min, system_wavelen_max,
                                system_wavelen_step)
        self.frozen_filt_dict = frozen_filt_dict
        self.frozen_eff_lambda = frozen_filt_eff_wavelen
        self.verbosity = optimizer_verbosity

        if type(rand_state) is int:
            rand_state = np.random.RandomState(rand_state)

        x1 = deepcopy(starting_points)

        dim_list, x0 = self.set_starting_points(starting_points,
                                                self.num_filters,
                                                filt_min,
                                                filt_max,
                                                ratio=self.ratio,
                                                width=self.width,
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

        if starting_vals is not None:

            len_y0 = len(starting_vals)
            x_submit = []
            for x_sample in x1[:len_y0]:
                x_submit.append(list(x_sample))
            opt.tell(x_submit, list(starting_vals))

            if len_y0 < len(x1):
                x0 = [list(x_on) for x_on in x1[len_y0:]]
            else:
                x0 = [list(x_on) for x_on in x0[len_y0:]]

        print(x0)

        with Parallel(n_jobs=procs, batch_size=1, backend=parallel_backend,
                      verbose=self.verbosity) as parallel:
            while i < n_opt_points:
                if ((i == 0) and (load_optimizer is None)):
                    x = x0
                    test_filt_dicts = \
                        [self.f.create_filter_dict_from_shape_params(
                            self.ratio, self.width, x_i
                            )
                         for x_i in x0]
                elif ((load_optimizer is not None) and
                      (starting_points is not None) and (i == 0)):
                    x = x1  # Hack to avoid redoing the 3 base points
                    print(x)
                    test_filt_dicts = \
                        [self.f.create_filter_dict_from_shape_params(
                            self.ratio, self.width, x_i
                            )
                         for x_i in x1]
                else:
                    x = []
                    test_filt_dicts = []
                    pts_needed = procs
                    pts_tried = 0
                    while len(x) < procs:
                        x_pot = opt.ask(n_points=pts_needed, strategy='cl_max')
                        for point in x_pot:
                            filt_dict = \
                                self.create_and_validate_filter_dict(
                                    point, filt_min, filt_max,
                                    self.num_filters, self.f,
                                    self.ratio, self.width
                                )
                            if filt_dict is not None:
                                x.append(point)
                                test_filt_dicts.append(filt_dict)
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
                                filt_dict = \
                                    self.create_and_validate_filter_dict(
                                        next_pt, filt_min, filt_max,
                                        self.num_filters, self.f,
                                        self.ratio, self.width
                                    )
                                if filt_dict is not None:
                                    x.append(list(np.sort(next_pt)))
                                    test_filt_dicts.append(filt_dict)
                                    random_points_used += 1
                                    pts_needed -= 1
                        print(pts_tried)
                    print("Random Points Used: %i" % random_points_used)

                y = parallel(delayed(unwrap_self_f)(arg1, val) for
                             arg1, val in zip([self]*len(x), test_filt_dicts))

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

    def set_filters(self, filt_dict):

        """
        Combine optimized filters with any frozen filters, sorted by center.

        Parameters
        ----------
        filt_dict : BandpassDict
            The optimized (non-frozen) filters.

        Returns
        -------
        BandpassDict
            ``filt_dict`` unchanged if no frozen filters were set; otherwise a
            new dictionary merging the frozen and optimized filters, sorted by
            effective wavelength.

        Raises
        ------
        ValueError
            If the combined effective-wavelength list is not a list (i.e.
            ``frozen_filt_eff_wavelen`` was not supplied as a list).
        """

        if self.frozen_filt_dict is None:
            return filt_dict
        else:
            filter_wavelengths = self.frozen_eff_lambda +\
                self.f.find_filt_centers(filt_dict)
            filter_names_unsort = list(self.frozen_filt_dict.keys()) +\
                list(filt_dict.keys())
            filter_list_unsort = list(self.frozen_filt_dict.values()) +\
                list(filt_dict.values())
            if type(filter_wavelengths) != list:
                raise ValueError("Make sure frozen_filt_eff_wavelen is a list")
            sort_idx = np.argsort(filter_wavelengths)
            filter_names = [filter_names_unsort[idx] for idx in sort_idx]
            filter_list = [filter_list_unsort[idx] for idx in sort_idx]

            return BandpassDict(filter_list, filter_names)

    def calc_results(self, filt_dict):

        """
        Evaluate the (negated) information gain for one filter set.

        Combines the proposed filters with any frozen filters, builds a
        :class:`~siggi.calcIG.calcIG` on the redshift grid, and returns the
        negated information gain so the optimizer can minimize it.

        Parameters
        ----------
        filt_dict : BandpassDict
            The optimized (non-frozen) filters to evaluate.

        Returns
        -------
        float
            Minus the information gain, in bits, or ``0`` if the combined
            filter set is empty/invalid.
        """

        filt_dict = self.set_filters(filt_dict)

        if filt_dict == 0:
            return 0

        c = calcIG(filt_dict, self.shift_seds, self.z_probs,
                   self.z_vals,
                   sky_mag=self.sky_mag,
                   ref_filter=self.calib_filter,
                   phot_params=self.phot_params)
        step_result = c.calc_IG(rand_state=np.random.RandomState(1344))

        if self.verbosity >= 10:
            print(self.f.find_filt_centers(filt_dict), step_result)

        return -1.*step_result
