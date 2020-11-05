import numpy as np

__all__ = ["_siggiBase"]


class _siggiBase(object):

    """
    Base class for siggi optimization.
    """

    def set_starting_points(self, x0, num_filters, filt_min, filt_max,
                            ratio=None, width=None, rand_state=None):

        if width is not None:
            assert (ratio is not None), \
                str("Ratio cannot be None when setting a width for now.")

        if ((width is not None) and (width > (filt_max - filt_min))):
            raise ValueError('Width must not be greater than filter bounds.')

        if x0 is None:
            x0 = []
            add_pts = 7
        else:
            # Check that form is appropriate
            if ratio is not None:
                if width is None:
                    assert (np.shape(x0)[1] == 2*num_filters), \
                        str("Wrong shape of x0. " +
                            "Need 2*number of filters in each starting point.")
                else:
                    assert (np.shape(x0)[1] == num_filters), \
                        str("Wrong shape of x0. " +
                            "Need 1*number of filters in each starting point.")
            else:
                assert (np.shape(x0)[1] == 4*num_filters), \
                    str("Wrong shape of x0. " +
                        "Need 4*number of filters in each starting point.")
            add_pts = 7 - len(x0)

        if rand_state is None:
            rand_state = np.random.RandomState()

        if ratio is not None:
            if width is None:
                x0_len = 2*num_filters
                dim_list = [(filt_min, filt_max)
                            for n in range(x0_len)]
                cuts = 2
            else:
                x0_len = num_filters
                dim_list = [(filt_min, filt_max) for n in range(x0_len)]
                cuts = 1
        else:
            x0_len = 4*num_filters
            dim_list = [(filt_min,
                         filt_max) for n in range(x0_len)]
            cuts = 4

        # Create multiple starting points
        x_starts = list(np.linspace(filt_min, filt_max,
                                    num_filters+1))

        if width is not None:
            x_starts = list(np.linspace(filt_min, filt_max-(width),
                                        num_filters))

            x_full_space = np.empty(x0_len)
            for i in range(num_filters):
                x_full_space[cuts*i] = x_starts[i]
            x0.append(list(x_full_space))

            space_length = filt_max - filt_min
            space_third = space_length / 3
            x_starts_l = list(np.linspace(filt_min + space_third,
                                          filt_max-(width), num_filters))
            x_space_l = np.empty(x0_len)
            for i in range(num_filters):
                x_space_l[cuts*i] = x_starts_l[i]
            x0.append(list(x_space_l))

            x_starts_r = list(np.linspace(filt_min + 2*space_third,
                                          filt_max-(width), num_filters))
            x_space_r = np.empty(x0_len)
            for i in range(num_filters):
                x_space_r[cuts*i] = x_starts_r[i]
            x0.append(list(x_space_r))

            if add_pts > 0:
                for i in range(add_pts):
                    x_random = rand_state.uniform(low=filt_min,
                                                  high=filt_max-width,
                                                  size=x0_len)
                    x_random = list(np.sort(x_random))
                    x0.append(x_random)

            return dim_list, x0

        x_full_space = np.empty(x0_len)
        for i in range(num_filters):
            x_full_space[cuts*i] = x_starts[i]
            x_full_space[cuts*i:cuts*(i+1)] = np.linspace(x_starts[i],
                                                          x_starts[i+1],
                                                          cuts)
        x0.append(list(x_full_space))

        space_length = filt_max - filt_min
        x_starts_l = list(np.linspace(filt_min,
                                      filt_max - space_length/2.,
                                      num_filters+1))
        x_half_space_l = np.empty(x0_len)
        for i in range(num_filters):
            x_half_space_l[cuts*i] = x_starts_l[i]
            x_half_space_l[cuts*i:cuts*(i+1)] = np.linspace(x_starts_l[i],
                                                            x_starts_l[i+1],
                                                            cuts)
        x0.append(list(x_half_space_l))

        x_starts_r = list(np.linspace(filt_min + space_length/2.,
                                      filt_max,
                                      num_filters+1))
        x_half_space_r = np.empty(x0_len)
        for i in range(num_filters):
            x_half_space_r[cuts*i] = x_starts_l[i]
            x_half_space_r[cuts*i:cuts*(i+1)] = np.linspace(x_starts_r[i],
                                                            x_starts_r[i+1],
                                                            cuts)
        x0.append(list(x_half_space_r))

        if add_pts > 0:
            for i in range(add_pts):
                x_random = rand_state.uniform(low=filt_min,
                                              high=filt_max,
                                              size=x0_len)
                x_random = list(np.sort(x_random))
                x0.append(x_random)

        return dim_list, x0

    def create_and_validate_filter_dict(self, filt_edges, filt_min, filt_max,
                                        num_filters, filter_obj,
                                        ratio=None, width=None):

        # Make sure input is correct shape
        if ratio is not None:
            if width is None:
                assert (len(filt_edges) == num_filters*2)
            else:
                assert (len(filt_edges) == num_filters)
                assert ((filt_max - filt_min) >= width)
        else:
            assert (len(filt_edges) == num_filters*4)

        assert filter_obj.wavelen_step is not None

        filt_input = filter_obj.calc_corners_from_shape_params(
            ratio, width, filt_edges
        )

        for filt_list in filt_input:
            filt_diffs = [filt_list[idx] - filt_list[idx-1]
                          for idx in range(1, len(filt_list))]
            filt_diffs = np.array(filt_diffs)
            if np.min(filt_diffs) < -1.e-6*filter_obj.wavelen_step:
                return None
            elif np.max(filt_diffs) <= 0:
                return None
            elif filt_list[-1] - filt_list[0] < 3.*filter_obj.wavelen_step:
                return None

        if filt_input[0][0] < filt_min:
            return None
        elif filt_input[-1][-1] > filt_max:
            return None

        filt_dict = filter_obj.create_filter_dict_from_corners(filt_input)

        # At this point if there is only one filter we are good
        # Else we will continue and check that the filters centers
        # Are increasing uniformly to limit the parameter space
        if len(list(filt_dict.values())) == 1:
            return filt_dict

        filt_centers = filter_obj.find_filt_centers(filt_dict)
        print(filt_centers, filt_input)
        filt_diffs = [filt_centers[idx] - filt_centers[idx-1]
                      for idx in range(1, len(filt_centers))]
        filt_diffs = np.array(filt_diffs)

        if np.min(filt_diffs) < 0:
            return None

        return filt_dict
