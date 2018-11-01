import os
import numpy as np
from copy import deepcopy
from functools import reduce
from . import filters, spectra, calcIG
from .lsst_utils import BandpassDict

__all__ = ["_siggiBase"]


class _siggiBase(object):

    """
    Base class for siggi optimization.
    """

    def set_starting_points(self, x0, num_filters, filt_min, filt_max,
                            ratio=None, rand_state=None):

        if x0 is None:
            x0 = []
            add_pts = 7
        else:
            # Check that form is appropriate
            if ratio is not None:
                assert (np.shape(x0)[1] == 2*num_filters), \
                    str("Wrong shape of x0. " +
                        "Need 2*number of filters in each starting point.")
            else:
                assert (np.shape(x0)[1] == 4*num_filters), \
                    str("Wrong shape of x0. " +
                        "Need 4*number of filters in each starting point.")
            add_pts = 7 - len(x0)

        if rand_state is None:
            rand_state = np.random.RandomState()

        if ratio is not None:
            x0_len = 2*num_filters
            dim_list = [(filt_min, filt_max)
                        for n in range(x0_len)]
            cuts = 2
        else:
            x0_len = 4*num_filters
            dim_list = [(filt_min,
                         filt_max) for n in range(x0_len)]
            cuts = 4

        # Create multiple starting points
        x_starts = list(np.linspace(filt_min, filt_max,
                                    num_filters+1))
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

    def validate_filter_input(self, filt_edges, filt_min, filt_max,
                              num_filters, ratio=None,
                              wavelen_step=0.1):

        # Make sure input is correct shape
        if ratio is not None:
            assert (len(filt_edges) == num_filters*2)
        else:
            assert (len(filt_edges) == num_filters*4)

        if filt_edges[0] < filt_min:
            return False
        elif filt_edges[-1] > filt_max:
            return False

        if ratio is not None:

            filt_input = []

            for i in range(num_filters):
                edges = np.array(filt_edges[2*i:2*(i+1)])
                bottom_len = edges[1] - edges[0]
                top_len = ratio*bottom_len
                center = edges[0] + bottom_len/2.
                top_left = center - top_len/2.
                top_right = center + top_len/2.
                filt_input.append([edges[0], top_left, top_right, edges[1]])
        else:
            filt_input = [filt_edges[4*i:4*(i+1)]
                          for i in range(num_filters)]

        for filt_list in filt_input:
            filt_diffs = [filt_list[idx] - filt_list[idx-1]
                          for idx in range(1, len(filt_list))]
            filt_diffs = np.array(filt_diffs)
            if np.min(filt_diffs) < -1.e-6*wavelen_step:
                return False
            elif np.max(filt_diffs) <= 0:
                return False
            elif filt_list[-1] - filt_list[0] < 3.*wavelen_step:
                return False

        # At this point if there is only one filter we are good
        # Else we will continue and check that the filters centers
        # Are increasing uniformly to limit the parameter space
        if num_filters == 1:
            return True

        filt_centers = self.find_filt_centers(filt_input)
        print(filt_centers, filt_input)
        filt_diffs = [filt_centers[idx] - filt_centers[idx-1]
                      for idx in range(1, len(filt_centers))]
        filt_diffs = np.array(filt_diffs)

        if np.min(filt_diffs) < 0:
            return False

        return True

    def find_filt_centers(self, filter_details):

        """
        Take in the filter input corners and calculate the weighted center
        of the filter in wavelength space.

        We calculate the center by finding the point where half of the
        area under the transmission curve is to the left and half to
        the right of the given point.

        Input
        -----

        filter_details, list, (n_filters, 4)

            Each row should have the lower left, upper left, upper right
            and lower right corners of the filter in wavelength space.

        Returns
        -------

        filt_centers, list of floats

            The wavelength values of the calculated centers of the input
            filters.
        """

        if (len(np.shape(filter_details)) == 2 and
                np.shape(filter_details)[1] == 4):
            pass
        elif (len(np.shape(filter_details)) == 1 and
              np.shape(filter_details)[0] == 4):
            filter_details = np.reshape(filter_details, (1, 4))
        else:
            raise ValueError("Input should be (n_filters, 4) size array")

        filt_centers = []

        for filt in filter_details:
            a1 = (filt[1] - filt[0])/2.
            a2 = (filt[2] - filt[1])
            a3 = (filt[3] - filt[2])/2.
            half_area = (a1 + a2 + a3)/2.

            if a1 == half_area:
                filt_centers.append(filt[1])
            elif a1 > half_area:
                frac_a1 = half_area/a1
                length_ha = np.sqrt(frac_a1*(filt[1] - filt[0])**2.)
                filt_centers.append(filt[0] + length_ha)
            elif (a1+a2) > half_area:
                half_a2 = half_area - a1
                length_ha = (half_a2/a2)*(filt[2] - filt[1])
                filt_centers.append(filt[1] + length_ha)
            elif (a1+a2) == half_area:
                filt_centers.append(filt[2])
            else:
                half_a3 = half_area - (a1+a2)
                frac_a3 = half_a3/a3
                length_ha = np.sqrt((1-frac_a3)*(filt[3]-filt[2])**2.)
                filt_centers.append(filt[3] - length_ha)

        return filt_centers
