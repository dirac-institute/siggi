import numpy as np
from siggi.lsst_utils import Bandpass, BandpassDict
from siggi.filters.baseFilter import baseFilter

__all__ = ["trapezoidFilter"]


class trapezoidFilter(baseFilter):

    """
    This class enables easy creation of trapeziodal filters
    """

    def create_filter_bandpasses_from_corners(self, filter_corners):

        """

        Return a bandpass dictionary with trapezoid filters.

        Input
        -----

        filter_corners, list, (n_filters, 4)

            Each row should have the lower left, upper left, upper right
            and lower right corners of the filter in wavelength space.

        Returns
        -------

        bandpass_dict, BandpassDict Object

            This can then be used to calculate magnitudes on spectra
        """

        # Verify wavelen grid setup

        if self.wavelen_grid_set is False:
            raise ValueError("Wavelen grid needs to be set")

        if (len(np.shape(filter_corners)) == 2 and
                np.shape(filter_corners)[1] == 4):
            pass
        elif (len(np.shape(filter_corners)) == 1 and
              np.shape(filter_corners)[0] == 4):
            filter_corners = np.reshape(filter_corners, (1, 4))
        else:
            raise ValueError("Input should be (n_filters, 4) size array")

        bandpass_list = []

        for band in filter_corners:

            offset = self.wavelen_step / 2.

            wavelen_arr = np.arange(self.wavelen_min,
                                    self.wavelen_max+offset,
                                    self.wavelen_step)

            sb = np.zeros(len(wavelen_arr))

            min_idx = np.where(wavelen_arr >=
                               band[0])[0][0]
            min_top_idx = np.where(wavelen_arr >=
                                   band[1])[0][0]

            max_idx = np.where(wavelen_arr >
                               band[3])[0][0]
            max_top_idx = np.where(wavelen_arr >=
                                   band[2])[0][0]

            climb_steps_right = max_idx - max_top_idx
            climb_steps_left = (min_top_idx+1) - min_idx

            sb[min_idx:max_idx] = 1.0

            climb_values_left = np.linspace(0., 1.0, climb_steps_left)
            sb[min_idx:min_top_idx+1] = climb_values_left

            climb_values_right = np.linspace(1.0, 0., climb_steps_right)
            sb[max_top_idx:max_idx] = climb_values_right

            bp_object = Bandpass(wavelen=wavelen_arr, sb=sb)

            bandpass_list.append(bp_object)
            name_list = ['filter_%i' % idx for idx in
                         range(len(bandpass_list))]

        return bandpass_list, name_list

    def create_filter_dict_from_corners(self, filter_corners):

        bandpass_list, bandpass_names = \
            self.create_filter_bandpasses_from_corners(filter_corners)

        bandpass_dict = BandpassDict(bandpass_list, bandpass_names)

        return bandpass_dict
