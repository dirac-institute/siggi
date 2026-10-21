import numpy as np
from siggi.lsst_utils import Bandpass, BandpassDict
from siggi.filters.baseFilter import baseFilter

__all__ = ["trapezoidFilter"]


class trapezoidFilter(baseFilter):

    """
    Build trapezoidal filter bandpasses from corner wavelengths.
    """

    def create_filter_bandpasses_from_corners(self, filter_corners):

        """
        Build trapezoid bandpass objects from their corner wavelengths.

        Parameters
        ----------
        filter_corners : array_like, shape (n_filters, 4)
            Each row holds the lower-left, upper-left, upper-right and
            lower-right corner wavelengths of a filter. A single filter may
            be passed as a length-4 one-dimensional array.

        Returns
        -------
        bandpass_list : list of Bandpass
            One trapezoid ``Bandpass`` object per input row.
        name_list : list of str
            The corresponding filter names (``filter_0``, ``filter_1`` ...).

        Raises
        ------
        ValueError
            If the wavelength grid has not been set
            (:meth:`~siggi.filters.baseFilter.baseFilter.set_wavelen_grid`),
            or if ``filter_corners`` is not an ``(n_filters, 4)`` array.
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

        """
        Build a ``BandpassDict`` of trapezoid filters from their corners.

        Parameters
        ----------
        filter_corners : array_like, shape (n_filters, 4)
            Lower-left, upper-left, upper-right and lower-right corner
            wavelengths of each filter.

        Returns
        -------
        BandpassDict
            One ``filter_i`` bandpass per input row.
        """

        bandpass_list, bandpass_names = \
            self.create_filter_bandpasses_from_corners(filter_corners)

        bandpass_dict = BandpassDict(bandpass_list, bandpass_names)

        return bandpass_dict
