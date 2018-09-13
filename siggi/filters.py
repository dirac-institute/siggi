import numpy as np
from . import Bandpass, BandpassDict

__all__ = ["filters"]


class filters(object):

    """
    This class enables easy creation of various filters that are loaded into
    a Bandpass dictionary.
    """

    def __init__(self, wavelen_min=300., wavelen_max=1200., wavelen_step=0.1):

        self.wavelen_min = wavelen_min
        self.wavelen_max = wavelen_max
        self.wavelen_step = wavelen_step

        return

    def trap_filters(self, filter_details):

        """

        Return a bandpass dictionary with trapezoid filters.

        Input
        -----

        filter_details, list, (n_filters, 4)

            Each row should have the lower left, upper left, upper right
            and lower right corners of the filter in wavelength space.

        Returns
        -------

        bandpass_dict, BandpassDict Object

            This can then be used to calculate magnitudes on spectra
        """

        if (len(np.shape(filter_details)) == 2 and
                np.shape(filter_details)[1] == 4):
            pass
        elif (len(np.shape(filter_details)) == 1 and
              np.shape(filter_details)[0] == 4):
            filter_details = np.reshape(filter_details, (1, 4))
        else:
            raise ValueError("Input should be (n_filters, 4) size array")

        bandpass_list = []

        for band in filter_details:

            offset = self.wavelen_step / 2.

            wavelen_arr = np.arange(self.wavelen_min,
                                    self.wavelen_max+offset,
                                    self.wavelen_step)

            sb = np.zeros(len(wavelen_arr))

            climb_width_l = band[1] - band[0]
            climb_width_r = band[3] - band[2]

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

            bandpass_dict = BandpassDict(bandpass_list, name_list)

        return bandpass_dict
