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

            offset = self.wavelen_step/2.

            wavelen_arr = np.arange(self.wavelen_min,
                                    self.wavelen_max+offset,
                                    self.wavelen_step)

            sb = np.zeros(len(wavelen_arr))

            climb_width_l = band[1] - band[0]
            climb_width_r = band[3] - band[2]

            if climb_width_l > 0.:
                slope_left = 1./climb_width_l
                climb_values_left = np.array([slope_left*i for i in
                                              np.arange(0,
                                                        climb_width_l+offset,
                                                        self.wavelen_step)])

                climb_steps_left = len(climb_values_left)
            else:
                climb_steps_left = 0

            if climb_width_r > 0.:
                slope_right = 1./climb_width_r
                climb_values_right = np.array([slope_right*i for i in
                                               np.arange(0,
                                                         climb_width_r+offset,
                                                         self.wavelen_step)])

                climb_steps_right = len(climb_values_right)
            else:
                climb_steps_right = 0

            min_idx = np.where(wavelen_arr >=
                               band[0])[0][0]
            max_idx = np.where(wavelen_arr >
                               band[3])[0][0]

            sb[min_idx:max_idx] = 1.0

            if (climb_steps_left > 0):
                sb[min_idx:min_idx+climb_steps_left] = climb_values_left
            if (climb_steps_right > 0):
                sb[max_idx-climb_steps_right:max_idx] = 1. - climb_values_right

            bp_object = Bandpass(wavelen=wavelen_arr, sb=sb)

            bandpass_list.append(bp_object)
            name_list = ['filter_%i' % idx for idx in
                         range(len(bandpass_list))]

            bandpass_dict = BandpassDict(bandpass_list, name_list)

        return bandpass_dict

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
