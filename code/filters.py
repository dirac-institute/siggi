import numpy as np
from lsst_utils import Bandpass


class filters(object):

    """
    This class enables easy creation of various filters that can be loaded into
    a Bandpass dictionary.
    """

    def __init__(self, wavelen_min=300, wavelen_max=15000, wavelen_step=0.1):

        self.wavelen_min = wavelen_min
        self.wavelen_max = wavelen_max
        self.wavelen_step = wavelen_step

        return

    def trap_filters(self, filter_details):

        """
        Input
        -----

        filter_details, numpy array, (n_filters, 3)

            Each row should have the center location, base width, top width
            of a filter

        Returns
        -------

        bandpass_list, list of Bandpass objects

            This bandpass list can then be used at the input to BandpassDict
        """

        if len(np.shape(filter_details)) == 2:
            pass
        elif (len(np.shape(filter_details)) == 1 and
              np.shape(filter_details)[0] == 3):
            filter_details = np.reshape(filter_details, (1, 3))
        else:
            raise ValueError("Input should be (n_filters, 3) size array")

        bandpass_list = []

        for band in filter_details:

            offset = self.wavelen_step/2.

            wavelen_arr = np.arange(self.wavelen_min, 
                                    self.wavelen_max+offset,
                                    self.wavelen_step)

            sb = np.zeros(len(wavelen_arr))

            climb_width = (band[1] - band[2])/2.
            slope = 1./climb_width
            climb_values = np.array([slope*i for i in \
                                     np.arange(0, climb_width+offset, 
                                               self.wavelen_step)])
            
            climb_steps = len(climb_values)
            min_idx = np.where(wavelen_arr >= (band[0]-(band[1]/2.)-offset))[0][0]
            max_idx = np.where(wavelen_arr >= (band[0]+(band[1]/2.)+offset))[0][0]
            sb[min_idx:max_idx] = 1.0
            sb[min_idx:min_idx+climb_steps] = climb_values
            sb[max_idx-climb_steps:max_idx] = 1. - climb_values

            bp_object = Bandpass(wavelen=wavelen_arr, sb=sb)

            bandpass_list.append(bp_object)

        return bandpass_list
