import numpy as np
from siggi.lsst_utils import Bandpass, BandpassDict
from siggi.filters.trapezoidFilter import trapezoidFilter

__all__ = ["combFilter"]


class combFilter(trapezoidFilter):

    """
    This class enables easy creation of comb filters
    """

    def create_filter_dict_from_corners(self, filter_corners):

        bandpass_list, bandpass_names = \
            self.create_filter_bandpasses_from_corners(filter_corners)

        comb_sb = np.zeros(len(bandpass_list[0].sb))
        for bandpass_obj in bandpass_list:
            comb_sb += bandpass_obj.sb

        comb_sb[comb_sb > 1.0] = 1.0

        comb_bandpass = Bandpass(wavelen=bandpass_list[0].wavelen,
                                 sb=comb_sb)

        bandpass_dict = BandpassDict([comb_bandpass], ['filter_0'])

        return bandpass_dict
