import numpy as np
from . import Sed
__all__ = ["calcIG"]

class calcIG(object):

    """
    This class will take a set of SEDs and a set of filters
    and calculate the information gain.
    """

    def __init__(self, filter_dict, sed_list):

        self._filter_dict = filter_dict
        self._sed_list = []
        for sed_obj in sed_list:
            sed_copy = Sed()
            sed_copy.setSED(wavelen=sed_obj.wavelen,
                            flambda=sed_obj.flambda)
            sed_copy.resampleSED(wavelen_match=filter_dict.values()[0].wavelen)
            sed_copy.flambda[np.where(np.isnan(sed_copy.flambda))] = 0.
            f_norm = sed_copy.calcFluxNorm(-10.0, filter_dict.values()[0])
            sed_copy.multiplyFluxNorm(f_norm)
            self._sed_list.append(sed_copy)

        return

    def draw_colors(self, num_points):

        mags_list = []
        color_list = []
        true_sed_list = []

        sed_on = 0
        for sed_obj in self._sed_list:
            flux_list = self._filter_dict.fluxListForSed(sed_obj)
            for i in range(num_points):
                flux_with_errors = flux_list + np.random.normal(loc=0.0, 
                                                                scale=np.sqrt(flux_list)*1e1,
                                                                size=2)
                mags = [sed_obj.magFromFlux(f) for f in flux_with_errors]
                mags_list.append(mags)
                color_list.append([mags[i] - mags[i-1] 
                                   for i in range(len(mags)-1)])
                true_sed_list.append(sed_on)
            sed_on += 1

        return mags_list, color_list, true_sed_list
