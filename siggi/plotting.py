import numpy as np
import matplotlib.pyplot as plt
from . import filters
from .lsst_utils import BandpassDict

__all__ = ["plotting"]


class plotting(object):

    def __init__(self, sed_list, best_point, width, ratio):

        f = filters()

        if type(width) == list:
            if type(ratio) == list:
                filter_info = [[filt_cent, filt_width, filt_width*filt_ratio] 
                               for filt_cent, filt_width, filt_ratio in
                               zip(best_point, width, ratio)]
            else:
                filter_info = [[filt_cent, filt_width, filt_width*ratio] 
                               for filt_cent, filt_width in
                               zip(best_point, width)]
        elif type(ratio) == list:
            filter_info = [[filt_cent, width, width*filt_ratio] 
                           for filt_cent, filt_ratio in
                           zip(best_point, ratio)]
        else:
            filter_info = [[filt_cent, width, width*ratio] 
                           for filt_cent in best_point]

        trap_dict = f.trap_filters(filter_info)

        filter_dict, atmos_filt_dict = \
            BandpassDict.addSystemBandpass(trap_dict)

        self.filter_dict = filter_dict
        self.sed_list = sed_list

    def plot_filters(self, fig=None):

        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        for sed_obj in self.sed_list:
            plt.plot(sed_obj.wavelen, sed_obj.flambda/np.max(sed_obj.flambda),
                     c='k', alpha=0.5)

        c_list = np.linspace(0, 1, len(self.filter_dict.values()))

        cmap = plt.get_cmap('rainbow')

        for filt, color in zip(self.filter_dict.values(), c_list):
            plt.fill(filt.wavelen, filt.sb,
                     c=cmap(color),
                     zorder=10, alpha=0.6)
        plt.xlim(filt.wavelen[0] - 50., filt.wavelen[-1] + 50)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Scaled Flux')

        return fig
