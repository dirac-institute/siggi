import numpy as np
import matplotlib.pyplot as plt
from . import filters
from .lsst_utils import BandpassDict

__all__ = ["plotting"]


class plotting(object):

    def __init__(self, sed_list, best_point, width, ratio,
                 frozen_filt_dict=None, frozen_filt_eff_wavelen=None):

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

        if frozen_filt_dict is None:
            self.filter_dict = filter_dict
        else:
            if (type(frozen_filt_eff_wavelen) != list):
                raise ValueError("If including frozen filters, " +
                                 "need list of eff. wavelengths.")
            filter_wavelengths = frozen_filt_eff_wavelen + best_point
            filter_names_unsort = frozen_filt_dict.keys() + \
                filter_dict.keys()
            filter_list_unsort = frozen_filt_dict.values() + \
                filter_dict.values()
            sort_idx = np.argsort(filter_wavelengths)
            filter_names = [filter_names_unsort[idx] for idx in sort_idx]
            filter_list = [filter_list_unsort[idx] for idx in sort_idx]

            self.filter_dict = BandpassDict(filter_list, filter_names)

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

    def plot_color_color(self, filter_names, fig=None):

        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        sed_mags = {filt_name: [] for filt_name in self.filter_dict.keys()}
        for sed_obj in self.sed_list:
            mags = self.filter_dict.magDictForSed(sed_obj)
            for filt_name in self.filter_dict.keys():
                sed_mags[filt_name].append(mags[filt_name])

        for key in sed_mags.keys():
            sed_mags[key] = np.array(sed_mags[key])

        plt.scatter(sed_mags[filter_names[0]] - sed_mags[filter_names[1]],
                    sed_mags[filter_names[2]] - sed_mags[filter_names[3]])
        plt.xlabel('%s - %s' % (filter_names[0], filter_names[1]))
        plt.ylabel('%s - %s' % (filter_names[2], filter_names[3]))

        return fig
