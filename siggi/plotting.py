import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
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

    # The following come from the ipython notebook here:
    # http://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    # Data manipulation:

    def make_segments(self, x, y):
        '''
        Create list of line segments from x and y coordinates,
        in the correct format for LineCollection:
        an array of the form   numlines x (points per line) x 2 (x and y) array
        '''

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments

    # Interface to LineCollection:

    def colorline(self, x, y, z=None, cmap=plt.get_cmap('copper'),
                  norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
        '''
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        '''

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        # to check for numerical input -- this is a hack
        if not hasattr(z, "__iter__"):
            z = np.array([z])

        z = np.asarray(z)

        segments = self.make_segments(x, y)
        lc = LineCollection(segments, array=z, cmap=cmap, norm=norm,
                            linewidth=linewidth, alpha=alpha)

        ax = plt.gca()
        ax.add_collection(lc)

        return lc

    def plot_color_color(self, filter_names, redshift_list, fig=None):

        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        sed_mags = {filt_name: [] for filt_name in self.filter_dict.keys()}
        for sed_obj in self.sed_list:
            for z_val in redshift_list:
                sed_copy = deepcopy(sed_obj)
                sed_copy.redshiftSED(z_val)
                mags = self.filter_dict.magDictForSed(sed_copy)
                for filt_name in self.filter_dict.keys():
                    sed_mags[filt_name].append(mags[filt_name])

        for key in sed_mags.keys():
            sed_mags[key] = np.array(sed_mags[key])

        cmap = plt.get_cmap('plasma')
        num_z = len(redshift_list)

        for sed_num in range(len(self.sed_list)):

            self.colorline(sed_mags[filter_names[0]][sed_num*num_z:
                                                     (sed_num+1)*num_z] -
                           sed_mags[filter_names[1]][sed_num*num_z:
                                                     (sed_num+1)*num_z],
                           sed_mags[filter_names[2]][sed_num*num_z:
                                                     (sed_num+1)*num_z] -
                           sed_mags[filter_names[3]][sed_num*num_z:
                                                     (sed_num+1)*num_z],
                           cmap=cmap)

        plt.xlabel('%s - %s' % (filter_names[0], filter_names[1]))
        plt.ylabel('%s - %s' % (filter_names[2], filter_names[3]))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0,
                                                                 vmax=2))
        sm._A = []
        plt.colorbar(sm, label='Redshift')

        return fig
