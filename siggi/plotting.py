import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from . import filters, calcIG
from .lsst_utils import BandpassDict

__all__ = ["plotting"]


class plotting(object):

    def __init__(self, sed_list, best_point, width, ratio,
                 frozen_filt_dict=None, frozen_filt_eff_wavelen=None,
                 sky_mag=19.0, sed_mags=22.0):

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

        self.sky_mag = sky_mag
        self.sed_mags = sed_mags

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

    def plot_color_color(self, filter_names, redshift_list,
                         cmap=plt.get_cmap('plasma'), fig=None,
                         include_err=True):

        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        shift_seds = []

        for sed_obj in self.sed_list:
            for z_val in redshift_list:
                sed_copy = deepcopy(sed_obj)
                sed_copy.redshiftSED(z_val)
                shift_seds.append(sed_copy)

        color_x_dict = BandpassDict([self.filter_dict[filt] for filt
                                     in filter_names[:2]], filter_names[:2])
        color_y_dict = BandpassDict([self.filter_dict[filt] for filt
                                     in filter_names[2:]], filter_names[2:])

        calc_ig = calcIG(color_x_dict, shift_seds,
                         np.ones(len(shift_seds)),
                         sky_mag=self.sky_mag, sed_mags=self.sed_mags)
        col_x, err_x = calc_ig.calc_colors()

        calc_ig = calcIG(color_y_dict, shift_seds,
                         np.ones(len(shift_seds)),
                         sky_mag=self.sky_mag, sed_mags=self.sed_mags)
        col_y, err_y = calc_ig.calc_colors()

        num_z = len(redshift_list)

        for sed_num in range(len(self.sed_list)):

            start_idx = sed_num*num_z
            end_idx = start_idx + num_z

            self.colorline(col_x[start_idx:end_idx],
                           col_y[start_idx:end_idx],
                           cmap=cmap)

        plt.xlabel('%s - %s' % (filter_names[0], filter_names[1]))
        plt.ylabel('%s - %s' % (filter_names[2], filter_names[3]))

        plt.xlim(np.min(col_x) - 0.5, np.max(col_x) + 0.5)
        plt.ylim(np.min(col_y) - 0.5, np.max(col_y) + 0.5)

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(
                                       vmin=np.min(redshift_list),
                                       vmax=np.max(redshift_list)))
        sm._A = []
        plt.colorbar(sm, label='Redshift')

        if include_err is True:
            plt.errorbar(col_x, col_y, xerr=err_x, yerr=err_y,
                         ms=2, alpha=0.5, ls=' ')

        return fig
