import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
from matplotlib.collections import LineCollection
from matplotlib import tri
from matplotlib.patches import Ellipse
from . import calcIG
from .filters import filterFactory
from .lsst_utils import BandpassDict


__all__ = ["plotting"]


class plotting(object):

    def __init__(self, sed_list, best_point, filt_type,
                 calib_filter=None, set_ratio=None, set_width=None,
                 frozen_filt_dict=None, frozen_filt_eff_wavelen=None,
                 sky_mag=20.47, sed_mags=25.0):

        self.filter_obj = filterFactory.create_filter_object(filt_type)
        self.filter_obj.set_wavelen_grid()

        filter_info = self.filter_obj.calc_corners_from_shape_params(
            set_ratio, set_width, best_point
        )

        filt_dict = self.filter_obj.create_filter_dict_from_corners(
            filter_info
        )

        hardware_filt_dict, total_filt_dict = \
            BandpassDict.addSystemBandpass(filt_dict)

        if frozen_filt_dict is None:
            self.filter_dict = hardware_filt_dict
        else:
            if (type(frozen_filt_eff_wavelen) != list):
                raise ValueError("If including frozen filters, " +
                                 "need list of eff. wavelengths.")
            filter_wavelengths = frozen_filt_eff_wavelen +\
                self.filter_obj.find_filt_centers(filt_dict)
            filter_names_unsort = frozen_filt_dict.keys() +\
                hardware_filt_dict.keys()
            filter_list_unsort = frozen_filt_dict.values() +\
                hardware_filt_dict.values()
            sort_idx = np.argsort(filter_wavelengths)
            filter_names = [filter_names_unsort[idx] for idx in sort_idx]
            filter_list = [filter_list_unsort[idx] for idx in sort_idx]

            self.filter_dict = BandpassDict(filter_list, filter_names)

        self.sed_list = sed_list

        self.sky_mag = sky_mag
        self.sed_mags = sed_mags
        self.set_ratio = set_ratio
        self.set_width = set_width

        bp_dict_folder = os.path.join(os.path.dirname(__file__),
                                      'data',
                                      'lsst_baseline_throughputs')
        bp_dict = BandpassDict.loadTotalBandpassesFromFiles(
            bandpassDir=bp_dict_folder)

        if calib_filter is None:
            self.calib_filter = bp_dict['i']
        else:
            self.calib_filter = calib_filter

    def plot_filters(self, fig=None):

        """
        Plot the filters over the top of the SEDs used to optimize.
        """

        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        for sed_obj in self.sed_list:
            plt.plot(sed_obj.wavelen, sed_obj.flambda/np.max(sed_obj.flambda),
                     c='k', alpha=0.5)

        c_list = np.linspace(0, 1, len(self.filter_dict.values()))

        cmap = plt.get_cmap('rainbow')

        for name, filt, color in zip(self.filter_dict.keys(),
                                     self.filter_dict.values(), c_list):
            plt.fill(filt.wavelen, filt.sb,
                     c=cmap(color),
                     zorder=10, alpha=0.6, label=name)
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
                         include_err=True, ellip_kwargs={}):

        """
        Plot the color-color tracks for each SED template as a function
        of redshift using pairs of filters.
        """

        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        shift_seds = []
        shift_values = []

        for idx, z_val in list(enumerate(redshift_list)):
            sed_copies = []
            for sed_obj in self.sed_list:
                sed_copy = deepcopy(sed_obj)
                f_norm = sed_copy.calcFluxNorm(self.sed_mags,
                                               self.calib_filter)
                sed_copy.multiplyFluxNorm(f_norm)
                sed_copy.redshiftSED(z_val, dimming=True)
                sed_copies.append(sed_copy)
                shift_values.append(idx)
            shift_seds.append(sed_copies)

        filler_probs = np.ones(len(shift_seds))
        filler_probs[0] = 0.

        color_x_dict = BandpassDict([self.filter_dict[filt] for filt
                                     in filter_names[0]], filter_names[0])
        color_y_dict = BandpassDict([self.filter_dict[filt] for filt
                                     in filter_names[1]], filter_names[1])

        calc_ig = calcIG(color_x_dict, shift_seds,
                         filler_probs, np.ones(len(shift_seds)),
                         sky_mag=self.sky_mag)

        col_x = []
        err_x = []
        for z_seds in shift_seds:
            cx, ex = calc_ig.calc_colors(z_seds)
            col_x.append(cx)
            err_x.append(ex)
        col_x = np.array(col_x).reshape(len(shift_values))
        err_x = np.array(err_x).reshape(len(shift_values))

        calc_ig = calcIG(color_y_dict, shift_seds,
                         filler_probs, np.ones(len(shift_seds)),
                         sky_mag=self.sky_mag)

        col_y = []
        err_y = []
        for z_seds in shift_seds:
            cy, ey = calc_ig.calc_colors(z_seds)
            col_y.append(cy)
            err_y.append(ey)
        col_y = np.array(col_y).reshape(len(shift_values))
        err_y = np.array(err_y).reshape(len(shift_values))

        for sed_num in range(len(self.sed_list)):

            start_idx = sed_num
            end_idx = len(shift_seds)*len(self.sed_list)
            slc = slice(start_idx, end_idx, len(self.sed_list))

            self.colorline(col_x[slc],
                           col_y[slc],
                           cmap=cmap)

        plt.xlabel('%s - %s' % (filter_names[0][0], filter_names[0][1]))
        plt.ylabel('%s - %s' % (filter_names[1][0], filter_names[1][1]))

        plt.xlim(np.min(col_x) - 0.5, np.max(col_x) + 0.5)
        plt.ylim(np.min(col_y) - 0.5, np.max(col_y) + 0.5)

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(
                                       vmin=np.min(redshift_list),
                                       vmax=np.max(redshift_list)))
        sm._A = []
        plt.colorbar(sm, label='Redshift')

        if include_err is True:
            ax = plt.gca()
            for x, y, w, h in zip(col_x, col_y, err_x, err_y):
                ellipse = Ellipse(xy=(x, y), width=w, height=h,
                                  **ellip_kwargs)
                ax.add_artist(ellipse)

        return fig

    def plot_ig_space(self, test_pts, test_vals, filter_idx,
                      return_centers=False):

        """
        Plot the information gain space by filter center position.
        """

        filt_centers = []

        for filter_set in test_pts:

            filter_dict = self.filter_obj.create_filter_dict_from_shape_params(
                self.set_ratio, self.set_width, filter_set
            )

            filt_centers.append(self.filter_obj.find_filt_centers(filter_dict))

        filt_centers = np.array(filt_centers)

        keep_idx = []
        for idx, filter_vals in list(enumerate(filt_centers)):
            if np.sum(1.0*np.isnan(filter_vals)) == 0:
                keep_idx.append(idx)
        filt_centers = filt_centers[keep_idx]

        xi, yi = filt_centers[:, filter_idx].T
        triang = tri.Triangulation(xi, yi)

        xx, yy = np.meshgrid(np.linspace(np.min(xi), np.max(xi), 100),
                             np.linspace(np.min(yi), np.max(yi), 100))

        interp_lin = tri.LinearTriInterpolator(triang, test_vals[keep_idx])
        zi_lin = interp_lin(xx, yy)

        extent = [np.min(xi), np.max(xi), np.min(yi), np.max(yi)]
        plt.imshow(zi_lin, cmap=plt.cm.plasma, origin='lower',
                   extent=extent, interpolation='bicubic')

        if return_centers is True:
            return xi, yi

    def plot_color_distributions(self, filter_names, redshift_list,
                                 cmap=plt.get_cmap('plasma'), fig=None,
                                 add_cbar=False, include_prior=None):

        """
        Plot the distribution of colors for each SED template as a function
        of redshift using pairs of filters.
        """

        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        shift_seds = []
        shift_values = []

        for idx, z_val in list(enumerate(redshift_list)):
            sed_copies = []
            for sed_obj in self.sed_list:
                sed_copy = deepcopy(sed_obj)
                f_norm = sed_copy.calcFluxNorm(self.sed_mags,
                                               self.calib_filter)
                sed_copy.multiplyFluxNorm(f_norm)
                sed_copy.redshiftSED(z_val, dimming=True)
                sed_copies.append(sed_copy)
                shift_values.append(idx)
            shift_seds.append(sed_copies)

        color_x_dict = BandpassDict([self.filter_dict[filt] for filt
                                     in filter_names], filter_names)
        filler_probs = np.ones(len(shift_seds))
        filler_probs[0] = 0.

        calc_ig = calcIG(color_x_dict, shift_seds,
                         filler_probs, shift_values,
                         sky_mag=self.sky_mag)

        col_x = []
        err_x = []
        for z_seds in shift_seds:
            cx, ex = calc_ig.calc_colors(z_seds)
            col_x.append(cx)
            err_x.append(ex)
        col_x = np.array(col_x).reshape((len(shift_values),
                                         len(filter_names)-1))
        err_x = np.array(err_x).reshape((len(shift_values),
                                         len(filter_names)-1))

        cmap = plt.get_cmap('plasma')

        c_extent = np.linspace(0, 1, len(redshift_list))

        i = 0
        for c, err, idx in zip(col_x, err_x, shift_values):
            pts = np.linspace(c[0]-5*err[0], c[0]+5*err[0], 100)
            vals = stats.norm.pdf(pts, loc=c[0], scale=err[0])
            if include_prior is not None:
                vals *= include_prior(redshift_list[idx])

            plt.fill_between(pts, 0,
                             vals, alpha=0.3,
                             color=cmap(c_extent[idx]))
            i += 1

        plt.ylabel('Probability Density')
        plt.xlabel('%s - %s Color' % (filter_names[0],
                                      filter_names[1]))

        if add_cbar is True:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                                        vmin=np.min(redshift_list),
                                        vmax=np.max(redshift_list)))
            sm._A = []

            new_ax = fig.add_axes([0.045, 0.03, 0.91, 0.03])

            cbar = plt.colorbar(sm, cax=new_ax, orientation='horizontal')
            cbar.set_label('Galaxy Redshift')

        return fig
