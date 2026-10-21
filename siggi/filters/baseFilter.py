import numpy as np


__all__ = ["baseFilter"]


class baseFilter(object):

    """
    Abstract base for building filters into a ``BandpassDict``.

    Defines the shape-parameter to corners to bandpass pipeline and the
    area-weighted filter-center calculation shared by the concrete filter
    types. Subclasses (e.g.
    :class:`~siggi.filters.trapezoidFilter.trapezoidFilter`) implement
    :meth:`create_filter_dict_from_corners`.
    """

    def __init__(self):

        self.wavelen_min = None
        self.wavelen_max = None
        self.wavelen_step = None
        self.wavelen_grid_set = False

        return

    def set_wavelen_grid(self, wavelen_min=300., wavelen_max=1200.,
                         wavelen_step=0.1):

        """
        Set the wavelength grid, in nanometres, on which filters are built.

        Parameters
        ----------
        wavelen_min : float, optional
            Minimum wavelength of the grid. Default 300.
        wavelen_max : float, optional
            Maximum wavelength of the grid. Default 1200.
        wavelen_step : float, optional
            Spacing between grid points. Default 0.1.
        """

        self.wavelen_min = wavelen_min
        self.wavelen_max = wavelen_max
        self.wavelen_step = wavelen_step
        self.wavelen_grid_set = True

        return

    def create_filter_dict_from_corners(self, filter_corners):

        """
        Build a ``BandpassDict`` from filter corner specifications.

        Abstract method; subclasses (e.g.
        :class:`~siggi.filters.trapezoidFilter.trapezoidFilter`) implement
        the mapping from corner coordinates to bandpass objects.

        Parameters
        ----------
        filter_corners : list
            One entry per filter, each giving the corner wavelengths that
            define the filter shape.

        Raises
        ------
        NotImplementedError
            Always, on the base class.
        """

        raise NotImplementedError

    def calc_corners_from_shape_params(self, set_ratio, set_width, best_point):

        """
        Convert optimizer shape parameters into filter corner coordinates.

        The interpretation of ``best_point`` depends on which shape
        constraints are fixed:

        - ``set_ratio`` and ``set_width`` both set: each value in
          ``best_point`` is a filter's left edge; width and top/bottom
          ratio are fixed.
        - ``set_ratio`` set, ``set_width`` unset: ``best_point`` is read in
          pairs giving each filter's lower-left/lower-right edges; the top
          width is ``set_ratio`` times the bottom width.
        - neither set: ``best_point`` is read in groups of four giving all
          four corners of each trapezoid directly.

        Parameters
        ----------
        set_ratio : float or None
            Fixed ratio of top width to bottom width, or ``None``.
        set_width : float or None
            Fixed bottom width, or ``None``.
        best_point : array_like
            Flat array of shape parameters from the optimizer.

        Returns
        -------
        list
            One ``[lower_left, top_left, top_right, lower_right]`` corner
            list per filter.
        """

        if set_ratio is not None:

            filter_corners = []

            if set_width is None:
                for i in range(int(len(best_point)/2)):
                    edges = np.array(best_point[2*i:2*(i+1)])
                    bottom_len = edges[1] - edges[0]
                    top_len = set_ratio*bottom_len
                    center = edges[0] + bottom_len/2.
                    top_left = center - top_len/2.
                    top_right = center + top_len/2.
                    filter_corners.append([edges[0], top_left,
                                           top_right, edges[1]])
            else:
                for i in range(int(len(best_point))):
                    edges = np.array(best_point[i:(i+1)])
                    bottom_len = set_width
                    top_len = set_ratio*bottom_len
                    center = edges[0] + bottom_len/2.
                    top_left = center - top_len/2.
                    top_right = center + top_len/2.
                    filter_corners.append([edges[0], top_left,
                                           top_right,
                                           edges[0]+set_width])
        else:
            filter_corners = [best_point[4*i:4*(i+1)]
                              for i in range(int(len(best_point)/4))]

        return filter_corners

    def create_filter_dict_from_shape_params(self, set_ratio, set_width,
                                             best_point):

        """
        Build a ``BandpassDict`` directly from optimizer shape parameters.

        Convenience wrapper that chains
        :meth:`calc_corners_from_shape_params` and
        :meth:`create_filter_dict_from_corners`.

        Parameters
        ----------
        set_ratio : float or None
            Fixed ratio of top width to bottom width, or ``None``.
        set_width : float or None
            Fixed bottom width, or ``None``.
        best_point : array_like
            Flat array of shape parameters from the optimizer.

        Returns
        -------
        BandpassDict
            The resulting filter bandpasses.
        """

        filter_corners = self.calc_corners_from_shape_params(
            set_ratio, set_width, best_point
        )
        filt_dict = self.create_filter_dict_from_corners(filter_corners)

        return filt_dict

    def find_filt_centers(self, filter_dict):

        """
        Calculate the area-weighted center wavelength of each filter.

        The center is the wavelength at which half of the area under the
        transmission curve lies to the left and half to the right.

        Parameters
        ----------
        filter_dict : BandpassDict
            Bandpass dictionary holding the filters to measure.

        Returns
        -------
        list of float
            The center wavelength of each filter, in the order of
            ``filter_dict``.
        """

        filt_centers = []

        for bandpass_obj in list(filter_dict.values()):
            sb_normalized = bandpass_obj.sb / np.sum(bandpass_obj.sb)
            sb_cumsum = np.cumsum(sb_normalized)
            filt_centers.append(
                bandpass_obj.wavelen[np.where(sb_cumsum >= 0.5)][0]
            )

        return filt_centers
