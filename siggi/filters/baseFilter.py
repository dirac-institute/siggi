import numpy as np


__all__ = ["baseFilter"]


class baseFilter(object):

    """
    This class enables easy creation of various filters that are loaded into
    a Bandpass dictionary.
    """

    def __init__(self):

        self.wavelen_min = None
        self.wavelen_max = None
        self.wavelen_step = None
        self.wavelen_grid_set = False

        return

    def set_wavelen_grid(self, wavelen_min=300., wavelen_max=1200.,
                         wavelen_step=0.1):

        self.wavelen_min = wavelen_min
        self.wavelen_max = wavelen_max
        self.wavelen_step = wavelen_step
        self.wavelen_grid_set = True

        return

    def create_filter_dict_from_corners(self, filter_corners):

        raise NotImplementedError

    def calc_corners_from_shape_params(self, set_ratio, set_width, best_point):

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

        filter_corners = self.calc_corners_from_shape_params(
            set_ratio, set_width, best_point
        )
        filt_dict = self.create_filter_dict_from_corners(filter_corners)

        return filt_dict

    def find_filt_centers(self, filter_dict):

        """
        Take in the filter input corners and calculate the weighted center
        of the filter in wavelength space.

        We calculate the center by finding the point where half of the
        area under the transmission curve is to the left and half to
        the right of the given point.

        Input
        -----

        filter_dict, bandpassDict object

            bandpassDict object with the filters inside

        Returns
        -------

        filt_centers, list of floats

            The wavelength values of the calculated centers of the input
            filters.
        """

        filt_centers = []

        for bandpass_obj in list(filter_dict.values()):
            sb_normalized = bandpass_obj.sb / np.sum(bandpass_obj.sb)
            sb_cumsum = np.cumsum(sb_normalized)
            filt_centers.append(
                bandpass_obj.wavelen[np.where(sb_cumsum >= 0.5)][0]
            )

        return filt_centers
