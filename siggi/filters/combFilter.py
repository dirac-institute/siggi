import numpy as np
from siggi.lsst_utils import Bandpass, BandpassDict
from siggi.filters.trapezoidFilter import trapezoidFilter

__all__ = ["combFilter"]


class combFilter(trapezoidFilter):

    """
    Build a single comb bandpass by summing trapezoid teeth.

    Subclass of
    :class:`~siggi.filters.trapezoidFilter.trapezoidFilter` that merges the
    individual trapezoids into one comb-shaped throughput curve.
    """

    def create_filter_dict_from_corners(self, filter_corners):

        """
        Build a single comb bandpass by summing trapezoid filters.

        The individual trapezoids described by ``filter_corners`` are summed
        into one throughput curve (clipped to a maximum of 1.0) and returned
        as a single-entry ``BandpassDict`` named ``filter_0``.

        Parameters
        ----------
        filter_corners : array_like, shape (n_filters, 4)
            Corner wavelengths of each trapezoid tooth of the comb.

        Returns
        -------
        BandpassDict
            A one-filter dict holding the combined comb bandpass.
        """

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
