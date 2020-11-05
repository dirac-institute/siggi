from siggi.filters import trapezoidFilter
from siggi.filters import combFilter

__all__ = ["filterFactory"]


class filterFactory(object):

    """
    Factory class for different filters.
    """

    @staticmethod
    def create_filter_object(filter_type):
        """Create the filter object.

        Parameters
        ----------
        filterType : str, 'trap' or 'comb'
            Type of filter to set up.

        Returns
        -------


        Raises
        ------
        ValueError
            The centroid find type is not supported.
        """

        if filter_type == 'trap':
            return trapezoidFilter()
        elif filter_type == 'comb':
            return combFilter()
        else:
            raise ValueError("The %s filter is not supported." % filter_type)