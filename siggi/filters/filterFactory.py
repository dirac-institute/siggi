from siggi.filters import trapezoidFilter
from siggi.filters import combFilter

__all__ = ["filterFactory"]


class filterFactory(object):

    """
    Factory for creating filter-shape objects by name.
    """

    @staticmethod
    def create_filter_object(filter_type):
        """Create a filter-shape object of the requested type.

        Parameters
        ----------
        filter_type : {'trap', 'comb'}
            Type of filter to set up: ``'trap'`` for a
            :class:`~siggi.filters.trapezoidFilter.trapezoidFilter` or
            ``'comb'`` for a :class:`~siggi.filters.combFilter.combFilter`.

        Returns
        -------
        baseFilter
            A new, unconfigured filter object of the requested type.

        Raises
        ------
        ValueError
            If ``filter_type`` is not ``'trap'`` or ``'comb'``.
        """

        if filter_type == 'trap':
            return trapezoidFilter()
        elif filter_type == 'comb':
            return combFilter()
        else:
            raise ValueError("The %s filter is not supported." % filter_type)