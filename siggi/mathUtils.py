import numpy as np
from scipy.special import gamma

__all__ = ["mathUtils"]


class mathUtils(object):

    """
    Mathematical helpers shared across ``siggi``.

    Provides Shannon-entropy and hyperellipsoid-volume utilities used by
    the information-gain calculation. Intended to be subclassed (e.g. by
    :class:`~siggi.calcIG.calcIG`).
    """

    def __init__(self):

        return

    def calc_h(self, sed_probs):

        """
        Calculate the Shannon entropy of a probability distribution.

        Parameters
        ----------
        sed_probs : array_like
            Probabilities that sum to 1. Each entry must be positive.

        Returns
        -------
        float
            The total entropy, ``-sum(p * log2(p))``, in bits.
        """

        h_sum = 0

        for py_i in sed_probs:
            h_sum += -(py_i)*np.log2(py_i)

        return h_sum

    def calc_integral_scaling(self, distances, n_dim, dim_scale):

        """
        Compute the volume of hyperellipsoid shells for integration.

        The integration works by calculating the volume of slices of a
        hyperellipsoid under the probability density curve. This method
        returns the area of each slice; multiplying by the pdf value gives
        the volume of the slice. Because the width of the probability
        distribution differs in each dimension, each dimension is scaled
        appropriately via ``dim_scale``.

        Parameters
        ----------
        distances : numpy.ndarray
            Monotonically increasing shell radii. Consecutive entries define
            the inner and outer radius of each shell.
        n_dim : int
            Number of dimensions of the hyperellipsoid.
        dim_scale : array_like
            Per-dimension scale factors (the relative widths of the
            distribution in each dimension), used to build the scaling
            determinant.

        Returns
        -------
        numpy.ndarray
            The scaled shell volume factor for each radius in ``distances``.

        Notes
        -----
        The hyperellipsoid volume uses the n-sphere volume term
        :math:`\\pi^{d/2} / \\Gamma(d/2 + 1)` scaled by the determinant of
        ``diag(dim_scale)``. See
        https://en.wikipedia.org/wiki/N-sphere and
        https://math.stackexchange.com/questions/332391.
        """

        # (r_outer ^ d) - (r_inner ^ d)
        norm_factor = (distances[1:]**n_dim -
                       distances[:-1]**n_dim)
        norm_factor = np.append((distances[0]**n_dim),
                                norm_factor)

        # Volume of hyperellipsoid: https://math.stackexchange.com/
        # questions/332391/volume-of-hyperellipsoid with V_d term
        # from wikipedia: https://en.wikipedia.org/wiki/N-sphere

        norm_factor *= ((np.pi**(n_dim/2.))/gamma((n_dim/2.)+1))
        norm_factor *= np.linalg.det(np.diag(dim_scale))

        return norm_factor
