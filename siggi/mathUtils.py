import numpy as np
from scipy.special import gamma

__all__ = ["integrationUtils"]


class integrationUtils(object):

    def __init__(self):

        return

    def calc_integral_scaling(self, distances, n_dim, dim_scale):

        """
        The integration works by calculating the volume of
        slices of a hyperellipsoid under the probability density
        curve.

        This part calculates the area of each slice and when
        it is returned it is multiplied by the pdf value
        to get a volume.

        Since the width of the probability distribution
        in each dimension will be different this scales
        each dimension in the integral appropriately.
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
