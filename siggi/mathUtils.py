import numpy as np
from scipy.special import gamma

__all__ = ["integrationUtils"]


class integrationUtils(object):

    def __init__(self):

        return

    def calc_integral_scaling(self, distances, n_dim, dim_scale):

        # Volume of hyperellipsoid: https://math.stackexchange.com/
        # questions/332391/volume-of-hyperellipsoid

        norm_factor = (distances[1:]**n_dim -
                       distances[:-1]**n_dim)
        norm_factor = np.append((distances[0]**n_dim),
                                norm_factor)
        norm_factor *= ((np.pi**(n_dim/2.))/gamma((n_dim/2.)+1))
        norm_factor *= np.linalg.det(np.diag(dim_scale))

        return norm_factor
