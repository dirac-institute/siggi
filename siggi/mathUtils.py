import numpy as np
from scipy.special import gamma


class integrationUtils(object):

    def __init__(self):

        return

    def calc_integral_scaling(self, distances, n_dim, dim_scale):

        norm_factor = (distances[1:]**n_dim -
                       distances[:-1]**n_dim)
        norm_factor = np.append((distances[0]**n_dim),
                                norm_factor)
        norm_factor *= ((np.pi**(n_dim/2.))/gamma((n_dim/2.)+1))
        norm_factor *= np.linalg.det(np.diag(dim_scale) /
                                     np.max(dim_scale))

        return norm_factor
