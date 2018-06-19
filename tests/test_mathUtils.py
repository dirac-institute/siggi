import sys
sys.path.append('..')
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.special import gamma
from siggi.mathUtils import integrationUtils


class testMathUtils(unittest.TestCase):

    def test_integrationUtils(self):

        gauss2d = stats.multivariate_normal
        num_pts = 20000

        test_pts = gauss2d.rvs(mean=np.ones(2),
                               cov=np.diagflat(np.ones(2)**2.),
                               size=num_pts)

        y_samples = test_pts.reshape(num_pts, 2)

        y_dist = cdist(y_samples, np.ones(2).reshape(1, 2)).flatten()
        y_sort = np.argsort(y_dist)
        y_dist = y_dist[y_sort]
        y_samples = y_samples[y_sort]

        i_utils = integrationUtils()
        norm_factor = i_utils.calc_integral_scaling(y_dist, 2,
                                                    np.ones(2))
        y_values = gauss2d.pdf(y_samples, mean=np.ones(2),
                               cov=np.diagflat(np.ones(2))**2.)
        integral_sum = np.nansum(y_values*norm_factor)

        np.testing.assert_almost_equal(integral_sum, 1.0, decimal=3)

if __name__ == '__main__':
    unittest.main()
