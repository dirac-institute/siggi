import sys
sys.path.append('..')
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.special import gamma
from siggi.mathUtils import integrationUtils


class testMathUtils(unittest.TestCase):

    def test_integral_scaling(self):

        y_distances = np.array([0.5, 1.0, 1.5])

        i_utils = integrationUtils()

        # Since we use absolute differences from zero and the unit
        # 1-ball is on the interval [-1, 1] the truth in 1-d is double
        # difference
        truth_1d = np.array([0.5, 0.5, 0.5])*2

        i_out_1d = i_utils.calc_integral_scaling(y_distances,
                                                 1, [1.0])

        np.testing.assert_array_almost_equal(truth_1d, i_out_1d)

        truth_2d = [np.pi*(0.5**2),
                    np.pi*(1.0**2 - 0.5**2),
                    np.pi*(1.5**2 - 1.0**2)]

        i_out_2d = i_utils.calc_integral_scaling(y_distances,
                                                 2, np.ones(2))

        np.testing.assert_array_almost_equal(truth_2d, i_out_2d)

        truth_3d = [(4./3.)*np.pi*(0.5**3),
                    (4./3.)*np.pi*(1.0**3 - 0.5**3),
                    (4./3.)*np.pi*(1.5**3 - 1.0**3)]

        i_out_3d = i_utils.calc_integral_scaling(y_distances,
                                                 3, np.ones(3))

        np.testing.assert_array_almost_equal(truth_3d, i_out_3d)

    def test_integration_1d(self):

        gauss = stats.multivariate_normal
        num_pts = 20000

        test_pts = gauss.rvs(mean=1.0, cov=[1.0], size=num_pts)

        y_samples = test_pts.reshape(num_pts, 1)

        y_dist = cdist(y_samples, [[1.0]]).flatten()
        y_sort = np.argsort(y_dist)
        y_dist = y_dist[y_sort]
        y_samples = y_samples[y_sort]

        i_utils = integrationUtils()
        norm_factor = i_utils.calc_integral_scaling(y_dist, 1, [1.0])

        y_values = gauss.pdf(y_samples, mean=1.0, cov=[1.0])
        integral_sum = np.nansum(y_values*norm_factor)

        np.testing.assert_almost_equal(integral_sum, 1.0, decimal=3)

    def test_integrationUtils_same_scale(self):

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

    def test_integrationUtils_diff_scale(self):

        gauss2d = stats.multivariate_normal
        num_pts = 20000

        test_pts = gauss2d.rvs(mean=np.ones(2),
                               cov=np.diagflat(np.array([.3, .5])**2.),
                               size=num_pts)

        y_samples = test_pts.reshape(num_pts, 2)

        inv_cov = np.linalg.inv(np.diagflat(np.array([.3, .5])**2.))
        y_dist = cdist(y_samples, np.ones(2).reshape(1, 2),
                       metric='mahalanobis',
                       VI=inv_cov).flatten()
        y_sort = np.argsort(y_dist)
        y_dist = y_dist[y_sort]
        y_samples = y_samples[y_sort]

        i_utils = integrationUtils()
        norm_factor = i_utils.calc_integral_scaling(y_dist, 2,
                                                    np.array([.3, .5]))
        y_values = gauss2d.pdf(y_samples, mean=np.ones(2),
                               cov=np.diagflat(np.array([.3, .5])**2.))
        integral_sum = np.nansum(y_values*norm_factor)

        np.testing.assert_almost_equal(integral_sum, 1.0, decimal=3)

if __name__ == '__main__':
    unittest.main()
