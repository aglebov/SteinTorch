import pytest
import numpy as np

from steintorch.divergence.ksd2 import _calculate_integral


def test_calculate_integral():
    with pytest.raises(AssertionError):
        # the integrand is not a matrix
        integrand = np.array([1., 2.])
        _calculate_integral(integrand)

    with pytest.raises(ValueError):
        # the dimensions of the integrand matrix and the weights vector do not align
        integrand = np.array([[1., 2.], [3., 4.]])
        weights = np.array([1., 2., 3.])
        _calculate_integral(integrand, weights)

    # no weights are provided
    integrand = np.arange(1, 10).reshape((3, 3)) ** 2
    np.testing.assert_almost_equal(_calculate_integral(integrand), 285 / 9)
    np.testing.assert_almost_equal(_calculate_integral(integrand, use_all_values=False), 178 / 6)

    # weights provided
    weights = np.array([1, 2, 3]) / 6
    np.testing.assert_almost_equal(
        _calculate_integral(integrand, weights),
        weights @ integrand @ weights
    )
    np.testing.assert_almost_equal(
        _calculate_integral(integrand, weights, use_all_values=False),
        (weights @ integrand @ weights - weights ** 2 @ np.diag(integrand)) / (1 - np.sum(weights ** 2))
    )

    # consistency with and without weights
    np.testing.assert_almost_equal(
        _calculate_integral(integrand),
        _calculate_integral(integrand, np.ones(3))
    )
    np.testing.assert_almost_equal(
        _calculate_integral(integrand, use_all_values=False),
        _calculate_integral(integrand, np.ones(3), use_all_values=False)
    )
