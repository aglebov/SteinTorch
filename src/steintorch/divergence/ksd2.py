"""Implementation of gradient-free kernel Stein discrepancy in pure numpy"""

from typing import Callable, Optional, Tuple

import numpy as np


def _calculate_integral(
        integrand: np.ndarray,
        weights: Optional[np.ndarray] = None,
        use_all_values: bool = True,
    ) -> float:
    """Calculate integral given a matrix of integrand values evaluated on a grid

    Parameters
    ----------
    integrand: np.ndarray
        matrix of integrand values of size N * N
    weights: np.ndarray
        vector of weights for points along each dimension of the integrand matrix
    use_all_values: bool
        if True, use all integrand values, otherwise ignore the diagonal elements

    Returns
    -------
    np.float64
        scalar value of the integral
    """
    # make sure the integrand matrix is two-dimensional square
    assert integrand.ndim == 2
    assert integrand.shape[0] == integrand.shape[1]
    N = integrand.shape[0]

    if weights is None:
        if use_all_values:
            return np.mean(integrand)
        else:
            # use only off-diagonal elements
            return 1 / (N * (N - 1)) * (np.sum(integrand) - np.sum(np.diag(integrand)))
    else:
        weights = weights / np.sum(weights)  # normalise the weights
        if use_all_values:
            return np.linalg.multi_dot((weights, integrand, weights))
        else:
            # use only off-diagonal elements
            quad_form = np.linalg.multi_dot((weights, integrand, weights))
            diagonal_terms = weights ** 2 @ np.diag(integrand)
            return (quad_form - diagonal_terms) / (1 - np.sum(weights ** 2))


class GradientFreeKSD:
    """Gradient-free kernel Stein discrepancy"""

    def __init__(
            self,
            log_p: Callable[[np.ndarray], np.ndarray],
            log_q: Callable[[np.ndarray], np.ndarray],
            score_q: Callable[[np.ndarray], np.ndarray],
            preconditioner: Optional[np.ndarray] = 1.0,
            sigma: float = 1.0,
            beta: float = 0.5,
            use_all_values: bool = True,
            clamp_qp: Optional[Tuple[float, float]] = None,
            min_log_p: Optional[float] = None,
    ):
        """Set parameters of kernel Stein discrepancy

        Parameters
        ----------
        log_p: Callable[[np.ndarray], np.ndarray]
            log PDF of the target distribution. This function must be able
            to take the array of samples (size N * d) and return a column vector of
            log-probabilities (size N * 1).
        log_q: Callable[[np.ndarray], np.ndarray]
            log PDF of the auxiliary distribution. This function must be able
            to take the array of samples (size N * d) and return a column vector of
            log-probabilities (size N * 1).
        score_q: Callable[[np.ndarray], np.ndarray]
            gradient of log PDF of the auxiliary distribution. This function must be able
            to take the array of samples (size N * d) and return the corresponding array
            of gradients (size N * d).
        preconditioner: np.ndarray
            multiplier to apply in the formula
            TODO: better description
        sigma: float
            parameter sigma in the inverse multiquadratic kernel
        beta: float
            parameter beta in the inverse multiquadratic kernel
        use_all_values: bool
            if True (default), all evaluations of the integrand are used to calculate
            the integral, otherwise the diagonal elements are excluded from calculation
        clamp_qp: Optional[Tuple[float, float]]
            bounds to use to truncate the ratios of densities in the formula. If not provided,
            truncation is not applied
        min_log_p: Optional[float]
            if set, the values of the log PDF of the target distribution will be truncated below
            at this value
        """
        if min_log_p is not None:
            if isinstance(log_p, np.ndarray):
                self.log_p = np.clip(log_p, a_min=min_log_p, a_max=None)
            else:
                self.log_p = lambda x: np.clip(log_p(x), a_min=min_log_p, a_max=None)
        else:
            self.log_p = log_p

        self.log_q = log_q
        self.score_q = score_q

        assert isinstance(preconditioner, float), 'Only a scalar preconditioner is supported'
        self.preconditioner = preconditioner

        self.sigma = sigma
        self.beta = beta
        self.use_all_values = use_all_values
        self.clamp_qp = clamp_qp

    def eval(self, sample: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Calculate squared gradient-free kernel Stein discrepancy

        The expression for the integral is given by equation (4) in Fisher, Oates (2023)
        "Gradient-Free Kernel Stein Discrepancy", however note that the expression
        in the paper contain a typo: the first term is missing the minus sign. The correct
        expression along with the proof is provided on page 18 of the same paper.

        Parameters
        ----------
        sample: torch.Tensor
            grid points at which to evaluate the integrand. Dimensions: N * d
        weights: torch.Tensor
            weights corresponding of elements in ``sample``

        Returns
        -------
        float
            scalar value of squared kernel Stein discrepancy
        """
        return _calculate_integral(self.stein_matrix(sample), weights, self.use_all_values)

    def stein_matrix(self, sample: np.ndarray) -> np.ndarray:
        """Generate a matrix of integrand values for calculating kernel Stein discrepancy

        The expression for the integral is given by equation (4) in Fisher, Oates (2023)
        "Gradient-Free Kernel Stein Discrepancy", however note that the expression
        in the paper contain a typo: the first term is missing the minus sign. The correct
        expression along with the proof is provided on page 18 of the same paper.

        The matrix is formed by evaluating the integrand on the grid formed by the Cartesian
        product of ``samples`` by itself.

        Parameters
        ----------
        sample: np.ndarray
            grid points at which to evaluate the integrand. Dimensions: N * d

        Returns
        -------
        np.ndarray
            matrix of dimensions N * N of integrand values evaluated at points ``sample[i], sample[j]``
        """
        N, d = sample.shape  # number of samples, dimension

        if isinstance(self.score_q, np.ndarray):
            q_scores = self.score_q
        else:
            q_scores = self.score_q(sample)
        assert q_scores.ndim == 2
        assert q_scores.shape == sample.shape  # N * d matrix expected

        def evaluate_for_sample(func_or_array):
            if isinstance(func_or_array, np.ndarray):
                # if already an array, just use the values
                values = func_or_array
            else:
                # otherwise, evaluate for the sample
                values = func_or_array(sample)

            # the value is either a vector of size N or a column vector of size N * 1
            assert values.ndim in (1, 2)
            assert values.shape[0] == N
            if values.ndim == 2:
                assert values.shape[1] == 1

            return values

        log_p_sample = evaluate_for_sample(self.log_p)
        log_q_sample = evaluate_for_sample(self.log_q)

        score_prods = q_scores @ q_scores.T  # size N * N
        diffs = sample[:, np.newaxis, :] - sample  # size N * N * d
        score_diffs = q_scores[:, np.newaxis, :] - q_scores  # size N * N * d
        score_diff_prod = np.einsum('ijk,ijk->ij', diffs, score_diffs)  # size N * N
        dists_squared = np.einsum('ijk,ijk->ij', diffs, diffs)  # size N * N

        log_q_p = log_q_sample - log_p_sample  # size N or N * 1
        log_qp_diff = log_q_p.reshape(N, 1) + log_q_p.reshape(1, N)  # size N * N

        if self.clamp_qp is not None:
            log_qp_diff = np.clip(log_qp_diff, a_min=-self.clamp_qp[0], a_max=self.clamp_qp[1])
        coeff = np.exp(log_qp_diff)

        divisor = self.sigma ** 2 + self.preconditioner * dists_squared
        k = -4 * self.beta * (self.beta + 1) * self.preconditioner ** 2 * dists_squared / divisor ** (self.beta + 2)
        k_x = 2 * self.beta * self.preconditioner * (d + score_diff_prod) / divisor ** (self.beta + 1)
        k_xy = score_prods / divisor ** self.beta
        return coeff * (k + k_x + k_xy)
