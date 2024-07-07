from typing import Callable, Optional, Tuple

from numpy import diagonal
import torch
from torch.linalg import inv, det, multi_dot

from steintorch.divergence.base import Divergence


def _calculate_integral(
        integrand: torch.Tensor,
        weights: Optional[torch.Tensor],
        use_all_values: bool = True,
    ) -> torch.Tensor:
    """Calculate integral given a matrix of integrand values evaluated on a grid

    Parameters
    ----------
    integrand: torch.Tensor
        matrix of integrand values of size N * N
    weights: torch.Tensor
        vector of weights for points along each dimension of the integrand matrix
    use_all_values: bool
        if True, use all integrand values, otherwise ignore the diagonal elements

    Returns
    -------
    torch.Tensor
        scalar value of the integral
    """
    # make sure the integrand matrix is two-dimensional square
    assert integrand.ndim == 2
    assert integrand.shape[0] == integrand.shape[1]
    N = integrand.shape[0]

    if weights is None:
        if use_all_values:
            return torch.mean(integrand)
        else:
            # use only off-diagonal elements
            return 1 / (N * (N - 1)) * (torch.sum(integrand) - torch.sum(torch.diag(integrand)))
    else:
        weights = weights / sum(weights)  # normalise the weights
        if use_all_values:
            return multi_dot((weights.unsqueeze(0), integrand, weights.unsqueeze(1))).flatten()
        else:
            # use only off-diagonal elements
            quad_form = multi_dot((weights.unsqueeze(0), integrand, weights.unsqueeze(1))).flatten()
            diagonal_terms = torch.dot(weights.pow(2), torch.diag(integrand))
            return quad_form - diagonal_terms


class KSD(Divergence):
    """Kernel Stein discrepancy"""

    def __init__(self, kernel, preconditioner=torch.Tensor([1])):

        # TODO: Possibly add distribution to __init__ utilise get_score and make score_func optional argument in subsequent code
        # TODO: Different kernels and Stein operators, currently just IMQ with Langevin.

        self.preconditioner = preconditioner

        if kernel is not None:
            if kernel.is_stein is False:
                raise ValueError("Given kernel object is not a Stein kernel")
        self.kernel = kernel

    def eval(self, sample, score, weights=None, preconditioner=None, beta=0.5, V_statistic=True):
        N, d = sample.size()  # number of samples, dimension
        stein_mat = self.stein_matrix(sample=sample, score=score, preconditioner=preconditioner, beta=beta)
        return _calculate_integral(stein_mat, weights, V_statistic)

    def stein_matrix(self, sample, score, preconditioner=None, beta=0.5, no_grad=False):
        N, d = sample.size()  # number of samples, dimension

        if preconditioner is None:
            preconditioner = self.preconditioner

        if type(score) is torch.Tensor:
            scores = score
        else:
            scores = score(sample)

        if no_grad is True:
            sample = sample.detach()
            scores = scores.detach()

        PRECON_SIZE = preconditioner.size()
        if len(PRECON_SIZE) == 1 and PRECON_SIZE[0] == 1:
            # scalar preconditioner
            score_prods = (scores.unsqueeze(1) * scores)
            dists = torch.cdist(sample, sample).pow(2)
            diffs = sample.unsqueeze(1) - sample
            score_diffs = scores.unsqueeze(1) - scores
            score_diff_prod = torch.bmm(diffs.view(N * N, 1, d), score_diffs.view(N * N, d, 1)).reshape(N, N)

            k = -4 * beta * (beta + 1) * preconditioner.pow(2) * dists / ((1 + dists * preconditioner).pow(beta + 2))
            k_x = 2 * beta * (d * preconditioner + score_diff_prod * preconditioner) / ((1 + dists * preconditioner).pow(beta + 1))
            k_xy = score_prods.sum(dim=2) / ((1 + dists * preconditioner).pow(beta))
            output = (k + k_x + k_xy)

        elif len(PRECON_SIZE) == 2:
            # full matrix preconditioner
            sqrt_precon = torch.linalg.cholesky(preconditioner)
            trace_precon = torch.trace(preconditioner)

            tsample = torch.matmul(sample, sqrt_precon)
            ttsample = torch.matmul(sample, preconditioner)

            score_prods = (scores.unsqueeze(1) * scores)
            dists2 = torch.cdist(ttsample, ttsample).pow(2)
            dists = torch.cdist(tsample, tsample).pow(2)
            diffs = (sample.unsqueeze(1) - sample).reshape(N * N, d)
            tdiffs = torch.matmul(diffs, sqrt_precon)
            score_diffs = (scores.unsqueeze(1) - scores).reshape(N * N, d)
            tscore_diffs = torch.matmul(score_diffs, sqrt_precon)
            score_diff_prod = torch.bmm(tdiffs.view(N * N, 1, d), tscore_diffs.view(N * N, d, 1)).reshape(N, N)

            k = -4 * beta * (beta + 1) * dists2 / ((1 + dists).pow(beta + 2))
            k_x = 2 * beta * (trace_precon + score_diff_prod) / ((1 + dists).pow(beta + 1))
            k_xy = score_prods.sum(dim=2) / ((1 + dists).pow(beta))
            output = (k + k_x + k_xy)
        else:
            # vector precondioner
            raise NotImplementedError

        return output


class GradientFreeKSD(KSD):
    """Gradient-free kernel Stein discrepancy"""

    def __init__(self, kernel, preconditioner=torch.Tensor([1])):
        super().__init__(kernel, preconditioner)

    def eval(
            self,
            sample: torch.Tensor,
            log_p: Callable[[torch.Tensor], torch.Tensor],
            log_q: Callable[[torch.Tensor], torch.Tensor],
            score_q: Callable[[torch.Tensor], torch.Tensor],
            preconditioner: torch.Tensor = None,
            sigma: float = 1.0,
            beta: float = 0.5,
            weights: torch.Tensor = None,
            V_statistic: bool = True,
            clamp_qp: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            min_log_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate squared gradient-free kernel Stein discrepancy

        The expression for the integral is given by equation (4) in Fisher, Oates (2023)
        "Gradient-Free Kernel Stein Discrepancy", however note that the expression 
        in the paper contain a typo: the first term is missing the minus sign. The correct
        expression along with the proof is provided on page 18 of the same paper.

        Parameters
        ----------
        sample: torch.Tensor
            grid points at which to evaluate the integrand. Dimensions: N * d
        log_p: Callable[[torch.Tensor], torch.Tensor]
            log PDF of the target distribution. This function must be able
            to take the array of samples (size N * d) and return a column vector of 
            log-probabilities (size N * 1).
        log_q: Callable[[torch.Tensor], torch.Tensor]
            log PDF of the auxiliary distribution. This function must be able
            to take the array of samples (size N * d) and return a column vector of 
            log-probabilities (size N * 1).
        score_q: Callable[[torch.Tensor], torch.Tensor]
            gradient of log PDF of the auxiliary distribution. This function must be able
            to take the array of samples (size N * d) and return the corresponding array
            of gradients (size N * d).
        preconditioner: torch.Tensor
            multiplier to apply in the formula
            TODO: better description
        sigma: float
            parameter sigma in the inverse multiquadratic kernel
        beta: float
            parameter beta in the inverse multiquadratic kernel
        weights: torch.Tensor
            weights corresponding of elements in ``sample``
        V_statistic: bool
            if True (default), all evaluations of the integrand are used to calculate
            the integral, otherwise the diagonal elements are excluded from calculation
        clamp_qp: Optional[Tuple[torch.Tensor, torch.Tensor]]
            bounds to use to truncate the ratios of densities in the formula. If not provided,
            truncation is not applied
        min_log_p: Optional[torch.Tensor]
            if set, the values of the log PDF of the target distribution will be truncated below
            at this value

        Returns
        -------
        torch.Tensor
            scalar value of squared kernel Stein discrepancy
        """
        if min_log_p is not None:
            if type(log_p) is torch.Tensor:
                new_log_p = torch.clamp(log_p, min=min_log_p)
            else:
                new_log_p = lambda x: torch.clamp(log_p(x), min=min_log_p)
        else:
            new_log_p = log_p

        # calculate the matrix of integrand values on the grid
        stein_mat = self.stein_matrix(sample=sample,
                                      log_p=new_log_p,
                                      log_q=log_q,
                                      score_q=score_q,
                                      preconditioner=preconditioner,
                                      sigma=sigma,
                                      beta=beta,
                                      clamp_qp=clamp_qp)

        return _calculate_integral(stein_mat, weights, V_statistic)

    def stein_matrix(
            self,
            sample: torch.Tensor,
            log_p: Callable[[torch.Tensor], torch.Tensor],
            log_q: Callable[[torch.Tensor], torch.Tensor],
            score_q: Callable[[torch.Tensor], torch.Tensor],
            preconditioner: torch.Tensor = None,
            sigma: float = 1.0,
            beta: float = 0.5,
            clamp_qp: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            no_grad: bool = False) -> torch.Tensor:
        """Generate a matrix of integrand values for calculating kernel Stein discrepancy

        The expression for the integral is given by equation (4) in Fisher, Oates (2023)
        "Gradient-Free Kernel Stein Discrepancy", however note that the expression 
        in the paper contain a typo: the first term is missing the minus sign. The correct
        expression along with the proof is provided on page 18 of the same paper.

        The matrix is formed by evaluating the integrand on the grid formed by the Cartesian
        product of ``samples`` by itself.

        Parameters
        ----------
        sample: torch.Tensor
            grid points at which to evaluate the integrand. Dimensions: N * d
        log_p: Callable[[torch.Tensor], torch.Tensor]
            log PDF of the target distribution. This function must be able
            to take the array of samples (size N * d) and return a column vector of 
            log-probabilities (size N * 1).
        log_q: Callable[[torch.Tensor], torch.Tensor]
            log PDF of the auxiliary distribution. This function must be able
            to take the array of samples (size N * d) and return a column vector of 
            log-probabilities (size N * 1).
        score_q: Callable[[torch.Tensor], torch.Tensor]
            gradient of log PDF of the auxiliary distribution. This function must be able
            to take the array of samples (size N * d) and return the corresponding array
            of gradients (size N * d).
        preconditioner: torch.Tensor
            multiplier to apply in the formula
            TODO: better description
        sigma: float
            parameter sigma in the inverse multiquadratic kernel
        beta: float
            parameter beta in the inverse multiquadratic kernel
        clamp_qp: Optional[Tuple[torch.Tensor, torch.Tensor]]
            bounds to use to truncate the ratios of densities in the formula. If not provided,
            truncation is not applied.
        no_grad: bool
            detach the calculated values from the PyTorch graph. Default: False

        Returns
        -------
        torch.Tensor
            matrix of dimensions N * N of integrand values evaluated at points ``sample[i], sample[j]``
        """
        preconditioner = preconditioner or self.preconditioner
        if preconditioner.ndim != 1 or preconditioner.shape[0] != 1:
            raise NotImplementedError('Only a scalar preconditioner is supported')

        N, d = sample.size()  # number of samples, dimension

        if type(score_q) is torch.Tensor:
            q_scores = score_q
        else:
            q_scores = score_q(sample)
        assert q_scores.ndim == 2
        assert q_scores.shape == sample.shape

        if type(log_q) is torch.Tensor:
            log_q_sample = log_q
        else:
            log_q_sample = log_q(sample)
        assert log_q_sample.ndim == 2
        assert log_q_sample.shape[0] == N
        assert log_q_sample.shape[1] == 1

        if type(log_p) is torch.Tensor:
            log_p_sample = log_p
        else:
            log_p_sample = log_p(sample)
        assert log_p_sample.ndim == 2
        assert log_p_sample.shape[0] == N
        assert log_p_sample.shape[1] == 1

        if no_grad:
            log_p_sample = log_p_sample.detach()
            log_q_sample = log_q_sample.detach()
            q_scores = q_scores.detach()

        score_prods = q_scores @ q_scores.T  # size N * N
        diffs = sample.unsqueeze(1) - sample  # size N * N * d
        score_diffs = q_scores.unsqueeze(1) - q_scores  # size N * N * d
        score_diff_prod = torch.bmm(
            diffs.view(N * N, 1, d),
            score_diffs.view(N * N, d, 1),
        ).reshape(N, N)  # size N * N
        dists_squared = torch.cdist(sample, sample).pow(2)  # size N * N

        log_q_p = log_q_sample - log_p_sample  # size N * 1
        log_qp_diff = (log_q_p.unsqueeze(1) + log_q_p).squeeze()  # size N * N

        if clamp_qp is not None:
            clamped_diff = torch.clamp(log_qp_diff, min=-clamp_qp[0], max=clamp_qp[1])
            coeff = clamped_diff.exp()
        else:
            coeff = log_qp_diff.exp()

        divisor = (sigma ** 2 + preconditioner * dists_squared)
        k = -4 * beta * (beta + 1) * preconditioner.pow(2) * dists_squared / divisor.pow(beta + 2)
        k_x = 2 * beta * preconditioner * (d + score_diff_prod) / divisor.pow(beta + 1)
        k_xy = score_prods / divisor.pow(beta)
        return coeff * (k + k_x + k_xy)
