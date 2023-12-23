import numpy as np
from scipy.special import expit


class BaseLoss:
    def func(self, w):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        bias = w[0]
        weights = w[1:]
        m = (X @ weights + bias) * y
        return np.logaddexp(np.zeros_like(m), -m).mean() + self.l2_coef * (weights @ weights)

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        bias = w[0]
        weights = w[1:]
        m = (X @ weights + bias) * y
        sigma = expit(m)
        ans0 = y @ (sigma - 1) / X.shape[0]
        ans1 = X.T @ (y * (sigma - 1)) / X.shape[0] + self.l2_coef * weights * 2
        return np.r_[ans0, ans1]
