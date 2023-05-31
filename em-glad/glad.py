import numpy as np
from scipy.special import expit, log_expit
from collections import namedtuple
from copy import copy


class GLAD:
    def __init__(self, n_outer_steps=1000, n_inner_steps=1, lr=1e-3):
        self.n_outer_steps = n_outer_steps
        self.n_inner_steps = n_inner_steps
        self.lr = lr

    def fit(self, L, y):
        n, m = L.shape
        self.L = L
        self.L_tilde = 2 * L - 1

        # initialize parameters
        self.alpha = np.random.normal(loc=0, scale=1, size=m)
        self.logbeta = np.random.normal(loc=0, scale=1, size=n)
        self.q = np.ones((n, 1))

        # stats
        accuracy = []
        log_like = []
        alphas = []
        betas = []

        for _ in range(self.n_outer_steps):
            # EM-algorithm
            lower_bound = self._expectation()
            self._maximization()

            # collect stats
            accuracy.append(self.score(y))
            log_like.append(lower_bound)
            alphas.append(copy(self.alpha))
            betas.append(copy(np.exp(self.logbeta)))

        return alphas, betas, log_like, accuracy

    def score(self, y):
        """choose label basing on pesterior distribution of latent variables"""
        ans = np.mean((self.q > 0.5) != y[:, None])
        if ans < 0.5:
            ans = np.mean((self.q > 0.5) == y[:, None])

        return ans

    def _expectation(self):
        self.q = self._posterior()
        return self._lower_bound(self.q, self.alpha, np.exp(self.logbeta))

    def _maximization(self):
        for _ in range(self.n_inner_steps):
            grad_alpha, grad_logbeta = self._grads(
                self.q, self.alpha, np.exp(self.logbeta))
            self.alpha += self.lr * grad_alpha
            self.logbeta += self.lr * grad_logbeta

    def _posterior(self):
        """posterior distribution of latent variables"""

        return expit(-np.exp(self.logbeta) * (self.L_tilde @ self.alpha))[:, None]

    def _lower_bound(self, q, alpha, beta):
        """(it's not a) lower bound of expected log likelihood (it's assymptotically equiv)"""

        # matrix M_{ij} = alpha_j * beta_i of shape (n_problems, n_experts)
        alpha_beta = np.einsum('i,j->ij', beta, alpha)

        return np.sum(alpha_beta * (-q * self.L_tilde + self.L) + log_expit(-alpha_beta))

    def _grads(self, q, alpha, beta):
        """gradients wrt alpha and logbeta"""
        # matrix M_{ij} = alpha_j * beta_i of shape (n_problems, n_experts)
        alpha_beta = np.einsum('i,j->ij', beta, alpha)

        # shared operand for both grad_alpha and grad_logbeta
        tmp = -q * self.L_tilde + self.L - expit(alpha_beta)

        # gradients wrt alpha
        grad_alpha = np.einsum('ij,i->j', tmp, beta)

        # gradients wrt alpha
        grad_logbeta = np.einsum('ij,j', tmp, alpha) * beta

        res = namedtuple('grads', ['alpha', 'logbeta'])
        return res(grad_alpha, grad_logbeta)
