#! /usr/bin/python3
import numpy as np
import typing
from typing import Callable


class UnscentedKalmanFilter:

    # TODO What's needed to seed this
    def __init__(self, mean, covariance):
        if self.mean is not None:
            self.mean = mean
        else:
            raise Exception("Mean cannot be null!")
        if self.covariance is None:
            self.covariance = np.eye(self.mean.ndim)
        else:
            self.covariance = covariance

        self.sigma, self.Wc, self.Wm = self.gen_sigma_points()


    def gen_sigma_points(self, beta=2, alpha=0.5):
        """Generate the Van der Waals' sigma points, mean and covariance weights
        for the current system mean and covariance.
        """

        n = self.mean.ndim
        sigmas = np.empty(2*n+1)

        kappa = 3 - mean.ndim
        lambda_ = (alpha ** 2) * (n + kappa) - n
        sigma[0] = self.mean

        # TODO Make sure this produces column vectors and not row vectors (or
        # vice versa)
        # TODO Is the square root of a matrix defined elementwise, or is there
        # an alternative def?
        c = n + lambda_
        for i in range(1, n+1):
            sigma[i] = self.mean + (np.sqrt(n + lambda_)
                         * np.sqrt(self.covariance)))[i]

        for i in range(n+1, 2 * n):
            sigma[i] = (self.mean - (np.sqrt(n + lambda_)
                         * np.sqrt(self.covariance)))[n-i]

        W_0 = 1 / (2*(n + lambda_))
        Wm = np.full(2*n+1, W_0)
        Wc = np.full(2*n+1, W_0)
        Wc[0] = Wc[0] + 1 - alpha ** 2 + beta
        Wm[0] = lambda_ / (n + lambda_)

        return (sigmas, Wc, Wm)


    def predict(self):
        """TODO Document
        """
        self.Y = self.f(self.sigma)
        self.mean = np.dot(self.Wm, self.sigma)
        # TODO Implementing this is a little bit trickier
        self.covariance = None

    def update(self, measurement):
        self.Z = None
        self.mean_z = None
        y = measurement - self.mean_z
        self.measurement_cov = None
        self.kalman_gain = None
        self.x += self.kalman_gain @ y
        self.covariance -= self.kalman_gain @ self.measurement_cov
        @ self.kalman_gain.T

    def f(self, args):
        return 0

    def h(self, args):
        return 0
