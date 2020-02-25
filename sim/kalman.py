#! /usr/bin/python3
import numpy as np
from scipy import linalg
import typing
from typing import Callable


class UnscentedKalmanFilter:

    # TODO What's needed to seed this
    def __init__(self, mean, covariance):
        if mean is not None:
            self.mean = mean
        else:
            raise Exception("Mean cannot be null!")
        if covariance is None:
            self.covariance = np.eye(self.mean.ndim)
        else:
            self.covariance = covariance


    def gen_sigma_points(self, beta=2, alpha=0.5):
        """Generate the Van der Waals' sigma points, mean and covariance weights
        for the current system mean and covariance.
        """

        n = self.mean.ndim
        sigmas = np.zeros((2*n+1, n))
        kappa = 3 - n
        lambda_ = (alpha ** 2) * (n + kappa) - n
        sigmas[0] = self.mean

        # Because we need to take the "square root" of a matrix, there is leeway
        # in how to define this operation. We can choose to define it in
        # a manner where we want S*S^T to bring us back to the original matrix
        U = linalg.cholesky((n + lambda_) * self.covariance)

        for i in range(n):
            sigmas[i+1] = self.mean + U[i]
            sigmas[n+i+1] = self.mean - U[i]

        W_0 = 1 / (2*(n + lambda_))
        Wm = np.full(2*n+1, W_0)
        Wc = np.full(2*n+1, W_0)
        Wc[0] = Wc[0] + 1 - alpha ** 2 + beta
        Wm[0] = lambda_ / (n + lambda_)

        return (sigmas, Wc, Wm)


    def predict(self):
        """TODO Document
        """
        self.sigmas, self.Wc, self.Wm = self.gen_sigma_points()

        self.Y = self.f(self.sigma, dt)
        self.mean = np.dot(self.Wm, self.sigma)
        

    def update(self, measurement):
        self.Z = None
        self.mean_z = None
        y = measurement - self.mean_z
        self.measurement_cov = None
        self.kalman_gain = None
        self.x += self.kalman_gain @ y
        self.covariance -= self.kalman_gain @ self.measurement_cov @ self.kalman_gain.T

    def f(self, sigmas, dt):
        return 0

    def h(self, dt):
        return 0


if __name__ == "__main__":
    ukf = UnscentedKalmanFilter(np.array([0]),None) 
