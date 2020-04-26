#! /usr/bin/python3
import numpy as np
from scipy import linalg
import typing
from typing import Callable

def calculate_sigma_points(mu, P, alpha=0.5, beta=2.0):
    """Calculate the van der Waal sigma points based on the state mean and
    covariance."""
    n, = mu.shape
    lambda_ = (alpha ** 2) * 3 - n
    sigmas = np.zeros((2*n+1, n))
    U = linalg.cholesky((n+lambda_)*P)

    sigmas[0] = mu
    for i in range(n):
        sigmas[i+1]     = mu + U[i]
        sigmas[n+i+1]   = mu - U[i]
    return sigmas

def calculate_weights(n, alpha=0.5, beta=2.0):
    lambda_ = (alpha ** 2) * 3 - n

    W_0 = 0.5/(n + lambda_)
    Wc = np.full(2*n + 1, W_0)
    Wm = np.full(2*n + 1, W_0)
    Wc[0] = lambda_ / (n + lambda_) + (1.0 - alpha ** 2 + beta)
    Wm[0] = lambda_ / (n + lambda_)

    return (Wc, Wm)


class UKFilter:
    # TODO Add type annotations
    def __init__(self, mu, P, Q, R, dt, process_model, measurement_function,
                 state_mean, state_residual, measurement_mean, measurement_residual):
        # State variables
        self.mu = mu
        self.P = P
        self.Q = Q
        self.dt = dt

        # State functions
        self.process_model = process_model
        self.state_mean = state_mean
        self.state_res = state_residual

        # Measurement variables
        self.R = R

        # Measurement functions
        self.measurement = measurement_function
        self.meas_mean = measurement_mean
        self.meas_res = measurement_residual

        # Priors: after the update, mu and P become the posteriors
        self.x_prior = None
        self.P_prior = None

        # These are independent of the current mean, so we can calculate them
        # once
        self.Wc, self.Wm = calculate_weights(self.mu.shape[0])

    def predict(self):
        # Create the sigma points dependent on the current mean
        self.sigmas = calculate_sigma_points(self.mu, self.P)

        # Pass the sigma points through the process model to sample
        # the current mean and covariance
        self.f_sigmas = np.zeros((len(self.sigmas), len(self.mu)))
        for i, sigma in enumerate(self.sigmas):
            self.f_sigmas[i] = self.process_model(sigma, self.dt)

        self.x_prior, self.P_prior = self.unscented_transform(self.f_sigmas,
                                                              self.Q,
                                                              self.state_mean,
                                                              self.state_res)
        return (self.x_prior, self.P_prior)

    def update(self, z):
        if np.isscalar(z):
            z = np.array([z])

        # pylint: disable=E1136
        h_sigmas = np.zeros((self.f_sigmas.shape[0], len(z)))
        for i, f_sigma in enumerate(self.f_sigmas):
            h_sigmas[i] = self.measurement(f_sigma)
        print(h_sigmas)

        z_prior, Pz = self.unscented_transform(h_sigmas, self.R, self.meas_mean,
                                               self.meas_res)

        # pylint: disable=E1136
        Pxz = np.zeros((self.f_sigmas.shape[1], h_sigmas.shape[1]))
        for i in range(self.f_sigmas.shape[0]):
            Pxz += self.Wc[i] * np.outer(
                self.state_res(self.f_sigmas[i], self.x_prior),
                self.meas_res(h_sigmas[i], z_prior)
            )

        K = Pxz @ np.linalg.inv(Pz)
        y = self.meas_res(z, z_prior)

        self.mu = self.x_prior + K @ y
        self.P = self.P_prior - K @ Pz @ K.T
        print("P\n", self.P)

    def unscented_transform(self, sigmas: np.ndarray, Q: np.ndarray,
                            mean_func: Callable[[np.ndarray], np.ndarray],
                            residual_func: Callable[[np.ndarray], np.ndarray]):
        k, n = sigmas.shape

        x = mean_func(self.Wm, sigmas)

        P = np.zeros((n, n))
        for i in range(k):
            y = residual_func(sigmas[i], x)
            P += self.Wc[i] * np.outer(y, y)
        P += Q

        return (x, P)

# TODO The structure we are going for is [x y v \theta \dot{\theta}], so only
# the only the last column needs any angle correction

# Normalizing angles to the [0, 2pi] range
def normalize_angle(x):
    return x % (2 * np.pi)
