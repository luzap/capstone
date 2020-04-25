#! /usr/bin/python3
import numpy as np
from scipy import linalg
import typing
from typing import Callable

# TODO Test against linear implementation

def calculate_sigma_points(mu, P, alpha=0.5, beta=2.0):
    """Calculate the van der Waal sigma points based on the state mean and
    covariance."""
    n = mu.ndim
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

    W_0 = 1.0/(2 * (n + lambda_))
    Wc = np.full(2*n + 1, W_0)
    Wm = np.full(2*n + 1, W_0)
    Wc[0] = lambda_ / (n + lambda_) + (1.0 - alpha ** 2 + beta)
    Wm[0] = lambda_ / (n + lambda_)

    return (Wc, Wm)


class UKFilter:
    # TODO Set timestep
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

        # Sigmas
        self.sigmas = None
        self.f_sigmas = []
        self.h_sigmas = []

        # Priors: after the update, mu and P become the posteriors
        self.x_prior = None
        self.P_prior = None

        # These are independent of the current mean, so we can calculate them
        # once
        self.Wc, self.Wm = calculate_weights(self.mu.ndim)

    def predict(self):
        self.sigmas = calculate_sigma_points(self.mu, self.P)

        self.f_sigmas = []
        for i, sigma in enumerate(self.sigmas):
            self.f_sigmas[i] = self.process_model(sigma, self.dt)

        self.x_prior, self.P_prior = self.unscented_transform(self.f_sigmas,
                                                              self.Q,
                                                              self.state_mean,
                                                              self.state_res)

        return (self.x_prior, self.P_prior)

    def update(self, z):
        n = self.x_prior.ndim
        self.h_sigmas = []

        for i, f_sigma in enumerate(self.f_sigmas):
            self.h_sigmas[i] = self.measurement(f_sigma)

        z_prior, Pz = self.unscented_transform(self.h_sigmas, self.R, self.meas_mean,
                                               self.meas_res)

        Pxz = np.zeros((n, z.ndim))
        for i in range(n):
            Pxz += self.Wc[i] * np.outer(self.f_sigmas[i] - self.x_prior,
                                         self.h_sigmas[i] - z_prior)

        K = np.dot(Pxz, np.inv(Pz))
        self.mu = self.x_prior + np.dot(K, z - z_prior)
        self.P = self.P_prior - np.dot(K, Pz).dot(K.T)
        return (self.mu, self.P)

    def unscented_transform(self, sigmas: np.ndarray, Q: np.ndarray,
                            mean_func: Callable[[np.ndarray], np.ndarray],
                            residual_func: Callable[[np.ndarray], np.ndarray]):
        k, n = sigmas.shape

        # TODO Check which row of this is the angle and then normalize
        x = np.zeros(n)
        x = mean_func(sigmas, self.Wm)

        P = np.zeros((n, n))
        for i in range(k):
            y = residual_func(sigmas[k], x)
            P += self.Wc[i] * np.outer(y, y)
        P += Q

        return (x, P)


# TODO The structure we are going for is [x y v \theta \dot{\theta}], so only
# the only the last column needs any angle correction
def prediction_average(sigma, Wm):
    pass

def prediction_residual(sigma, Wm):
    pass

def measurement_average(sigma, Wm):
    pass

def measurement_residual(sigma, Wm):
    pass

# Normalizing angles to the [0, 2pi] range
def normalize_angle(x):
    return x % (2 * np.pi)
