#! /usr/bin/python3
import numpy as np
from scipy import linalg
import typing
import functools
from typing import Callable

# TODO Process model
# TODO Factor out related measurements into class
# TODO Test against linear implementation
# TODO Design the noise functions R - we don't a priori know the measurement
# inaccuracies, so let's take it as the identity matrix?

def compute_continuous_white_noise(process_model):
    """We are taking the white noise that characterises the error in the model
    to be of a continous variety. Since we have no empirical derivation of the
    error, we shall instead follow a theoretical one, wherein the discrete-time
    white noise model can be given by the integral Q = \int_0^{\Delta t} f(t)
    Q_c f(t)^T \dd t, where Q_c is the continuous time noise and f(t) is our
    process model.

    Via some theory of stochastic processes, we know that we can approximate Q_c
    with the following nxn matrix
                        [0   ...    0  ]
                        [... ...   ... ]
                        [0   ... \Phi_s],
    where \Phi_s is the spectral density of the white noise. To get this, we
    shall simply evaluate the integral at the given time and use the given
    matrix.

    In practice, we often don't know the spectral noise of the problem, so it
    becomes an issue of fine-tuning, which we can also try to do.
    """
    import sympy
    dt, phi = sympy.symbols(r'\Delta{t} \Phi_s')
    continuous_noise = sympy.Matrix([[0, 0, 0], [0,0,0], [0,0,1]]) * phi
    Q = sympy.integrate(process_model * continuous_noise * process_model.T,
                        (dt, 0, dt))
    return Q

def calculate_sigma_points(mu, P, alpha=0.5, beta=2.0):
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

# Individually pass the sigma points through the fn
# Note that sigmas should be refreshed with every predict

def fx(sigma, dt, **args):
    return 0

def hx(sigma, **args):
    return 0

class UKFilter:

    def __init__(self, mu, P, Q, R):
        # State variables
        self.mu = mu
        self.P = P
        self.Q = Q

        # Measurement variables
        self.R = R

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
        n = self.mu.ndim
        self.sigmas = calculate_sigma_points(self.mu, self.P)

        self.f_sigmas = []
        for i in range(n):
            self.f_sigmas[i] = prediction_ut(self.sigmas,
                                             self.Wm,
                                             self.Wc,
                                             self.Q)

        self.x_prior, self.P_prior = prediction_ut(self.f_sigmas,
                                                   self.Wm, self.Wc,
                                                   self.Q)

        return (self.x_prior, self.P_prior)

    def update(self, z):
        n = self.x_prior.ndim
        self.h_sigmas = []

        for i in range(n):
            self.h_sigmas[i] = hx(self.f_sigmas[i])

        z_prior, Pz = measurement_ut(self.h_sigmas,
                                     self.Wm,
                                     self.Wc,
                                     self.R)

        Pxz = np.zeros((n, z.ndim))
        for i in range(n):
            Pxz += self.Wc[i] * np.outer(self.f_sigmas[i] - self.x_prior,
                                         self.h_sigmas[i] - z_prior)


        K = np.dot(Pxz, np.inv(Pz))
        self.mu = self.x_prior + np.dot(K, z - z_prior)
        self.P = self.P_prior - np.dot(K, Pz).dot(K.T)
        return (self.mu, self.P)


    @classmethod
    def unscented_transform(sigmas: np.ndarray, Wm: np.ndarray, Wc: np.ndarray,
                            mean_function: Callable[np.ndarray, np.ndarray],
                            residual_function: Callable[np.ndarray, np.ndarray]):
        k, n = sigmas.shape

        # TODO Check which row of this is the angle and then normalize
        x = np.zeros(n)
        x = mean_function(sigmas, Wm)

        P = np.zeros((n, n))
        for i in range(k):
            y = residual_function(sigmas[k], x)
            P += Wc[i] * np.outer(y, y)

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

prediction_ut = functools.partial(unscented_transform,
                                  mean_function=prediction_average,
                                  residual_function=prediction_residual)

measurement_ut = functools.partial(unscented_transform,
                                   mean_function=measurement_residual,
                                   residual_function=measurement_residual)

# Normalizing angles to the [0, 2pi] range
def normalize_angle(x):
    return x % (2 * np.pi)
