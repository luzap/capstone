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


def predict(mu, P, Wm, Wc, Q):
    n = mu.ndim
    sigmas = calculate_sigma_points(mu, P)
    Wc, Wm = calculate_weights(n)

    f_sigmas = []
    for i in range(n):
        f_sigmas[i] = prediction_ut(sigmas, Wm, Wc, Q)

    x_prior, P_prior = prediction_ut(f_sigmas, Wm, Wc, Q)

    return (x_prior, P_prior, sigmas, Wc, Wm)

def update(z, x_prior, P_prior, f_sigmas, Wc, Wm, R):
    n = x_prior.ndim
    h_sigmas = []

    for i in range(n):
        h_sigmas[i] = hx(f_sigmas[i])

    z_prior, Pz = measurement_ut(h_sigmas, Wm, Wc, R)

    Pxz = np.zeros((n, z.ndim))
    for i in range(n):
        Pxz += Wc[i] * np.outer(f_sigmas[i] - x_prior, h_sigmas[i] - z_prior)

    K = np.dot(Pxz, np.inv(Pz))
    x = x_prior + np.dot(K, z - z_prior)
    P = P_prior - np.dot(K, Pz).dot(K.T)
    return (x, P)

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
