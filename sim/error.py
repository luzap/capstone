#!/usr/bin/python3
# TODO Eventually get rid of this dependency
import sympy
import functools
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import optimize, integrate

def get_data(mu, Sigma, sample_num=10000):
    """Sample the normal distribution defined by the given mu and Sigma. The default sample number
    may take a second or two to generate, but seems to give fairly close results."""
    sums = np.zeros(sample_num)
    for i, _ in enumerate(sums):
        sample = np.random.multivariate_normal(mu, Sigma)
        sums[i] = np.dot(sample, sample.T)
    return sums


def solve_chi_saddlepoint(mu, Sigma):
    """Compute the saddlepoint approximation for the generalized chi square distribution given a mean and a covariance matrix. Currently has two different ways of solving:
        1. If the mean is close to zero, the system can be solved symbolically.
        2. TODO If the mean is further away from zero, the system becomes more complex, and thus is more difficult to solve. Instead, we will either be using a non-linear solver, or the Taylor series expansion around the squared magnitude of the mean, as that is where we think it should be centered."""
    P = None
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    if (eigenvectors == np.diag(eigenvalues)).all():
        print("Diagonal matrix")
        P = np.eye(len(mu))
    else:
        print("Non-diagonal matrix")
        P = eigenvectors.T
    Sigma_12 = np.linalg.cholesky(Sigma)
    b = P @ Sigma_12 @ mu
    x, t = sympy.symbols("x t")

    # Cumulant function (symbolic computation)
    K = 0
    for i, l in enumerate(eigenvalues):
        K += (b[i] * l)/(1 - 2 * t * l) - 1/2 * sympy.ln(1 - 2 * l * t)
    Kp = sympy.diff(K, t)
    Kpp = sympy.diff(K, t, t)

    # If close to zero, the computation can be numeric
    if np.linalg.norm(b) < 0.01:
        s_hat = sympy.solve(Kp - x, t)[0]
        f = 1 / sympy.sqrt(2 * sympy.pi * Kpp.subs(t, s_hat)) * sympy.exp(K.subs(t, s_hat) - s_hat * x)
        return sympy.utilities.lambdify(x, f)
    else:
        saddles = sympy.utilities.lambdify(x, Kp)
        dsaddles = sympy.utilites.lambdify(x, Kpp)

        # TODO What's the range?
        xs = np.arange(0.1, 100, 0.1)
        # TODO The estimate here is wrong. Try something by visual inspection, or do by root finding
        sols = optimize.fsolve(saddles - xs, np.dot(b, b))
        # TODO What are the dimensions here
        saddlepoints = np.zeros((2, len(xs)))
        i = 0
        for sol in sols:
            if dsaddles(sol) > 0:
                saddlepoints[i] = np.array([xs, sol])
                i += 1


        # TODO How are we defininig this function so that it would look up the
        # appropriate saddlepoint
        def approx(x):
            return

        # TODO Test if this works
        c = integrate.quad(approx, 0, np.inf)

        return (approx, c)


def sample_distribution(fn, mu, sample_num):
    """Perform rejection sampling with a distribution fn. The `mu` is needed to
    determine the interval over which we're sampling."""
    xs = np.zeros(sample_num)
    # TODO We need to get an upper and lower bound. We can do this by interating
    # over the interval via binary search or we can set naive bounds
    uniform = functools.partial(np.random.uniform, low = np.dot(mu, mu)/4,
                                high = 7/4 * np.dot(mu,mu), size=(sample_num, 2))
    X = uniform()

    # There is no good reason to prefer having an indexed loop, considering we don't
    # a priori know much about the problem
    while True:
        # We're doing rejection sampling, so we take col 1: x; col 2: y,
        # extracting all of the entries that satisfy the requirement, and then
        # transposing that into a 1D array
        samples = fn[fn(X[:, 0]) < X[:, 1]][:, 0].T
        # Since we're only ever looking for `sample_num` samples, we need to reject
        # everything that goes over that number. Therefore, we clamp the last index
        # to the number of samples that we want
        next_batch_index = min(sample_num, current_samples+len(samples))
        xs[next_batch_index - len(samples): next_batch_index] = samples[:next_batch_index]

        if next_batch_index == sample_num:
            break

        X = uniform()

    return xs


if __name__ == "__main__":
    plt.style.use("seaborn-ticks")

    fig, axes = plt.subplots(1, 1)
    fig.tight_layout()
    x_col = np.arange(0.1, 15, 0.1)

    mu2 = np.array([0, 0])
    Sigma2 = np.eye(len(mu2))
    x_col = np.arange(0.1, 15, 0.1)

    f2 = solve_chi_saddlepoint(mu2, Sigma2)
    data2 = get_data(mu2, Sigma2)

    axes[0].plot(x_col, f2(x_col), label="saddlepoint approx.")
    axes[0].hist(data2, bins=50, density=True, label="MC histogram")
    axes[0].set_title("Error distribution in {}D".format(len(mu2)))
    axes[0].set_xlabel("Magnitude (m)")
    axes[0].set_ylabel("Probability")
    axes[0].legend()


    plt.show()
