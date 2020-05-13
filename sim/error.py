#!/usr/bin/python3
# TODO Eventually get rid of this dependency
import sympy
import symengine as sym
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
        P = np.eye(len(mu))
    else:
        print("Non-diagonal")
        P = eigenvectors.T
    Sigma_12 = np.linalg.cholesky(Sigma)
    b = P @ Sigma_12 @ mu

    x = sym.Symbol("x")
    t = sym.Symbol("t")

    # Cumulant function
    K = 0
    for i, l in enumerate(eigenvalues):
        K += (t * b[i] ** 2 * l)/(1 - 2 * t * l) - 1/2 * sym.log(1 - 2 * l * t)

    Kp = sym.diff(K, t)
    Kpp = sym.diff(K, t, t)

    roots = sym.lib.symengine_wrapper.solve(sym.Eq(Kp, x), t).args
    if len(roots) > 1:
        for expr in roots:
            trial = Kpp.subs(t, expr).subs(x, np.dot(b,b))
            if trial >= 0.0:
                s_hat = expr
    else:
        s_hat = roots[0]

    f = 1 / sym.sqrt(2 * sym.pi * Kpp.subs(t, s_hat)) * sym.exp(K.subs(t, s_hat) - s_hat * x)
    fp = sym.Lambdify(x, f)
    c = integrate.quad(fp, 0, np.inf)[0]

    return lambda x: 1/c * fp(x)


def sample_distribution(fn, mu, sample_num):
    """Perform rejection sampling with a distribution fn. The `mu` is needed to
    determine the interval over which we're sampling.

    Based on pages 83-84 of "Data Reduction and Error Analysis for the Physical
    Sciences".
    """
    xs = np.zeros(sample_num)
    # TODO We need to get an upper and lower bound. We can do this by interating
    # over the interval via binary search or we can set naive bounds
    uniform = functools.partial(np.random.uniform, low = np.dot(mu, mu)/4,
                                high = 7/4 * np.dot(mu,mu), size=(sample_num*2, 2))
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

    mu2 = np.array([10, 5])
    Sigma2 = np.eye(len(mu2))

    f2 = solve_chi_saddlepoint(mu2, Sigma2)
    data2 = get_data(mu2, Sigma2)
    x_col = np.arange(0.1, np.amax(data2), 0.1)


    axes.plot(x_col, f2(x_col), label="saddlepoint approx.")
    axes.hist(data2, bins=50, density=True, label="MC histogram")
    axes.set_title("Error distribution in {}D".format(len(mu2)))
    axes.set_xlabel("Magnitude (m)")
    axes.set_ylabel("Probability")
    axes.legend()


    plt.show()
