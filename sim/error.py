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
        1. If the mean is close to zero, the system can be solved symbolically."""
    P = None
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    if (eigenvectors == np.diag(eigenvalues)).all():
        P = np.eye(len(mu))
    else:
        P = eigenvectors.T
    Sigma_12 = np.linalg.cholesky(Sigma)
    b = P @ Sigma_12 @ mu

    x = sym.Symbol("x")
    t = sym.Symbol("t")

    # Cumulant function
    K = 0
    for i, l in enumerate(eigenvalues):
        K += (t * b[i] ** 2 * l)/(1 - 2 * t * l) - 1/2 * sym.log(1 - 2 * l * t)

    Kp = sym.diff(K, t).simplify()
    Kpp = sym.diff(K, t, t).simplify()

    roots = sym.lib.symengine_wrapper.solve(sym.Eq(Kp, x), t).args
    if len(roots) > 1:
        for expr in roots:
            trial = Kpp.subs(t, expr).subs(x, np.dot(b,b))
            if trial >= 0.0:
                s_hat = expr
    else:
        s_hat = roots[0]

    f = 1 / sym.sqrt(2 * sym.pi * Kpp.subs(t, s_hat)) * sym.exp(K.subs(t, s_hat) - s_hat * x)
    fp = sym.Lambdify(x, f.simplify())

    c = integrate.quad(fp, 0, np.inf)[0]
    return lambda x: 1/c * fp(x)

def get_hist_distr(mu, Sigma):
    data = get_data(mu, Sigma)
    hist, bin_edges = np.histogram(data, density=True)

    def f(x):
        data = np.zeros(len(x))
        ids = np.digitize(x, bin_edges)
        for ind in ids:
            if ind < (len(hist) - 1):
                data[ind] = hist[ind]
            else:
                data[ind] = 0

        return data

    return (f, bin_edges[0], bin_edges[-1])


def sample_distribution(fn, low, high, sample_num):
    """Perform rejection sampling with a distribution fn. The `mu` is needed to
    determine the interval over which we're sampling.

    Based on pages 83-84 of "Data Reduction and Error Analysis for the Physical
    Sciences".
    """
    distribution = functools.partial(np.random.uniform, low=low,
                                high=high, size=sample_num*2)
    unit = functools.partial(np.random.uniform, low=0.0, high=1.0, size=sample_num*2)

    X = distribution()
    Y = unit()

    current = 0
    xs = []
    while current < sample_num:
        samples = X[fn(X) > Y]
        next_samples = min(len(samples), sample_num - current)
        xs.extend(samples[:next_samples])
        current += next_samples
        X = distribution()
        Y = unit()

    return np.asarray(xs)


# Based on the answers from https://stackoverflow.com/questions/17821458/random-number-from-histogram,
# this offers a more robust way of handling sampling from a histogram
def kde(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = st.gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def generate_rand_from_pdf(pdf, x_grid):
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    values = np.random.rand(1000)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = x_grid[value_bins]
    return random_from_cdf


if __name__ == "__main__":
    plt.style.use("seaborn-ticks")

    fig, axes = plt.subplots(1, 1)
    # fig.tight_layout()
    x_col = np.arange(0.1, 15, 0.1)

    mu2 = np.array([10, 10])
    Sigma2 = np.eye(len(mu2))

    f2 = solve_chi_saddlepoint(mu2, Sigma2)
    data2 = get_data(mu2, Sigma2)
    x_col = np.arange(0.1, np.amax(data2), 0.1, dtype=np.longdouble)

    axes.plot(x_col, f2(x_col), label="saddlepoint approx.")
    axes.hist(data2, bins=50, density=True, label="MC histogram")
    axes.set_title("Error distribution in {}D".format(len(mu2)))
    axes.set_xlabel("Magnitude (m)")
    axes.set_ylabel("Probability")
    axes.legend()


    plt.show()
