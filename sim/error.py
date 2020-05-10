#!/usr/bin/python3
import numpy as np
import sympy
import scipy.optimize as scio
import matplotlib.pyplot as plt
import scipy.stats as st

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
    # TODO Compute numerically as well
    K = 0
    for i, l in enumerate(eigenvalues):
        K += (b[i] * l)/(1 - 2 * t * l) - 1/2 * sympy.ln(1 - 2 * l * t)
    Kp = sympy.diff(K, t)
    Kpp = sympy.diff(K, t, t)
    # TODO Handle the case with multiple \hat{s}
    s_hat = sympy.solve(Kp - x, t)[0]

    f = 1 / sympy.sqrt(2 * sympy.pi * Kpp.subs(t, s_hat)) * sympy.exp(K.subs(t, s_hat) - s_hat * x)
    print(sympy.latex(f))
    fm = sympy.utilities.lambdify(x, f)
    return fm


if __name__ == "__main__":
    # TODO Add subplots and theming
    plt.style.use("seaborn-ticks")

    fig, axes = plt.subplots(2, 1)
    fig.tight_layout()
    x_col = np.arange(0.1, 15, 0.1)

    mu2 = np.array([0, 0])
    Sigma2 = np.eye(len(mu2))
    mu3 = np.array([0, 0, 0])
    Sigma3 = np.eye(len(mu3))
    x_col = np.arange(0.1, 15, 0.1)

    f2 = solve_chi_saddlepoint(mu2, Sigma2)
    data2 = get_data(mu2, Sigma2)

    axes[0].plot(x_col, f2(x_col), label="saddlepoint approx.")
    axes[0].hist(data2, bins=50, density=True, label="MC histogram")
    axes[0].set_title("Error distribution in {}D".format(len(mu2)))
    axes[0].set_xlabel("Magnitude (m)")
    axes[0].set_ylabel("Probability")
    axes[0].legend()

    f3 = solve_chi_saddlepoint(mu3, Sigma3)
    data3 = get_data(mu3, Sigma3)

    axes[1].plot(x_col, f3(x_col), label="saddlepoint approx.")
    axes[1].hist(data3, bins=50, density=True, label="MC histogram")
    axes[1].set_title("Error distribution in {}D".format(len(mu3)))
    axes[1].set_xlabel("Magnitude (m)")
    axes[1].set_ylabel("Probability")
    axes[1].legend()

    plt.show()
