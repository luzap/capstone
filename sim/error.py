#! /usr/bin/python3
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exponential(x, mu, sigma, height):
    return height * np.exp(-(x - mu) ** 2/(sigma ** 2))

def fit_distribution(mu, Sigma, best_fit=exponential, sample_num=10000):
    sums = np.zeros(sample_num)
    for i, _ in enumerate(sums):
        sample = np.random.multivariate_normal(mu, Sigma)
        sums[i] = np.dot(sample, sample.T)

    # We classify the entire thing via histograms, take their centerpoints
    # and use those centerpoints to fit a curve
    bin_heights, bin_borders = np.histogram(sums, bins="auto")
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths/2

    best_fit_coeffs, _ = curve_fit(best_fit, bin_centers, bin_heights, maxvfev=1000)
    x_width = np.linspace(bin_borders[0], bin_borders[-1], sample_num)
    plt.bar(bin_centers, bin_heights, width=bin_widths, label="histogram")
    plt.plot(x_width, exponential(x_width, *best_fit_coeffs), label="fit", c="red")
    plt.legend()
    plt.show()
    print(best_fit_coeffs)

if __name__ == "__main__":
    fit_distribution(np.array([0, 0]), np.array([[1, 0],[0, 1]]), sample_num=1000)
