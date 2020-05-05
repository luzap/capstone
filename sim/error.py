#! /usr/bin/python3
import numpy as np
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

if __name__ == "__main__":
    data = get_data(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    bin_heights, bin_borders = np.histogram(data, bins="auto", density=True)
    bin_width = np.diff(bin_borders)
    bin_center = bin_borders[:-1] + bin_width/2
    coeffs = st.exponweib.fit(data)
    mean, var = st.exponweib.stats(coeffs[0], coeffs[1], moments="mv")
    print(mean, var)

    data2 = get_data(np.array([5, 7]), np.array([[7, 2], [2, 18]]))
    coeffs2 = st.exponweib.fit(data2)
    mean1, var1 = st.exponweib.stats(coeffs2[0], coeffs2[1], moments="mv")
    print(mean1, var1)
