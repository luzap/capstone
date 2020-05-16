#! /usr/bin/python3

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import vehicle as vh
import error as err
import statsmodels.stats.weightstats


if __name__ == "__main__":
    class Vehicle:
        pass

    vecs = [Vehicle() for i in range(2)]

    time = 2
    dt = 0.01
    ticks = np.arange(0, time, dt)
    p_values = []

    # TODO Set initial point
    for i, vec in enumerate(vecs):
        vecs[i].est_state = np.zeros((len(ticks)+1, 3))
        vecs[i].est_state[0] = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0, 2*np.pi)
        ])

        vecs[i].gps_state = np.zeros((len(ticks)+1, 3))
        vecs[i].gps_state[0] = np.zeros(3)

    t1 = lambda t: 0.1 if t < 1 else 1 if t < 2 else 5
    a1 = lambda t: np.pi if t < 2 else 2*np.pi if t < 1.5 else 0
    t2 = lambda t: 2 if t < 1 else 0 if t < 2 else 1
    a2 = lambda t: 0

    vecs[0].est_trans_vel = vecs[0].gps_trans_vel = t1
    vecs[0].est_rot_vel = vecs[0].gps_rot_vel = a1

    vecs[1].est_trans_vel = vecs[1].gps_trans_vel = t2
    vecs[1].est_rot_vel = vecs[1].gps_rot_vel = a2

    t_values = []
    spoofed = False
    for i, t in enumerate(ticks):
        for vec in vecs:
            vec.gps_state[i+1] = vh.vehicle_2d_process_model(vec.gps_state[i], dt,
                                                             vec.gps_trans_vel(t),
                                                             vec.gps_rot_vel(t))

            vec.est_state[i+1] = vh.vehicle_2d_process_model(vec.est_state[i], dt,
                                                             vec.est_trans_vel(t),
                                                             vec.est_rot_vel(t))
        if i % 10 == 0:
            gps_diff = vecs[0].gps_state[i+1][:2] - vecs[1].gps_state[i+1][:2]
            est_diff = vecs[0].est_state[i+1][:2] - vecs[1].est_state[i+1][:2]

            # Instead of histograms, we're using kernel density estimation, which
            # is better than
            gps_data = err.get_data(gps_diff, np.array([[0.02, 0], [0, 0.02]]))
            gps_min = np.amin(gps_data)
            gps_max = np.amax(gps_data)
            gps_grid = np.linspace(gps_min, gps_max, 100)
            gps_kde = err.kde(gps_data, gps_grid)
            gps_samples = err.generate_rand_from_pdf(gps_kde, gps_grid)

            # We always assume the sensors are independently distributed
            est_data = err.get_data(est_diff, np.array([[0.1, 0], [0, 0.02]]))
            est_min = np.amin(est_data)
            est_max = np.amax(est_data)
            est_grid = np.linspace(est_min, est_max, 100)
            est_kde = err.kde(est_data, est_grid)
            est_samples = err.generate_rand_from_pdf(est_kde, est_grid)

            p = statsmodels.stats.weightstats.ttost_ind(est_samples, gps_samples,
                                                    low=0,
                                                    upp=2,
                                                    usevar='unequal')
            p_values.append(p[0])
            t_values.append(i)

        if i!= 0 and i % 50 == 0 and not spoofed:
            vecs[0].gps_trans_vel = lambda _: -2
            vecs[0].gps_rot_vel = lambda _: np.pi / 2
            spoofed = True
            pass

    plt.title("Probability of the null hypothesis being rejected")
    plt.xlabel("Iteration")
    plt.ylabel(r"$p$-value")
    plt.plot(t_values, p_values)
    plt.axvline(100, linestyle="--", alpha=0.9)
    print(p_values)

    plt.show()
