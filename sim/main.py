#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import kalman
import error
import vehicle

# [x y v \theta \dot{\theta}]
def state_mean(Wm, states):
    state_mean = np.zeros(5)
    sum_sin = np.sum(np.dot(np.sin(states[:, 3]), Wm))
    sum_cos = np.sum(np.dot(np.cos(states[:, 3]), Wm))

    state_mean[0] = np.sum(np.dot(states[:, 0], Wm))
    state_mean[1] = np.sum(np.dot(states[:, 1], Wm))
    state_mean[2] = np.sum(np.dot(states[:, 2], Wm))
    state_mean[3] = kalman.normalize_angle(np.arctan2(sum_sin, sum_cos))
    state_mean[4] = np.sum(np.dot(states[:, 4], Wm))
    return state_mean

def state_residual(s1, s2):
    y = s1 - s2
    y[3] = kalman.normalize_angle(y[3])
    return y

def measurement_function(state):
    return np.array([state[2], state[4]])

# TODO This is wrong
def measurement_mean(Wm, z):
    measurement_mean = np.zeros(2)
    sum_sin = np.sum(np.dot(np.sin(z[:, 1]), Wm))
    sum_cos = np.sum(np.dot(np.cos(z[:, 1]), Wm))

    measurement_mean[0] = np.sum(np.dot(z[:, 0], Wm))
    measurement_mean[1] = kalman.normalize_angle(np.arctan2(sum_sin, sum_cos))
    return measurement_mean

def measurement_residual(z1, z2):
    z = z1 - z2
    z[1] = kalman.normalize_angle(z[1])
    return z

if __name__ == "__main__":
    # Wanted a simple container to hold data in a predictable manner
    class Vehicle:
        pass

    time = 10.0
    dt = 0.1

    vecs = [None, None]
    vecs[0] = Vehicle()
    vecs[1] = Vehicle()

    # TODO Randomly assign the first two values
    vecs[0].state = np.array([0, 0, 0, 0, 0])
    vecs[1].state = np.array([0, 0, 0, 0, 0])
    # TODO Is this correct?
    vecs[0].error = np.eye(5)
    vecs[1].error = np.eye(5)

    Gamma = np.array([0, 0, 0, 0, np.sqrt(dt)], dtype=np.longdouble) * 1
    Q = np.outer(Gamma, Gamma.T)
    R = np.zeros((2, 2))

    vecs[0].self_estimator = kalman.UKFilter(vecs[0].state, vecs[0].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )
    vecs[0].other_estimator = kalman.UKFilter(vecs[1].state, vecs[1].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )
    vecs[1].self_estimator = kalman.UKFilter(vecs[1].state, vecs[1].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )
    vecs[1].other_estimator = kalman.UKFilter(vecs[0].state, vecs[0].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )

    ticks = np.arange(dt, time, dt)
    for i, vec in enumerate(vecs):
        vecs[i].true_state = np.zeros((len(ticks)+1, 3))
        vecs[i].true_state[0] = vecs[i].state[:3]

        vecs[i].est_state = np.zeros((len(ticks)+1, 3))
        vecs[i].est_state[0] = vecs[i].state[:3]

        vecs[i].gps_state = np.zeros((len(ticks)+1, 2))
        vecs[i].gps_state[0] = vecs[i].state[:2]


    # Values over time
    p_vals = np.zeros(int(time/0.5))

    # Velocity functions
    vecs[0].trans_vel_func = lambda _: 5
    vecs[0].rot_vel_func = lambda _: 0
    vecs[1].trans_vel_func = lambda _: 5
    vecs[1].rot_vel_func = lambda t: 5 if t < 1 else 0

    # SETUP: making the first prediction, given some control input from both
    trans_e = np.random.normal(1, 0.5)
    rot_e = np.random.normal(1, 0.1)
    for j, veh in enumerate(vecs):
            veh.self_estimator.predict(veh.trans_vel_func(0), veh.rot_vel_func(0),
                                       trans_e, rot_e)

            other_vec = vecs[(j+1) % len(vecs)]
            veh.other_estimator.predict(other_vec.trans_vel_func(0),
                                        other_vec.rot_vel_func(0),
                                        trans_e,
                                        rot_e)
    # FIrst loop
    spoofing = True
    for i, t in enumerate(ticks):
        trans_e = np.random.normal(1, 0.5)
        rot_e = np.random.normal(1, 0.1)

        for j, veh in enumerate(vecs):
            other_vec = vecs[(j+1) % len(vecs)]

            # This updates the predictions at the current time
            x, P = veh.self_estimator.update(np.array([
                veh.trans_vel_func(t-dt),
                veh.rot_vel_func(t-dt)
            ]))
            # TODO How are we using this info?
            veh.other_estimator.update(np.array([
                other_vec.trans_vel_func(t-dt),
                other_vec.rot_vel_func(t-dt)
            ]))

            veh.self_estimator.predict(veh.trans_vel_func(t),
                                       veh.rot_vel_func(t),
                                       trans_e, rot_e)

            veh.other_estimator.predict(other_vec.trans_vel_func(t),
                                        other_vec.rot_vel_func(t),
                                        trans_e,
                                        rot_e)
            # TODO Update all of the internal arrays

            # est_state[i+1] = np.array([x[0], x[1], x[3]])
            # current_pos = vehicle.vehicle_2d_process_model(true_state[i], dt, trans_vel, rot_vel)
            # true_state[i+1] = current_pos
            # gps_error_x = np.random.normal(1, 0.01)
            # gps_error_y = np.random.normal(1, 0.01)
            # gps_state[i+1] = np.array([current_pos[0] * gps_error_x,
            #                         current_pos[1] * gps_error_y])

        # # Check for spoofing every 5 ticks
        # if i % 0.5 == 0:
        #     est_dist_distr = error.solve_chi_saddlepoint(, np.eye(2))
        #     gps_state_distribution = error.solve_chi_saddlepoint(gps_state[i+1], 0.1*np.eye(2))
        #     est_state_sample = error.sample_distribution(est_state_distribution, mu[:2], 1, 30)
        #     gps_state_sample = error.sample_distribution(gps_state_distribution, gps_state[i+1], 1, 30)
        #     p_vals[current] = st.mannwhitneyu(est_state_sample, gps_state_sample)[0]

        # # After some normal operation, start the spoofing attack
        # if i % (time / 4) == 0:
        #     spoofing = True

    plt.plot(ticks, p_vals)
    plt.show()
