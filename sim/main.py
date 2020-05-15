#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import kalman
import error
import vehicle

# [x y v \theta \dot{\theta}]
def state_mean(states, Wm):
    state_mean = np.zeros(len(state))
    state_mean[0] = np.sum(np.dot(states[:, 0], Wm))
    state_mean[1] = np.sum(np.dot(states[:, 1], Wm))
    state_mean[2] = np.sum(np.dot(states[:, 2], Wm))
    state_mean[3] = kalman.normalize_angle()
    state_mean[4] = None # TODO Do we normalize angular velocity?
    return state_mean

def state_residual(s1, s2):
    residual = np.zeros(5)
    residual[:2] = s1[:2] - s2[:2]
    residual[3] = kalman.normalize_angle(s1[3] - s3[3])
    residual[4] = s1[4] - s2[4]
    return residual

def measurement_function(state):
    return np.array([state[2], state[4]])

def measurement_mean(z, Wm):
    measurement_mean = np.zeros(len(z))
    measurement_mean[0] = np.sum(np.dot(z[:, 0], Wm))
    measurement_mean[1] = kalman.normalize_angle()
    return measurement_mean

def measurement_residual(z1, z2):
    z = np.zeros(2)
    z[0] = z1[0] - z[2]
    z[1] = kalman.normalize_angle(z1[1] - z[2])
    return z

if __name__ == "__main__":
    # Wanted a simple container to hold data in a predictable manner
    class Vehicle:
        pass

    time = 10.0
    dt = 0.01

    vecs = [None, None]
    vecs[0] = Vehicle()
    vecs[1] = Vehicle()

    # TODO Randomly assign the first two values
    vecs[0].state = np.array([0, 0, 0, 0, 0])
    vecs[1].state = np.array([0, 0, 0, 0, 0])
    # TODO Is this correct?
    vecs[0].error = np.eye(5)
    vecs[1].error = np.eye(5)


    Gamma = np.array([0.5*dt, 0.5*dt, dt, 0.5*dt ** 2, dt])
    Q = np.outer(Gamma, Gamma.T)

    R = np.eye(2)

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

    ticks = np.arange(0, time, dt)
    for i, _ in enumerate(vecs):
        vecs[i].true_state = np.zeros((len(ticks), 3))
        vecs[i].est_state = np.zeros((len(ticks), 3))
        vecs[i].gps_state = np.zeros((len(ticks), 2))


    # True velocity is one-to-one with the inputs, however gets distorted by sensor
    # noise, which is why we will be multiplying both velocities by a normally
    # distributed random variable, with mean 1 and variance 1 for the translational
    # case and mean 1, variance 0.1 in the rotational case, since the rotational velocity
    # is bound to have less error (at least on cars)
    p_vals = np.zeros(int(time/0.5))

    trans_vel_func = None
    rot_vel_func = None

    current = 0
    for i, t in enumerate(ticks):

        # TODO How are we generating velocities?
        trans_vel = 0
        trans_e = np.random.normal(1, 0.5)
        # TODO Replace this with a function
        rot_vel = 0
        rot_e = np.random.normal(1, 0.1)

        # TODO Here, estimate
        for veh in vecs:
            veh.self_estimator.predict(trans_vel, rot_vel, trans_e, rot_e)

        self_state_estimator.predict(trans_vel, rot_vel, trans_e, rot_e)
        x, P = self_state_estimator.update(np.array([trans_vel, rot_vel]))

        est_state[i+1] = np.array([x[0], x[1], x[3]])
        current_pos = vehicle.vehicle_2d_process_model(true_state[i], dt, trans_vel, rot_vel)
        true_state[i+1] = current_pos
        gps_error_x = np.random.normal(1, 0.01)
        gps_error_y = np.random.normal(1, 0.01)
        gps_state[i+1] = np.array([current_pos[0] * gps_error_x,
                                   current_pos[1] * gps_error_y])

    if i % 0.5 == 0:
        est_state_distribution = error.solve_chi_saddlepoint(mu[:2], np.eye(2))
        gps_state_distribution = error.solve_chi_saddlepoint(gps_state[i+1], 0.1*np.eye(2))
        est_state_sample = error.sample_distribution(est_state_distribution, mu[:2], 1, 30)
        gps_state_sample = error.sample_distribution(gps_state_distribution, gps_state[i+1], 1, 30)
        p_vals[current] = st.mannwhitneyu(est_state_sample, gps_state_sample)[0]

    if i % (3 * time / 4) == 0:
        # Change the functions so that the spooing starts
        vecs[0].trans_vel_func = spoofed_vel_func
        vecs[0].rot_vel_func = spoofed_rot_func


    plt.plot(ticks, p_vals)
    plt.show()
