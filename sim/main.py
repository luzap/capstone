#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats import st

import kalman
import error
import vehicle

# [x y v \theta \dot{\theta}]
def state_mean(states, Wm):
    # TODO How do the means work exactly
    pass

def state_residual(s1, s2):
    residual = np.zeros(5)
    residual[:2] = s1[:2] - s2[:2]
    residual[3] = kalman.normalize_angle(s1[3] - s3[3])
    residual[4] = s1[4] - s2[4]
    return residual

def measurement_function(state):
    return np.array([state[2], state[4]])

def measurement_mean(z, Wm):
    # TODO The first is not angular, the second is
    pass

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

    vecs = []
    vec[0] = Vehicle()
    vec[1] = Vehicle()

    # TODO Randomly assign the first two values
    vec[0].state = np.array([0, 0, 0, 0, 0])
    vec[1].state = np.array([0, 0, 0, 0, 0])
    # TODO Is this correct?
    vec[0].error = np.eyes(5)
    vec[1].error = np.eyes(5)


    Gamma = np.array([0.5*dt, 0.5*dt, dt, 0.5*dt ** 2, dt])
    Q = np.outer(Gamma, Gamma.T)

    R = np.eye(2)

    vec[0].self_estimator = kalman.UKFilter(vec[0].state, vec[0].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )
    vec[0].other_estimator = kalman.UKFilter(vec[1].state, vec[1].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )
    vec[1].self_estimator = kalman.UKFilter(vec[1].state, vec[1].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )
    vec[1].other_estimator = kalman.UKFilter(vec[0].state, vec[0].error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )

    for i, _ in enumerate(veh):
        veh[i].true_state = np.zeros((len(ticks), 3))
        veh[i].est_state = np.zeros((len(ticks), 3))
        veh[i].gps_state = np.zeros((len(ticks), 2))

    ticks = np.arange(0, time, dt)

    # True velocity is one-to-one with the inputs, however gets distorted by sensor
    # noise, which is why we will be multiplying both velocities by a normally
    # distributed random variable, with mean 1 and variance 1 for the translational
    # case and mean 1, variance 0.1 in the rotational case, since the rotational velocity
    # is bound to have less error (at least on cars)
    p_vals = np.zeros(time/0.5)

    current = 0
    for i, t in enumerate(ticks):

        # TODO How are we generating velocities?
        trans_vel = 0
        trans_e = np.random.normal(1, 0.5)
        # TODO Replace this with a function
        rot_vel = 0
        rot_e = np.random.normal(1, 0.1)

        # TODO Here, estimate
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
        p_vals[current] = st.mannwhitneyu(est_state_sample, gps_state_sample)


    plt.plot(ticks, p_vals)
    plt.show()
