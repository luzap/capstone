#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

import kalman
import error
import vehicle

# [x y v \theta \dot{\theta}]
def state_mean(states, Wm):
    pass

def state_residual(state):
    pass

def measurement_function(z):
    pass

def measurement_mean(z, Wm):
    pass

def measurement_residual(z):
    pass


if __name__ == "__main__":
    vehicle_state = np.array([0, 0, 0, 0, 0]) # \mu
    vehicle_error = np.zeros((5, 5)) # P
    time = 10.0
    dt = 0.01

    Gamma = np.array([0.5*dt, 0.5*dt, dt, 0.5*dt ** 2, dt])
    Q = np.outer(Gamma, Gamma.T)

    R = np.eye(2)

    self_state_estimator = kalman.UKFilter(vehicle_pos, vehicle_error, Q, R, dt,
                                           vehicle.vehicle_2d_process_model,
                                           measurement_function,
                                           state_mean,
                                           state_residual,
                                           measurement_mean,
                                           measurement_residual
                                          )

    ticks = np.arange(0, time, dt)
    true_state = np.zeros((len(ticks), 3))
    est_state = np.zeros((len(ticks), 3))

    # True velocity is one-to-one with the inputs, however gets distorted by sensor
    # noise, which is why we will be multiplying both velocities by a normally
    # distributed random variable, with mean 1 and variance 1 for the translational
    # case and mean 1, variance 0.1 in the rotational case, since the rotational velocity
    # is bound to have less error (at least on cars)

    for dt in ticks:
        self_state_estimator.predict()
        x, P = self_state_estimator.update()
