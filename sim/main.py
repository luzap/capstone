#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

import kalman
import error
import vehicle

# [x y v \theta \dot{\theta}]
def mean_function(state_vector):
    pass

def residual_function(state):
    pass

# TODO How do we convert vehicle measurement to state?
def measurement_function(z):
    pass


# Mean is given by x, y, \theta
# Measurement is given by v and \dot{\theta}

if __name__ == "__main__":
    print("2D vehicle test")
    # NB: here we are making the assumption that our model is perfect
    vehicle_pos = np.array([0, 0, 0])
    vehicle_error = np.array([[0, 0, 0], [0, 0, 0]])
    # TODO Factor this out
    vehicle_vel = lambda _: 1
    vehicle_ang_vel = lambda _: 0
    kalman_filter = kalman.UKFilter(vehicle_pos, vehicle_error,
                                    np.eye(len(vehicle_pos)),
                                    # What do we measure?
                                    vehicle.vehicle_2d_process_model,
                                    0
                                   )
    time = 50
    dt = 0.01

    print("Linear case")
    ticks = np.arange(0, time, dt)
    true_state = np.zeros((2, len(ticks)))
    est_state = np.zeros((2, len(ticks)))

    # TODO Iterate regularly
    # TODO Add some noise to the velocity (definitely not how I'm doing it now)
