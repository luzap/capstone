#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

import kalman
import error
import vehicle

if __name__ == "__main__":
    print("2D vehicle test")
    # NB: here we are making the assumption that our model is perfect
    vehicle_pos = np.array([0, 0])
    vehicle_error = np.array([[0, 0], [0, 0]])
    vehicle_vel = lambda _: 1
    vehicle_ang_vel = lambda _: 0
    kalman_filter = kalman.UKFilter(vehicle_pos, vehicle_error,
                                    # Oh God, what is Q here?
                                    # What do we measure?
                                    vehicle.


    )
    time = 50
    dt = 0.01

    print("Linear case")
    ticks = np.arange(0, time, dt)
    true_state = np.zeros((2, len(ticks)))
    est_state = np.zeros((2, len(ticks)))

    for i in ticks:


    

    # TODO Iterate regularly
    # TODO Add some noise to the velocity (definitely not how I'm doing it now)



    print("3D vehicle test")
