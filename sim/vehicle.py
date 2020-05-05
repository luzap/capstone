#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from typing import List, Callable
from collections import namedtuple

# TODO At least test this today with the Kalman filter
def vehicle_2d_process_model(x0, dt, vel: float, ang_vel: float, use_noise: bool
                             = False):
        # TODO Figure out sensible error values based on the velocity
        e_v = np.random.normal(0.0, scale=0.05) if not use_noise else 1.0
        e_a = np.random.normal(0.0, scale=0.01) if not use_noise else 1.0
        translational_velocity = (vel * e_v) * np.array([np.cos(x0[2]),
                                                         np.sin(x0[2]),
                                                         0
                                                        ])
        angular_velocity = (ang_vel * e_a)  * np.array([0, 0, 1])

        return x0 + translational_velocity * dt + angular_velocity * dt

def vehicle_3d_process_model(x, dt, vel, yaw, pitch, roll):
    dx = vel * np.array([np.cos(x[3]) * np.cos(x[4]),
                         np.sin(x[3]) * np.cos(x[4]),
                         np.sin(x[3]), 0, 0, 0])
    dtheta = yaw * np.array([0, 0, 0, 1, 0, 0])
    dbeta = pitch * np.array([0, 0, 0, 0, 1, 0])
    dalpha = roll * np.array([0, 0, 0, 0, 0, 1])

    return (dx + dtheta + dbeta + dalpha) * dt
