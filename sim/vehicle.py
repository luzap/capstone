#! /usr/bin/python3
import numpy as np

def vehicle_2d_process_model(x0, dt, vel: float, ang_vel: float, use_noise: bool
                             = False):
        e_v = np.random.normal(1.0, scale=vel*0.1) if not use_noise else 1.0
        e_a = np.random.normal(1.0, scale=vel*0.01) if not use_noise else 1.0
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
