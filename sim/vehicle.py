#! /usr/bin/python3
import numpy as np

def vehicle_2d_process_model(x0, dt, vel: float, ang_vel: float, trans_e: float = 1,
                            rot_e: float = 0):
        translational_velocity = (vel * trans_e) * np.array([np.cos(x0[2]),
                                                         np.sin(x0[2]),
                                                         0,0, 0])

        angular_velocity = (ang_vel * rot_e)  * np.array([0, 0, 1, 0, 0])

        return x0 + translational_velocity * dt + angular_velocity * dt + \
                np.array([0, 0, vel, 0, ang_vel])

def vehicle_3d_process_model(x, dt, vel, yaw, pitch, roll):
    dx = vel * np.array([np.cos(x[3]) * np.cos(x[4]),
                         np.sin(x[3]) * np.cos(x[4]),
                         np.sin(x[3]), 0, 0, 0])
    dtheta = yaw * np.array([0, 0, 0, 1, 0, 0])
    dbeta = pitch * np.array([0, 0, 0, 0, 1, 0])
    dalpha = roll * np.array([0, 0, 0, 0, 0, 1])

    return (dx + dtheta + dbeta + dalpha) * dt
