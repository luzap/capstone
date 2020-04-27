#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from typing import List, Callable
from collections import namedtuple

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


def animate_2d(length: int,
               vel: Callable[[float], float], ang_vel:
               Callable[[float], float]):
    # TODO Need plot for position and angle, then for residuals of angle
    # and residuals of position
    fig, axes = plt.subplots(2, 2)
    axes[0].grid()

    # TODO Initialize random start positions w/


    # # ax.plot returns a list, which is why the destructuring is necessary
    # veh_line = axes[0].plot(v.state[0], v.state[1], 'o-', lw=2)[0]
    # ang_line =  axes[1].plot(v.state[2], v.current_time, 'b-', lw=2)[0]

    # for axis in axes:
    #     axis.set_xlim(0, length)
    # # TODO Add axes names and plot names
    # axes[1].set_ylim(0, 2*np.pi)

    # veh_path = np.zeros((length * 100, 2))
    # ang_path = np.zeros((length * 100, 2))

    # def animate(frame):
    #     next(v)
    #     veh_path[frame] = v.state[0:2]
    #     ang_path[frame] = np.array([v.current_time, v.state[2]])

    #     veh_line.set_data(veh_path[:, 0], veh_path[:, 1])
    #     ang_line.set_data(ang_path[:, 0], ang_path[:, 1])
    #     return veh_line, ang_line

    # # TODO Add optional delay parameter
    # _ = matplotlib.animation.FuncAnimation(fig, animate, frames=length*100,
    #                                        interval=0.1, repeat=False, blit=True)
    # plt.show()



def vehicle_3d_process_model(x, dt, vel, yaw, pitch, roll):
    dx = vel * np.array([np.cos(x[3]) * np.cos(x[4]),
                         np.sin(x[3]) * np.cos(x[4]),
                         np.sin(x[3]), 0, 0, 0])
    dtheta = yaw * np.array([0, 0, 0, 1, 0, 0])
    dbeta = pitch * np.array([0, 0, 0, 0, 1, 0])
    dalpha = roll * np.array([0, 0, 0, 0, 0, 1])

    return (dx + dtheta + dbeta + dalpha) * dt




class Vehicle3D:

    def __init__(self, x: float, y: float, z: float, theta: float, beta: float,
                 alpha: float, vel: Callable[[float], float],
                 yaw: Callable[[float], float], pitch: Callable[[float], float],
                 roll: Callable[[float], float], dt: float = 0.01):
        self.state = np.array([x, y, z, theta, beta, alpha])
        self.current_time = 0
        self.dt = dt
        self.vel = vel
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


    def __next__(self):
        self.__move()
        self.current_time += self.dt
