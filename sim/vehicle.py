#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from typing import List, Callable
from collections import namedtuple

class Vehicle2D:
    """An idealized vehicle model, assuming the vehicle is a point in 2D space
    and is perfectly aware of its location. To drive this vehicle, you specify
    two functions, ```vel``` and ```ang_vel```, which determine the velocities
    at a given point in time.

    TODO Add an error term after every iteration. The result given by move will
    be the "true" position, with some error
    TODO Add a Kalman filter that's tracking its own path, given the precise
    controls between two points.
    TODO Add Kalman filters that take inputs from one other vehicle
    """

    def __init__(self, x: float, y: float, bearing: float,
                 vel: Callable[[float], float], ang_vel: Callable[[float], float],
                 dt: float = 0.01):
        # TODO Let this be the "GPS" position, where we will add a little
        # error every iterations
        self.state = np.array([float(x), float(y), float(bearing)])
        # TODO This is the Kalman derived position that the vehicle keeps for
        # itself given IMU data, so we need to hook up the Kalman filter
        self.relative_state = self.state.copy()
        self.vel = vel
        self.ang_vel = ang_vel
        self.dt = dt
        self.current_time = 0

    def __move(self):
        """Iterate the model through a single iteration. Note that we had to
        slightly modify the model that we were initially working with, as it
        interpreted movement along a straight line as degenerate behavior.
        """
        t = self.current_time
        dt = self.dt
        translational_velocity = self.vel(t) * np.array([np.cos(self.state[2]),
                                                         np.sin(self.state[2]),
                                                         0])
        angular_velocity = self.ang_vel(t) * np.array([0, 0, 1])

        self.state = self.state +  translational_velocity * dt + angular_velocity * dt

    # TODO We could make this a little bit more transparent
    def __next__(self):
        """For the sake of simplicity, every vehicle is an iterator, and each
        iteration advances the timestamp by a given degree.
        """
        self.__move()
        self.current_time += self.dt

def animate_2d(length: int, vel: Callable[[float], float], ang_vel:
               Callable[[float], float]):
    fig, axes = plt.subplots(2, 1)
    axes[0].grid()

    v = Vehicle2D(0, 0, 0, vel, ang_vel)
    # ax.plot returns a list, which is why the destructuring is necessary
    veh_line = axes[0].plot(v.state[0], v.state[1], 'o-', lw=2)[0]
    ang_line =  axes[1].plot(v.state[2], v.current_time, 'b-', lw=2)[0]

    for axis in axes:
        axis.set_xlim(0, length)
    # TODO Add axes names and plot names
    axes[1].set_ylim(0, 2*np.pi)

    veh_path = np.zeros((length * 100, 2))
    ang_path = np.zeros((length * 100, 2))

    def animate(frame):
        next(v)
        veh_path[frame] = v.state[0:2]
        ang_path[frame] = np.array([v.current_time, v.state[2]])

        veh_line.set_data(veh_path[:, 0], veh_path[:, 1])
        ang_line.set_data(ang_path[:, 0], ang_path[:, 1])
        return veh_line, ang_line

    # TODO Add optional delay parameter
    _ = matplotlib.animation.FuncAnimation(fig, animate, frames=length*100,
                                           interval=0.1, repeat=False, blit=True)
    plt.show()



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

    def __move(self):
        t = self.current_time
        dt = self.dt
        st = self.state

        dx = self.vel(t) * np.array([np.cos(st[3]) * np.cos(st[4]),
                                     np.sin(st[3]) * np.cos(st[4]),
                                     np.sin(st[3]), 0, 0, 0])
        dtheta = self.yaw(t) * np.array([0, 0, 0, 1, 0, 0])
        dbeta = self.pitch(t) * np.array([0, 0, 0, 0, 1, 0])
        dalpha = self.roll(t) * np.array([0, 0, 0, 0, 0, 1])

        self.state = (dx + dtheta + dbeta + dalpha) * dt

    def __next__(self):
        self.__move()
        self.current_time += self.dt
