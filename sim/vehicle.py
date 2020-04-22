#! /usr/bin/python3
import numpy as np
from typing import List, Callable
from collections import namedtuple

class Vehicle2D:
    """An idealized vehicle model, assuming the vehicle is a point in 2D space
    and is perfectly aware of its location. To drive this vehicle, you specify
    two functions, ```vel``` and ```ang_vel```, which determine the velocities
    at a given point in time.

    TODO Consider adding an error term to the position upon every iteration
    """

    def __init__(self, x: float, y: float, bearing: float,
                 vel: Callable[float], ang_vel: Callable[float], dt: float = 0.01):
        self.state = np.array([float(x), float(y), float(bearing)])
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

    def __next__(self):
        """For the sake of simplicity, every vehicle is an iterator, and each
        iteration advances the timestamp by a given degree.
        """
        self.__move()
        self.current_time += self.dt


class Vehicle3D:

    def __init__(self, x: float, y: float, z: float, theta: float, beta: float,
                 alpha: float, vel: Callable[float],
                 yaw: Callable[float], pitch: Callable[float],
                 roll: Callable[float], dt: float = 0.01):
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
