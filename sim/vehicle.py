#! /usr/bin/python3
import numpy as np
from collections import namedtuple

action = namedtuple("action", "name delta time")

class Vehicle:

    __X = 0
    __Y = 1
    __BEARING = 2

    def __init__(self, x: float, y: float, dt = 0.01, vel=1.0, ang_vel=0,
                 bearing=0.0):
        self.state = np.array([float(x), float(y), float(bearing)])
        self.vel = vel
        self.ang_vel = ang_vel
        self.dt = dt
        self.current_time = 0
        self.controls = []

    def move(self):
        # TODO See what the default heading here is 
        self.state += np.array([self.vel * np.cos(self.state[2]) * self.dt,
                                self.vel * np.sin(self.state[2]) * self.dt,
                                0.0])
        if self.ang_vel > 0.001:
            print('this procs')
            r = abs(self.vel / self.ang_vel)
            dtheta = self.ang_vel * self.dt
            self.state += np.array([r * np.cos(self.state[2] + dtheta) * self.dt,
                                    r * np.sin(self.state + dtheta) * self.dt,
                                    dtheta
                                   ])
        print(self.state)

    # Control format: ACTION:TIME
    def control(self, controls):
        pass

    def __next__(self):
        pass




