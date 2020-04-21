#! /usr/bin/python3
import numpy as np
from collection import namedtuple

__X = 0
__Y = 1
__BEARING = 2

action = namedtuple("action", "name delta time")

class Vehicle:

    def __init__(self, x: fl, y, dt = 0.01, vel=1, ang_vel=0, bearing=0):
        self.state = np.array([x, y, bearing])
        self.vel = vel
        self.ang_vel = ang_vel
        self.dt = dt
        self.current_time = 0
        self.controls = []

    def move(self):
        # TODO See what the default heading here is 
        self.state += np.array([-self.vel * np.sin(self.state[__BEARING]),
                                self.vel * np.cos(self.state[__BEARING]),
                                0])
        if self.ang_vel:
            r = abs(self.vel / self.ang_vel)
            dtheta = self.ang_vel * self.dt
            self.state += np.array([r * np.sin(self.state[__BEARING] + dtheta),
                                    r * np.cos(self.state + dtheta),
                                    dtheta
                                   ])

    # Control format: ACTION:TIME
    def control(self, controls):
        pass

    def __next__(self):
        pass




