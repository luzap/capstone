#! /usr/bin/python3
import numpy as np
from typing import List
from collections import namedtuple

class Actions:
    STOP = 0
    SLOW = 1
    SPEED = 2
    TURN = 3

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

    # Control format: ACTION:AMOUNT:TIME
    def control(self, controls: List[str]):
        for con in controls:
            parts = con.strip().split(":")
            self.controls.append(action(
                getattr(Actions, parts[0]),
                float(parts[1]), float(parts[2])))
        # Making sure that the first action comes first
        self.controls.sort(key=lambda con: con[2])

    def __next__(self):
        # TODO Check for current command and time
        # TODO Figure out how much turning needs to be done
        #

        pass




