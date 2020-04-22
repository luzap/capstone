#! /usr/bin/python3
import numpy as np
from typing import List
from collections import namedtuple

class Actions:
    SPEED = 0
    TURN = 1

action = namedtuple("action", "time action change duration")

class Vehicle2D:
    """An idealized vehicle model, assuming constant acceleration when
    specified, and no error in position measurement. To simulate imperfect
    sensors, we could add an error term from a zero mean Gaussian to the
    position and velocity (we know the inputs precisely).
    """

    def __init__(self, x: float, y: float, bearing: float = 0.0,
                 dt: float = 0.01, vel: float = 1.0, ang_vel: float =0.0):
        self.state = np.array([float(x), float(y), float(bearing)])
        self.vel = vel
        self.ang_vel = ang_vel
        self.dt = dt
        self.current_time = 0
        self.controls = []
        self.change = {index: 0}
        # TODO Do I need to add an error distribution for the measurements?

    def move(self):
        """Iterate the model through a single iteration. Note that we had to
        slightly modify the model that we were initially working with, as it
        interpreted movement along a straight line as degenerate behavior.
        """
        self.state += np.array([self.vel * np.cos(self.state[2]) * self.dt,
                                self.vel * np.sin(self.state[2]) * self.dt,
                                0.0])
        if self.ang_vel > 0.001:
            r = abs(self.vel / self.ang_vel)
            dtheta = self.ang_vel * self.dt
            self.state += np.array([r * np.cos(self.state[2] + dtheta) * self.dt,
                                    r * np.sin(self.state + dtheta) * self.dt,
                                    dtheta
                                   ])
        print(self.state)

    def drive(self, controls: List[str]):
        """Given a list of strings in the form "TIME ACTION CHANGE DURATION",
        drive() outputs a list of actions that the vehicle needs to take at
        specified points in time. The possible names for the actions are given
        by the `Actions` enum. If the action is a TURN, the angle is taken as
        being in degrees, and then for calculation convenience transformed into
        radians.
        """

        for  con in controls:
            parts = con.strip().split(" ")
            current_control = action(
                float(parts[0]),
                getattr(Actions, parts[1]),
                float(parts[2]),
                float(parts[3]))
            # It's a little more convenient to have any turning be expressed
            # in radians than in degrees, but it's much simpler to specify
            # them in degrees
            if current_control.action == Actions.TURN:
                current_control = action(current_control.time,
                                         current_control.action,
                                         np.radians(current_control.change),
                                        current_control.duration)
            self.controls.append(current_control)

        # Making sure that the first action comes first
        self.controls.sort(key=lambda con: con[0])

    def __next__(self):
        """Modelled as an iterator, each vehicle updates its own state based on
        prior instructions. If no instructions are given, the vehicle will
        move from left to right by one velocity unit until the simulation stops.
        """

        if self.controls[0].time == self.current_time:
            self.current_action = self.controls.pop(0)
            data_points = self.current_time - self.current_action.duration


            if self.current_action.action == Actions.STOP:
                # TODO What is the right abstraction here?
                self.change['vel'] = np.linspace(self.vel,
                                                 self.current_action.change,
                                                 data_points)

            if self.current_action.action == Actions.TURN:
                pass
        

        # If there are no events to speak of, continue
        self.move()
        self.current_time += self.dt


class Vehicle3D:

    def __init__(self, x: float, y:float, z:float, args):
        pass
