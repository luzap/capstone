#!/usr/bin/python3
import numpy as np

def builder(timings: str):
    """Given a long string, this is the easiest way of generating control
    functions for testing out the models."""

    f = "def f(t: float):\n"
    h = "def h(t: float):\n"
    fmt = "\tif t < {time}:\n\t\treturn {vel}\n"

    d = {}
    for line in timings.split("\n"):
        time, vel, ang = [float(part) for part in line.split(" ")]
        f = "{header}{new}".format(header=f, new=fmt.format(time=time, vel=vel))
        h = "{header}{new}".format(header=h, new=fmt.format(time=time, vel=ang))

    exec(f, d)
    eval(h, d)

    return d['f'], d['h']

def unit_vel(_):
    return 1

def null_vel(_):
    return 0
