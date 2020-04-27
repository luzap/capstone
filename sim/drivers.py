#! /usr/bin/python3
import pandas as pd
import numpy as np

def read_data():
    gps = pd.read_csv("../data/AGZ_subset/Log Files/OnboardGPS.csv",
                      skipinitialspace=True)
    pose = pd.read_csv("../data/AGZ_subset/Log Files/OnboardPose.csv",
                       skipinitialspace=True)

    gps.drop([], axis=1, inplace=True)
