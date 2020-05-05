#! /usr/bin/python3
import pandas as pd
import numpy as np

# TODO Normalize the column data to something more appropriate for lat, long, alt\n,
# TODO Merge the GPS and pose data on the timestep, should that be the case\n,
# TODO Start pushing this through the Kalman filter and see the residuals\n,
# TODO If the residuals are fine, then we move on

def read_data():
    gps = pd.read_csv("../data/AGZ_subset/Log Files/OnboardGPS.csv",
                      skipinitialspace=True)
    pose = pd.read_csv("../data/AGZ_subset/Log Files/OnboardPose.csv",
                       skipinitialspace=True)

    gps.drop(["imgid", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16",
              "Unnamed: 17", "num_sat", "fix_type", "eph_m", "epv_m"],
             axis=1, inplace=True)

