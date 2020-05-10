# Detecting and Preventing GPS spoofing against vehicular networks

The simulation code for my capstone project, structured as follows:

- `coordinates.py`: ECEF to geodesic coordinate conversions and vice versa. Used to interpret data in the ECEF coordinate frame
- `drivers.py`: data ingestion and processing from the Zurich Micro Aerial Vechile dataset
- `error.py`: modelling the distribution of squared distances from the origin to some point whose location is determined by a given Gaussian distribution.
- `helpers.py`: helper functions.
- `kalman.py`: an implementation of a working Unscented Kalman filter (UKF), compared against other implementations.
- `main.py`: main driver file for the simulation.
- `vehicle.py`: vehicular driving models in both 2D and 3D.
