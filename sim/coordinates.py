#! /usr/bin/python3 
import numpy as np

# Semi-major axis
a: float = 6378137.0 # m
# Semi-minor axis
b: float = 6356752.314140 # m 

# Square of the first eccentricity of the ellipsoid
e_2 = 1 - (b ** 2)/(a ** 2)

def geodetic_to_ECEF(lat, long, h):
    # The conversion assumes all of the angles are in radians
    phi = np.radians(lat)
    lambda_ = np.radians(long)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    N_phi = a / np.sqrt(1 - e_2 * sin_phi * sin_phi)
    x = (N_phi + h) * cos_phi * np.cos(lambda_)
    y = (N_phi + h) * cos_phi * np.sin(lambda_)
    z = ((1 - e_2) * N_phi + h) * sin_phi
    return (x, y, z)

def ECEF_to_geodetic(x: float, y: float, z: float):

    # Transformation of Cartesian to Geodetic Coordinates without Iterations
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    E = np.sqrt(np.square(a) - np.square(b))
    r_E = np.square(r) - np.square(E)
    Q = np.hypot(x, y)

    u = np.sqrt(0.5 * (r_E) + 0.5 * np.hypot(r_E, 2 * E * z))

    # TODO Any possibilities of error here?
    beta = np.arctan((np.hypot(u, E) / u) * (z / Q))
    lat = np.arctan(a/b * np.tan(beta))
    long = np.arctan2(y, x)
    h = np.hypot(z - b * np.sin(beta), Q - a * np.cos(beta))

    return (np.degrees(lat), np.degrees(long), h)


if __name__ == "__main__":
    coords = (24.522673, 54.434157, 0)
    print(coords)
    ecef = geodetic_to_ECEF(*coords)
    print(ecef)
    lla = ECEF_to_geodetic(*ecef)
    print(lla)
