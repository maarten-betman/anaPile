import numpy as np


def grain_pressure(depth, gamma_sat, u2=None):
    h = np.diff(depth)
    h = np.append(h, h[-1])
    if u2 is None:
        u2 = depth * 0.01

    return np.cumsum(h * gamma_sat) - u2
