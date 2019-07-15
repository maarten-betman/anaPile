import numpy as np


def grain_pressure(depth, gamma_sat, u2=None):
    """
    Determine the grain pressure over the depth.

    Parameters
    ----------
    depth : array
        Depth values in [m], returned by Pygef.
    gamma_sat : array
        Saturated gamma values in [MPa]. Determined by classification.
    u2 : array
        Water pressure in [MPa]
    Returns
    -------
    grain_pressure : array
        Grain pressure in [MPa]
    """
    h = np.diff(depth)
    h = np.append(h, h[-1])
    if u2 is None:
        u2 = depth * 0.01

    return np.cumsum(h * gamma_sat) - u2


