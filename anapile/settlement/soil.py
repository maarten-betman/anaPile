import numpy as np


def settlement_over_depth(cs, cp, depth, sigma, t=10000, p=2.0, ocr=1.0):
    """
    Koppejan settlement calculation. Parameters, p and sigma should be the same units.

    Parameters
    ----------
    cs : array
        C's secondary stiffness over depth
    cp : array
        C'p; primary stiffness over depth.
    depth : array
    sigma : array
        Grain pressure in [MPa]
    t : int
        Time in days.
    p : float
        Load in [MPa].
    ocr : float
        Over consolidation ratio.

    Returns
    -------
    settlement : array
        Cumulative settlement over depth in [m]
    """

    eps = np.nan_to_num(koppejan(cs, cp, sigma, t, p, ocr))

    dh = eps[:-1] * np.diff(depth)
    cum_dh = np.cumsum(dh[::-1])[::-1]
    return np.append(cum_dh, cum_dh[-1])


def koppejan(cs, cp, sigma, t=10000, p=2.0, ocr=1.0):
    """
    Koppejan settlement strain.

    Parameters
    ----------
    cs : float
        C's; secondary stiffness over depth
    cp : float
        C'p; primary stiffness over depth.
    sigma : float
        Grain pressure
    t : int
        Time in days.
    p : float
        Load in MPa.
    ocr : float
        Over consolidation ratio.

    Returns
    -------
    eps : float
        strain
    """
    c_factor = 4.0
    return (1.0 / (cp * c_factor) + 1.0 / (cs * c_factor) * np.log10(t)) * np.log(
        sigma * ocr / sigma
    ) + (1.0 / cp + 1.0 / cs * np.log10(t)) * np.log((sigma + p) / (sigma * ocr))
