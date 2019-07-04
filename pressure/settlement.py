import numpy as np


def settlement_over_depth(cs, cp, depth, sigma, pile_tip_level=None, t=10000, p=2., ocr=1.):
    """
    Koppejan settlement calculation.

    Parameters
    ----------
    cs : array
        C's secondary stiffness over depth
    cp : array
        C'p; primary stiffness over depth.
    depth : array
    sigma : array
    pile_tip_level : float
        Depth value of pile tip w/ respect to ground level
    t : int
        Time in days.
    p : float
        Load in MPa.
    ocr : float
        Over consolidation ratio.

    Returns
    -------
    settlement : array
        Cumulative settlement over depth
    """

    if pile_tip_level is None:
        pile_tip_level = cs.shape[0]

    # evaluate values ot pile tip level
    cp = cp[:pile_tip_level]
    cs = cs[:pile_tip_level]
    sigma = sigma[:pile_tip_level] * 1e3
    c_factor = 4.

    eps = (1./(cp * c_factor) + 1./(cs * c_factor) * np.log10(t)) * np.log(sigma * ocr / sigma) + \
          (1. / cp + 1. / cs * np.log10(t)) * np.log((sigma + p) / (sigma * ocr))

    eps = np.nan_to_num(eps)

    dh = eps[:pile_tip_level - 1] * np.diff(depth[:pile_tip_level])
    return np.cumsum(dh[::-1])[::-1]
