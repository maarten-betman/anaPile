import numpy as np


def chamfer_positive_friction(qc, depth):
    """
    Parameters
    ----------
    qc : array
        CPT's qc values
    depth : array
        Depth of cpt w/ respect to ground level

    Returns
    -------
    qc : array
        Chamfered qc values
    """
    # number of indexes per meter
    n_per_m = int(1 / (depth[1] - depth[0]))  # 1 / dl

    # Max values at 15 MPa.
    qc[qc > 15] = 15

    # find the indexes where qc > 12
    idx = np.where(np.sign(np.array(qc) - 12) > 0)[0]

    if idx.shape[0] > 0:
        # subsequent indexes should have a differentiated value of 1. Where they are not one, we have another layer.
        switch_points = np.where(np.diff(idx) != 1)[0]
        switch_points = np.append(switch_points, idx.shape[0] - 1)
        j = 0

        # count the number of indexes in the layer
        for i in switch_points:
            n = idx[i] - idx[j]

            if n <= n_per_m:
                qc[idx[j]: idx[i] + 1] = 12

            j = i + 1

    return qc


def sign_tipping_idx(arr):
    """
    Find the first index where a positive number becomes negative.

    Parameters
    ----------
    arr : array

    Returns
    -------
    out : array

    """
    sign = np.sign(arr)
    return np.where((np.roll(sign, 1) - sign) != 0)[0]


def positive_friction(df, ppni, relative_dz, circum, alpha_s=0.01, indexes=None, return_qc=False):
    """

    :param df:
    :param ppni: (int) Pile base level index.
    :param relative_dz: (array) (setting pile - setting soil)
    :param circum: (flt) Circumference shaft.
    :param alpha_s: (flt) factor
    :param indexes: (array) tipping points from positive to negative and vice versa.
    :param return_qc: (bool) Return the chamfered qc values used in the calculation.
    :return: (array)
    """

    if indexes is None:
        indexes = sign_tipping_idx(relative_dz)
    l = df.l.values[:ppni]
    qc = chamfer_positive_friction(np.array(df.qc.values[:ppni]), l)

    for i in range(len(indexes) - 1):

        i_start = indexes[i]
        i_stop = indexes[i + 1]

        i_middle = (i_stop - i_start) // 2 + i_start

        if relative_dz[i_middle] < 0:
            qc[i_start: i_stop + 1] = 0

    # axial force
    dN = circum * alpha_s * qc
    pf = -dN[:-1] * np.diff(l)
    if return_qc:
        return pf, qc
    else:
        return pf


def compute_pile_tip_resistance(ptl, qc, depth, d_eq, alpha, beta, s, A, return_q_components=False):
    """

    Parameters
    ----------
    ptl : float
        Pile tip level
    qc : array
        CPT's qc values
    depth : array
        Depth of cpt w/ respect to ground level
    d_eq : float
        Equivalent diameter pile.
    alpha : float
    beta : float
    s : float
    A : float
        Pile tip area.
    return_q_components : bool
        Return separate qI, qII, qIII tracks

    Returns
    -------
    rb : float
        Resistance at pile tip.

    """
    d07 = 0.7 * d_eq + ptl
    d4 = 4 * d_eq + ptl
    d8 = ptl - 8 * d_eq

    # closest indices
    ppni = np.argmin(np.abs(ptl - depth))
    d07i = np.argmin(np.abs(d07 - depth))
    d4i = np.argmin(np.abs(d4 - depth))
    d8i = np.argmin(np.abs(d8 - depth))

    # range from ppn to 4d
    qc_range_ppn_4d = qc[ppni:d4i + 1]

    # cumulative min from 4d back to ppn
    qc2_range = np.fmin.accumulate(qc_range_ppn_4d[::-1])[::-1]

    # array used for determining the minimal rb-max. The minimal mean of both q1 and q2 are decisive
    qc_calc = qc2_range + qc_range_ppn_4d
    qc_calc_mean_range = qc_calc.cumsum() / np.arange(1, qc_calc.shape[0] + 1)

    # index of the minimal mean of both qc1 and qc2
    di = d07i - ppni

    min_idx = np.argmin(qc_calc_mean_range[di:])
    idx = min_idx + di if min_idx > 0 else min_idx + di + 1

    # now the actual mean can be determined
    qc1 = np.mean(qc_range_ppn_4d[:idx])
    qc2 = np.mean(qc2_range[:idx])

    # qc3 starts where qc2 is ended and accumulates the minimum again.
    qc3_range = qc[d8i: ppni + 1].copy()
    qc3_range[-1] = qc2_range[0]
    qc3_range = np.fmin.accumulate(qc3_range[::-1])[::-1]
    qc3 = np.mean(qc3_range)

    rb = min(alpha * beta * s * 0.5 * (0.5 * (qc1 + qc2) + qc3), 15) * A

    if return_q_components:
        return rb, qc1, qc2, qc3
    return rb

