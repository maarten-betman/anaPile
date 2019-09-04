import numpy as np


def compute_pile_tip_resistance(
    ptl, qc, depth, d_eq, alpha, beta, s, area, return_q_components=False
):
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
    area : float
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
    qc_range_ppn_4d = qc[ppni: d4i + 1]

    # Array with average qc for each successive range starting at ptl with bottom increasing to 4d below ptl
    q1_means = np.cumsum(qc_range_ppn_4d) / np.arange(1, qc_range_ppn_4d.shape[0] + 1)

    # Array with average qc for each successive range from 4d below ptl to ptl. For each point in a range the qc
    # is smaller than or equal to the qc of the point below
    q2_means = [sum(np.fmin.accumulate(qc_range_ppn_4d[:n + 1][::-1])) / (n + 1) for n, point in
                enumerate(qc_range_ppn_4d)]

    # Array with (q1+q2)/2 for each point between ptl and 4d below ptl
    q12_means = (q1_means + q2_means) / 2.

    # ending values of qc2 for start of q3
    q2_endings = [np.fmin.accumulate(qc_range_ppn_4d[:n + 1][::-1])[n] for n, point in enumerate(qc_range_ppn_4d)]

    #  The minimum value of q12 should be taken between 0.7D to 4D below ptl
    di = d07i - ppni
    q12i = np.argmin(q12_means[di:])

    qc1 = q1_means[di:][q12i]
    qc2 = q2_means[di:][q12i]

    # qc3 starts where qc2 ended.
    qc3_range = qc[d8i: ppni + 1].copy()
    qc3_range[-1] = q2_endings[di:][q12i]
    qc3_range = np.fmin.accumulate(qc3_range[::-1])[::-1]
    qc3 = np.mean(qc3_range)

    rb = min(alpha * beta * s * 0.5 * (0.5 * (qc1 + qc2) + qc3), 15) * area

    if return_q_components:
        return rb, qc1, qc2, qc3
    return rb


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
    qc = np.array(qc)
    # number of indexes per meter
    n_per_m = int(1 / (depth[1] - depth[0]))  # 1 / dl

    # Max values at 15 MPa.
    qc[qc > 15] = 15

    qc[qc < 2] = 0

    # find the indexes where qc > 15
    idx_gt_15 = np.where(np.sign(np.array(qc) - 14.99) > 0)[0]

    if idx_gt_15.shape[0] > 0:
        # subsequent indexes should have a differentiated value of 1. Where they are not one, we have another layer.
        switch_points = np.where(np.diff(idx_gt_15) != 1)[0]
        # indexes of new layers
        switch_points = np.append(switch_points, idx_gt_15.shape[0] - 1)
        j = 0

        # if layer at 15 MPa < 1 m, chamfer the whole layer to 12 MPa

        for i in switch_points:
            # count the number of indexes in the layer (proxy for depth of layer)
            n = idx_gt_15[i] - idx_gt_15[j]

            # layer is smaller than 1 m at 15 MPa
            if n <= n_per_m:

                # define the indexes of the layer where the thickness was measured
                idx_top = idx_gt_15[j]
                idx_btm = idx_gt_15[i]

                # now find the closest indexes of that layer that are 12 MPa.
                try:
                    idx_closest_12_above = np.argwhere(qc[:idx_top] < 12).flatten()[-1]
                except IndexError:  # we are at the top of the cpt and don't see a 12 MPa above?
                    idx_closest_12_above = idx_top
                try:
                    idx_closest_12_below = (
                        np.argwhere(qc[idx_btm:] < 12).flatten()[0] + idx_btm
                    )
                except IndexError:  # we are at the bottom of cpt and don't see a 12 MPa below?
                    idx_closest_12_below = idx_btm

                # chamfer the layer.
                qc[idx_closest_12_above:idx_closest_12_below] = 12

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
    return np.where((np.roll(sign, 1) - sign) != 0)[0][1]


def positive_friction(depth, chamfered_qc, circum, alpha_s=0.01):
    """
    Only pass masked arrays. You should determine which part of array positive friction occurs.

    Parameters
    ----------
    depth : array
    chamfered_qc : array
        qc values, where layers are chamfered to 12 MPa and 15 MPa
    circum : float
        Circumference of the pile.
    alpha_s : float
        Alpha s factor
    Returns
    -------
    friction_forces : array
        Friction values. Sum for the total friction.
    """
    if len(depth) <= 1:
        return np.array([0])
    shaft_stress = circum * alpha_s * chamfered_qc
    force = shaft_stress[:-1] * np.diff(depth)
    return np.append(force, force[-1])


def negative_friction(
    depth,
    grain_pressure,
    circum,
    phi=None,
    delta=None,
    gamma_m=1.0,
    start_at_ground_level=True,
):
    """
    Only pass masked arrays. You should determine which part of array negative friction occurs.

    Parameters
    ----------
    depth : array
    grain_pressure : array
        Grain pressure.
    phi : array
        If delta is not defined, phi will be used to determine delta by delta = phi * 2 / 3
    delta : array
        Delta values for friction adhesion.
    gamma_m : float
        Reduction factor.
    start_at_ground_level : bool
        If the negative friction starts at ground level, the first value in depth, is the first thickness value.

    Returns
    -------
    friction_forces : array
        Friction values. Sum for the total friction.
    """
    if len(depth) <= 1:
        return np.array([0])
    if delta is None:
        delta = phi * 2 / 3

    k0_tan_d = (1 - np.sin(phi)) * np.tan(delta)
    k0_tan_d[k0_tan_d < 0.25] = 0.25

    h = np.diff(depth)

    if start_at_ground_level:
        h = np.r_[depth[0], h]
    else:
        h = np.append(h, h[-1])
    return gamma_m * circum * k0_tan_d * h * grain_pressure


def reduced_qc_due_to_excavation(qc, grain_pressure, grain_pressure_after, t):
    return qc * (grain_pressure_after / grain_pressure)**t


def grain_pressure_after_excavation(grain_pressure, depth, excavation_depth):
    idx = np.argmin(np.abs(depth - excavation_depth))
    gp = grain_pressure - grain_pressure[idx]
    gp[gp < 0] = 0
    return gp

