from anapile.settlement import fitted_curves
from scipy import interpolate
from scipy.optimize import root_scalar
import numpy as np


def pile_settlement(rb, rs, curve_type, pile_force_sls, deq):
    """
    Determine the settlement of the pile at the base (pile tip level).

    Parameters
    ----------
    rb : float
        Resistance at the base in [MN]
    rs : float
        Resistance at the shaft in [MN]
    curve_type : int
        Which type of curve to use, determined by the pile type.
    pile_force_sls : float
        Pile force in the SLS in [MN]
    deq : float
        Equivalent diameter in [m]

    Returns
    -------
    out : tuple
        Settlement at the base of the pile in [m],
        ratio of the base resistance,
        ratio of the shaft resistance

    """
    rb *= 1e3
    rs *= 1e3
    pile_force_sls *= 1e3
    if curve_type == 1:
        rs_ratio = fitted_curves.rs_rs_max_type_1
        rb_ratio = fitted_curves.rb_rb_max_type_1
    elif curve_type == 2:
        rs_ratio = fitted_curves.rs_rs_max_type_2
        rb_ratio = fitted_curves.rb_rb_max_type_2
    else:
        rs_ratio = fitted_curves.rs_rs_max_type_3
        rb_ratio = fitted_curves.rb_rb_max_type_3

    # settlement
    ds_rs = fitted_curves.sbi
    ds_rb = fitted_curves.sbi_deq / deq

    # forces
    rb_a = rb_ratio * rb
    rs_a = rs_ratio * rs

    f_rs = interpolate.interp1d(ds_rs, rs_a)
    f_rb = interpolate.interp1d(ds_rb, rb_a)

    def optim_f(x):
        return f_rs(x) + f_rb(x) - pile_force_sls

    sol = root_scalar(optim_f, bracket=[0, 900], method="brentq")

    return sol.root / 1000, f_rb(sol.root) / rb, f_rs(sol.root) / rs


def elastic_elongation(
    elastic_modulus, area, pile_force_sls, depth, negative_friction, positive_friction
):
    # MN to stress
    # * 1e6 -> N / (area * 1e6) mm2
    # MN / m2

    # forces along pile
    forces = (
        np.ones_like(depth) * pile_force_sls + positive_friction - negative_friction
    )
    stress = forces / area
    strain = stress / elastic_modulus

    h = np.diff(depth)
    h = np.r_[depth[0], h]
    return strain * h
