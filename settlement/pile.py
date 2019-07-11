from settlement import fitted_curves
from scipy import interpolate
from scipy.optimize import root_scalar


def pile_settlement(rb, rs, curve_type, pile_force_sls, deq):
    """
    Determine the settlement of the pile at the base.

    Parameters
    ----------
    rb : float
        Resistance at the base
    rs : float
        Resistance at the shaft
    curve_type : int
        Which type of curve to use, determined by the pile type.
    pile_force_sls : float
        Pile force in the SLS in [MN]
    deq : float
        Equivalent diameter in [m]

    Returns
    -------
    out : tuple
        Settlement at the base of the pile in [mm], ratio of hte base resistance, ratio of the shaft resistance

    """
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

    sol = root_scalar(optim_f, bracket=[0, 900], method='brentq')

    return sol.root, f_rb(sol.root) / rb, f_rs(sol.root) / rs
