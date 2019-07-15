from pygef import nap_to_depth
from anapile.pressure import bearing
from anapile.geo import soil
import numpy as np


def single_pile(gef, layer_table, pile_tip_level_nap, d_eq, circum, area, alpha_p=0.7, beta_p=1, s=1, gamma_m=1):
    # backfill cpt values with soil parameters
    df = soil.join_cpt_with_classification(gef, layer_table)

    # pile tip level -13.5 m
    ptl = nap_to_depth(gef.zid, pile_tip_level_nap)

    # find the index of the pile tip
    idx_ptl = np.argmin(np.abs(gef.df.depth.values - ptl))
    ptl_slice = slice(0, idx_ptl)

    # assume the tipping point is on top of the sand layers.
    tipping_point = soil.find_last_sand_layer(df.depth.values[ptl_slice], df.soil_code[ptl_slice])

    idx_tp = np.argmin(np.abs(df.depth.values - tipping_point))
    negative_friction_slice = slice(0, idx_tp)

    negative_friction = (bearing.negative_friction(depth=df.depth.values[negative_friction_slice],
                                                   grain_pressure=df.grain_pressure.values[negative_friction_slice],
                                                   circum=circum,
                                                   phi=df.phi[negative_friction_slice],
                                                   gamma_m=gamma_m)
                         ).sum() * 1000

    positive_friction_slice = slice(idx_tp, idx_ptl)

    chamfered_qc = bearing.chamfer_positive_friction(df.qc.values[positive_friction_slice],
                                                     gef.df.depth.values[positive_friction_slice])

    rs = (bearing.positive_friction(depth=df.depth.values[positive_friction_slice],
                                    chamfered_qc=chamfered_qc,
                                    circum=circum,
                                    alpha_s=0.01)
          ).sum() * 1000

    rb = (bearing.compute_pile_tip_resistance(ptl=ptl,
                                              qc=df.qc.values,
                                              depth=df.depth.values,
                                              d_eq=d_eq,
                                              alpha=alpha_p,
                                              beta=beta_p,
                                              s=s,
                                              area=area)
          ) * 1000
    return rb, rs, negative_friction
