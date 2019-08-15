from pygef import ParseGEF, nap_to_depth
from anapile.pressure import bearing
from anapile.geo import soil
import pandas as pd
import numpy as np

# read cpt and layers
gef = ParseGEF("../tests/files/example.gef")
layer_table = pd.read_csv("../tests/files/layer_table.csv")

# we know the groundwater level is 1 m NAP
gef.groundwater_level = 1

# backfill cpt values with soil parameters
df = soil.join_cpt_with_classification(gef, layer_table)

# pile tip level -13.5 m
ptl = nap_to_depth(gef.zid, -13.5)
pile_width = 0.25
circum = pile_width * 4

# find the index of the pile tip
idx_ptl = np.argmin(np.abs(gef.df.depth.values - ptl))
ptl_slice = slice(0, idx_ptl)

# assume the tipping point is on top of the sand layers.
tipping_point = soil.find_last_negative_friction_tipping_point(
    df.depth.values[ptl_slice], df.soil_code[ptl_slice]
)

idx_tp = np.argmin(np.abs(df.depth.values - tipping_point))
negative_friction_slice = slice(0, idx_tp)

negative_friction = (
    bearing.negative_friction(
        depth=df.depth.values[negative_friction_slice],
        grain_pressure=df.grain_pressure.values[negative_friction_slice],
        circum=circum,
        phi=df.phi[negative_friction_slice],
        gamma_m=1,
    )
).sum() * 1000

positive_friction_slice = slice(idx_tp, idx_ptl)

chamfered_qc = bearing.chamfer_positive_friction(df.qc.values, gef.df.depth.values)[
    positive_friction_slice
]

rs = (
    bearing.positive_friction(
        depth=df.depth.values[positive_friction_slice],
        chamfered_qc=chamfered_qc,
        circum=circum,
        alpha_s=0.01,
    )
).sum() * 1000

rb = (
    bearing.compute_pile_tip_resistance(
        ptl=ptl,
        qc=df.qc.values,
        depth=df.depth.values,
        d_eq=1.13 * pile_width,
        alpha=0.7,
        beta=1,
        s=1,
        area=pile_width ** 2,
    )
) * 1000

print(rb, rs, negative_friction)
