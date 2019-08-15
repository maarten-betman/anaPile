from anapile.pressure.compose import PileCalculationLowerBound
import pandas as pd
import numpy as np
from pygef import ParseGEF

# read cpt and layers
gef = ParseGEF("../tests/files/example.gef")
gef.groundwater_level = 1
layer_table = pd.read_csv("../tests/files/layer_table.csv")

pile_width = 0.25
circum = pile_width * 4
area = pile_width ** 2
d_eq = 1.13 * pile_width
alpha_p = 0.7
beta_p = 1
pile_factor_s = 1

# plot single calculation
cal = PileCalculationLowerBound(gef, d_eq, circum, area, layer_table)
cal.plot_pile_calculation(pile_tip_level=8, show=True)

# plot range
ptl_range = np.linspace(-5, gef.df.elevation_with_respect_to_NAP.min(), num=20)
cal.plot_pile_calculation_range(
    pile_tip_level=np.linspace(-3, gef.df.elevation_with_respect_to_NAP.min(), num=20),
    show=True,
)
