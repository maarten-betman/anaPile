from anapile.pressure.compose import single_pile
import pandas as pd
import numpy as np
from pygef import ParseGEF
import matplotlib.pyplot as plt

# read cpt and layers
gef = ParseGEF('../tests/files/example.gef')
gef.groundwater_level = 1
layer_table = pd.read_csv('../tests/files/layer_table.csv')

pile_tip_levels = np.linspace(5, gef.df.depth.max(), 20)

pile_width = 0.25
circum = pile_width * 4
area = pile_width**2
d_eq = 1.13 * pile_width

rb = []
rs = []
negative_friction = []
ptls = []
for ptl in pile_tip_levels:
    try:
        rb_, rs_, negative_friction_ = single_pile(gef, layer_table, ptl, d_eq, circum, area)
        rb.append(rb_)
        rs.append(rs_)
        negative_friction.append(negative_friction_)
        ptls.append(ptl)

    except IndexError:
        pass

rb = np.array(rb)
rs = np.array(rs)
negative_friction = np.array(negative_friction)

fig = plt.figure(figsize=(8, 12))
y_lim = [-gef.df.depth.max(), -gef.df.depth.min()]
ax = fig.add_subplot(1, 3, 1)
y = -gef.df.depth.values
ax.plot(gef.df.qc.values, y)
ax.set_ylim(y_lim)
plt.grid()

ax = fig.add_subplot(1, 3, 2)
ax.plot(gef.df.friction_number.values, y)
ax.set_ylim(y_lim)
plt.grid()

y = -np.array(ptls)
ax = fig.add_subplot(1, 3, 3, label='rb')
ax.plot(rb, y, label='rb')
ax.plot(rs, y, label='rs')
ax.plot(negative_friction, y, label='negative_friction')
ax.plot(rb + rs - negative_friction, y, label=r'$R_{,cal}$')
ax.set_ylim(y_lim)
plt.grid()
plt.legend(loc='upper right')
plt.show()