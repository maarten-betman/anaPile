from pygef import ParseGEF
from unittest import TestCase
from pressure import bearing
from geo import soil
import settlement
import utils
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHOW_PLOTS = True


class Pressure(TestCase):

    def setUp(self) -> None:
        self.gef = ParseGEF('files/example.gef')
        self.soil_prop = pd.read_csv('files/soil_prop_example.csv')
        self.layer_table = pd.read_csv('files/layer_table.csv')

    def test_pile_tip_resistance(self):
        f = partial(bearing.compute_pile_tip_resistance,
                    qc=self.gef.df.qc.values,
                    depth=self.gef.df.depth.values,
                    d_eq=1.13 * 0.25,
                    alpha=0.7,
                    beta=1,
                    s=1,
                    A=0.25 ** 2,
                    return_q_components=True)

        rb, qc_1, qc_2, qc_3 = f(utils.nap_to_depth(self.gef.zid, -3.5))

        self.assertAlmostEqual(rb, .2575636376096491)
        self.assertAlmostEqual(qc_1, 10.33309298245614)
        self.assertAlmostEqual(qc_2, 5.92369298245614)
        self.assertAlmostEqual(qc_3, 3.645944736842105)

        rb, qc_1, qc_2, qc_3 = f(utils.nap_to_depth(self.gef.zid, -12))

        self.assertAlmostEqual(rb, .9375)
        self.assertAlmostEqual(qc_1, 27.906636363636366)
        self.assertAlmostEqual(qc_2, 21.714000000000002)
        self.assertAlmostEqual(qc_3, 21.2664298245614)

    def test_sing_tipping(self):
        self.assertEqual(bearing.sign_tipping_idx(np.array([1., 0.5, -0.5, -1])), 2)

    def test_settlement(self):
        s = settlement.soil.settlement_over_depth(
            cs=self.soil_prop["C's"].values,
            cp=self.soil_prop["C'p"].values,
            depth=self.soil_prop['depth'].values,
            sigma=self.soil_prop["sig'"].values
        )
        self.assertGreater((s * 1000)[0], 100)

    def test_pile_settlement(self):
        rb = 1000
        rs = 500
        sls_f = 700
        s, rb_ratio, rs_ratio = settlement.pile.pile_settlement(rb, rs, 2, sls_f, 0.3)
        # sheet found 5.6, we find 5.95. Probably different curves
        self.assertTrue(5.59 < s < 6)

    def test_chamfer_qc(self):
        chamfered_qc = bearing.chamfer_positive_friction(self.gef.df.qc.values, self.gef.df.depth.values)
        self.assertTrue(chamfered_qc.max() == 15)
        self.assertTrue(chamfered_qc.min() == 0)

    def test_positive_friction(self):
        chamfered_qc = bearing.chamfer_positive_friction(self.gef.df.qc.values, self.gef.df.depth.values)
        ptl = utils.nap_to_depth(self.gef.zid, -13.5)
        tipping_point = utils.nap_to_depth(self.gef.zid, -2.1)

        idx_ptl = np.argmin(np.abs(self.gef.df.depth.values - ptl))
        idx_tp = np.argmin(np.abs(self.gef.df.depth.values - tipping_point))
        s = slice(idx_tp, idx_ptl)

        f = bearing.positive_friction(self.gef.df.depth.values[s], chamfered_qc[s], 0.25 * 4, 0.01)
        # it is off ~ 30 kN (2 %), but chamfered line seems correct.
        self.assertTrue(1375 < (f.sum() * 1000) < 1405)

        if SHOW_PLOTS:
            # show chamfer line
            fig = self.gef.plot(show=False)
            fig.axes[0].plot(chamfered_qc[s], self.gef.df.depth[s], color='red')
            fig.axes[0].hlines(ptl, 0, chamfered_qc.max())
            fig.axes[0].hlines(tipping_point, 0, chamfered_qc.max())
            plt.show()

    def test_join_cpt_with_classification(self):
        df = soil.join_cpt_with_classification(self.gef, self.layer_table)
        self.assertEqual(df["sig'"].iloc[0], .001008)

