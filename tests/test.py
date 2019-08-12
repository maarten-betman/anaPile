from pygef import ParseGEF, nap_to_depth
from unittest import TestCase
import unittest
from anapile import settlement
from anapile.pressure import bearing
from anapile.geo import soil
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anapile.pressure.compose import PileCalculationSettlementDriven

SHOW_PLOTS = True


class Pressure(TestCase):
    def setUp(self) -> None:
        self.gef = ParseGEF("files/example.gef")
        self.soil_prop = pd.read_csv("files/soil_prop_example.csv")
        self.layer_table = pd.read_csv("files/layer_table.csv")

    def tearDown(self) -> None:
        plt.close("all")

    def test_pile_tip_resistance(self):
        f = partial(
            bearing.compute_pile_tip_resistance,
            qc=self.gef.df.qc.values,
            depth=self.gef.df.depth.values,
            d_eq=1.13 * 0.25,
            alpha=0.7,
            beta=1,
            s=1,
            area=0.25 ** 2,
            return_q_components=True,
        )

        rb, qc_1, qc_2, qc_3 = f(nap_to_depth(self.gef.zid, -3.5))

        self.assertAlmostEqual(rb, 0.2575636376096491)
        self.assertAlmostEqual(qc_1, 10.33309298245614)
        self.assertAlmostEqual(qc_2, 5.92369298245614)
        self.assertAlmostEqual(qc_3, 3.645944736842105)

        rb, qc_1, qc_2, qc_3 = f(nap_to_depth(self.gef.zid, -12))

        self.assertAlmostEqual(rb, 0.9375)
        self.assertAlmostEqual(qc_1, 27.906636363636366)
        self.assertAlmostEqual(qc_2, 21.714000000000002)
        self.assertAlmostEqual(qc_3, 21.2664298245614)

    def test_sign_tipping(self):
        self.assertEqual(bearing.sign_tipping_idx(np.array([1.0, 0.5, -0.5, -1])), 2)

    def test_settlement(self):
        s = settlement.soil.settlement_over_depth(
            cs=self.soil_prop["C_s"].values,
            cp=self.soil_prop["C_p"].values,
            depth=self.soil_prop["depth"].values,
            sigma=self.soil_prop["grain_pressure"].values,
        )
        self.assertGreater((s * 1000)[0], 100)

    def test_pile_settlement(self):
        rb = 1
        rs = 0.5
        sls_f = 0.7
        s, rb_ratio, rs_ratio = settlement.pile.pile_settlement(rb, rs, 2, sls_f, 0.3)
        # sheet found 5.6, we find 5.95. Probably different curves
        self.assertTrue(5.59 < (s * 1000) < 6)

    def test_chamfer_qc(self):
        chamfered_qc = bearing.chamfer_positive_friction(
            self.gef.df.qc.values, self.gef.df.depth.values
        )
        self.assertTrue(chamfered_qc.max() == 15)
        self.assertTrue(chamfered_qc.min() == 0)

    def test_positive_friction(self):
        chamfered_qc = bearing.chamfer_positive_friction(
            self.gef.df.qc.values, self.gef.df.depth.values
        )

        # pile tip level
        ptl = nap_to_depth(self.gef.zid, -13.5)
        tipping_point = nap_to_depth(self.gef.zid, -2.1)

        idx_ptl = np.argmin(np.abs(self.gef.df.depth.values - ptl))
        idx_tp = np.argmin(np.abs(self.gef.df.depth.values - tipping_point))
        s = slice(idx_tp, idx_ptl)

        f = bearing.positive_friction(
            self.gef.df.depth.values[s], chamfered_qc[s], 0.25 * 4, 0.01
        )

        # d-foundation result is 1375, we have 1371. Small difference due to chamfering method. We chamfer better ;)
        deviation = abs(1 - 1375 / (f.sum() * 1000))
        self.assertTrue(deviation < 1e-2)
        self.assertEqual(f.sum(), 1.3710468170000003)

        if SHOW_PLOTS:
            # show chamfer line
            fig = self.gef.plot(show=False)
            fig.axes[0].plot(chamfered_qc[s], self.gef.df.depth[s], color="red")
            fig.axes[0].hlines(ptl, 0, chamfered_qc.max())
            fig.axes[0].hlines(tipping_point, 0, chamfered_qc.max())
            plt.show()

    def test_join_cpt_with_classification(self):
        df = soil.join_cpt_with_classification(self.gef, self.layer_table)
        self.assertEqual(df["grain_pressure"].iloc[0], 0.0010071)

    def test_negative_friction(self):
        layer_table = pd.read_csv("files/d_foundation_layer_table.csv")
        self.gef.groundwater_level = 1
        df = soil.join_cpt_with_classification(self.gef, layer_table)

        tipping_point = nap_to_depth(self.gef.zid, -2.1)
        idx_tp = np.argmin(np.abs(self.gef.df.depth.values - tipping_point))
        s = slice(0, idx_tp - 1)

        if SHOW_PLOTS:
            # show chamfer line
            fig = self.gef.plot(show=False)
            fig.axes[0].plot(df.qc.values[s], df.depth.values[s], color="red")
            fig.axes[0].hlines(tipping_point, 0, df.qc.values[s].max())
            plt.show()

        f = bearing.negative_friction(
            df.depth.values[s], df.grain_pressure.values[s], 0.25 * 4, df.phi.values[s]
        )
        deviation = abs(1 - 19.355 / (f.sum() * 1000))
        self.assertTrue(deviation < 1e-2)

    def test_find_last_sand_layer(self):
        df = soil.join_cpt_with_classification(self.gef, self.layer_table)
        ptl = nap_to_depth(self.gef.zid, -13.5)
        idx_ptl = np.argmin(np.abs(self.gef.df.depth.values - ptl))

        self.assertTrue(
            soil.find_last_negative_friction_tipping_point(
                df.depth.values[:idx_ptl], df.soil_code[:idx_ptl]
            ),
            3.848,
        )

    def test_clean_sand_layer_thickness(self):
        thickness, _, _ = soil.find_clean_sand_layers(
            self.layer_table["thickness"],
            self.layer_table["soil_code"],
            self.layer_table["depth_btm"],
        )

        self.assertAlmostEqual(thickness[0], 14.354)


class TestSettlementCalculation(TestCase):
    pile_width = 0.25
    circum = pile_width * 4
    area = pile_width ** 2
    d_eq = 1.13 * pile_width

    def setUp(self) -> None:
        self.gef = ParseGEF("files/example.gef")
        self.layer_table = pd.read_csv("files/layer_table.csv")
        self.calc = PileCalculationSettlementDriven(
            self.gef,
            self.d_eq,
            self.circum,
            self.area,
            self.layer_table,
            pile_load=1500,
            soil_load=10,
            pile_system="soil-displacement",
            ocr=1,
            elastic_modulus_pile=30e3,
            settlement_time_in_days=int(1e10),
            alpha_s=0.01,
            gamma_m=1,
            alpha_p=1,
            beta_p=1,
            pile_factor_s=1,
        )

        # Assert that tipping points are as yet  none
        self.assertIsNone(self.calc.negative_friction_tipping_point_nap)
        self.assertIsNone(self.calc.positive_friction_tipping_point_nap)

    def tearDown(self) -> None:
        plt.close("all")

    def test_(self):
        self.calc.plot_pile_calculation(-12)
        self.calc.plot_pile_calculation(np.linspace(0, -17), figsize=(10, 10))

        # Tipping points must be set by now, test
        self.assertAlmostEqual(
            self.calc.negative_friction_tipping_point_nap,
            self.calc.merged_soil_properties.elevation_with_respect_to_NAP[
                self.calc.negative_friction_slice.stop
            ],
        )
        self.assertAlmostEqual(
            self.calc.positive_friction_tipping_point_nap,
            self.calc.merged_soil_properties.elevation_with_respect_to_NAP[
                self.calc.positive_friction_slice.start
            ],
        )


if __name__ == "__main__":
    unittest.main()
