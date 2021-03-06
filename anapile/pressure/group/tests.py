from unittest import TestCase
import os
from anapile.pressure.group import PileGroupInPlane, PileGroup
from pygef import ParseGEF
import numpy as np
import pandas as pd


class PileGroupTest(TestCase):
    pile_width = 0.25
    circum = pile_width * 4
    area = pile_width ** 2
    d_eq = 1.13 * pile_width

    def setUp(self) -> None:
        self.cpts = []
        self.layer_tables = []
        basedir = "data/cpt-grid/"
        for cpt_path in os.listdir(basedir):
            self.cpts.append(ParseGEF(basedir + cpt_path))
            self.layer_tables.append(
                pd.read_csv("data/layer_tables_grid/" + cpt_path.replace("gef", "csv"))
            )

    def test_group_calculation(self):
        pg = PileGroupInPlane(
            self.cpts,
            self.layer_tables,
            pile_calculation_kwargs={
                "d_eq": self.d_eq,
                "circum": self.circum,
                "area": self.area,
                "pile_load": 1000,
                "soil_load": 10,
            },
        )
        self.assertTrue(len(pg.run_pile_calculations(-20)) == len(self.cpts))

    def test_single_pile_group_iters(self):
        """
        Runs an iteration with n_groups == n_cpts
        """
        pg = PileGroupInPlane(
            self.cpts,
            self.layer_tables,
            pile_calculation_kwargs={
                "d_eq": self.d_eq,
                "circum": self.circum,
                "area": self.area,
                "pile_load": 1000,
                "soil_load": 10,
            },
        )
        pg.run_pile_calculations(-20)
        pg.groups = np.arange(len(self.cpts))
        rc_k, variation_coefficients, valid = pg.run_group_calculation()
        self.assertTrue(variation_coefficients.sum() == 0)
        self.assertTrue(valid)
        pg.plot_overview_in_plane()
        pg.plot_group()

    def test_optimization(self):
        pg = PileGroupInPlane(
            self.cpts,
            self.layer_tables,
            pile_calculation_kwargs={
                "d_eq": self.d_eq,
                "circum": self.circum,
                "area": self.area,
                "pile_load": 1000,
                "soil_load": 10,
            },
        )
        pg.run_pile_calculations(-20)
        rc_k, variation_coefficients, valid = pg.optimize()
        pg.plot_overview_in_plane()
        self.assertTrue(valid, pg._valid_group_configuration())
        self.assertAlmostEqual(19598.4095, rc_k.sum(), places=3)
        self.assertAlmostEqual(0.7093, variation_coefficients.sum(), places=3)

    def test_optimization_depth(self):
        pg = PileGroup(
            self.cpts,
            self.layer_tables,
            pile_calculation_kwargs={
                "d_eq": self.d_eq,
                "circum": self.circum,
                "area": self.area,
                "pile_load": 1000,
                "soil_load": 10,
            },
        )

        max_depth = max(
            [cpt.df.elevation_with_respect_to_NAP.min() for cpt in self.cpts]
        )
        pile_tip_level = np.linspace(-5, max_depth, 15)
        pg.run_pile_calculations(pile_tip_level)
        pg.optimize()
        pg.plot_overview()
        # pg.plot_group()

