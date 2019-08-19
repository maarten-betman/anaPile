from unittest import TestCase
import os
from anapile.pressure.group import PileGroup
from pygef import ParseGEF
import pandas as pd


class PileGroupTest(TestCase):
    pile_width = 0.25
    circum = pile_width * 4
    area = pile_width ** 2
    d_eq = 1.13 * pile_width

    def setUp(self) -> None:
        self.cpts = []
        basedir = "data/cpt-grid/"
        for cpt_path in os.listdir(basedir):
            self.cpts.append(ParseGEF(basedir + cpt_path))
        # wrong layer_table doesn't matter for testing group behaviour
        self.layer_tables = [
            pd.read_csv("../../../tests/files/test_layer_table1.csv") for _ in self.cpts
        ]

    def test_group_calculation(self):
        pg = PileGroup(
            self.cpts,
            self.layer_tables,
            pile_calculation_kwargs={
                "d_eq": self.d_eq,
                "circum": self.circum,
                "area": self.area,
                "pile_load": 1000,
                "soil_load": 10
            },
        )
        self.assertTrue(len(pg.run_calculation(-12)) == len(self.cpts))

    def test_(self):
        pg = PileGroup(
            self.cpts,
            self.layer_tables,
            pile_calculation_kwargs={
                "d_eq": self.d_eq,
                "circum": self.circum,
                "area": self.area,
                "pile_load": 1000,
                "soil_load": 10
            },
        )
        pg.run_calculation(-12)
        pg.plot_overview()
