from unittest import TestCase
import os
from anapile.pressure.group import PileGroup
from pygef import ParseGEF
import pandas as pd


class PileGroupTest(TestCase):
    def setUp(self) -> None:
        self.cpts = []
        basedir = "data/cpt-grid/"
        for cpt_path in os.listdir(basedir):
            self.cpts.append(ParseGEF(basedir + cpt_path))
        # wrong layer_table doesn't matter for testing group behaviour
        self.layer_tables = [
            pd.read_csv("../../../tests/files/test_layer_table1.csv") for _ in self.cpts
        ]

    def test_(self):
        pg = PileGroup(self.cpts, self.layer_tables)
        pg.show_group()
