from pygef import ParseGEF
from unittest import TestCase
from pressure import bearing
import utils
from functools import partial


class Pressure(TestCase):

    def setUp(self) -> None:
        self.gef = ParseGEF('files/example.gef')

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

