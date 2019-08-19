from scipy import stats
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from anapile.pressure.compose import PileCalculationSettlementDriven
from anapile.plot import BasePlot


class PileGroup(BasePlot):
    def __init__(
        self,
        cpts,
        layer_tables,
        pile_calculation_kwargs=dict(soil_load=1, pile_load=750),
        pile_calculation=PileCalculationSettlementDriven,
    ):
        super().__init__()
        self.cpts = cpts
        self.coordinates = np.array(list(map(lambda cpt: (cpt.x, cpt.y), cpts)))
        self.layer_tables = layer_tables
        self.pile_calculation_kwargs = pile_calculation_kwargs
        self.pile_calculation = pile_calculation
        self.vor = spatial.Voronoi(self.coordinates)

    def show_group(self, show=True, voronoi=True, figsize=(6, 6)):
        self._create_fig(figsize)

        if voronoi:
            spatial.voronoi_plot_2d(self.vor, plt.gca())
        else:
            plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], 'o')
        plt.xlabel('x-coordinates')
        plt.ylabel('y-coordinates')
        for i, p in enumerate(self.coordinates):
            plt.text(p[0], p[1], '#%d' % i, ha='center')

        return self._finish_plot(show=show)

