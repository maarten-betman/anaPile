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
        self.rcal = None

    def plot_group(self, show=True, voronoi=True, figsize=(6, 6)):
        """

        Parameters
        ----------
        show : bool
         Show the matplotlib plot.
        voronoi : bool
         Plot voronoi cells
        figsize : tuple
            Matplotlib figsize

        Returns
        -------
        fig : matplotlib.pyplot.Figure

        """
        self._create_fig(figsize)

        if voronoi:
            spatial.voronoi_plot_2d(self.vor, plt.gca())
        else:
            plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], "o")
        plt.xlabel("x-coordinates")
        plt.ylabel("y-coordinates")
        for i, p in enumerate(self.coordinates):
            plt.text(p[0], p[1], "#%d" % i, ha="center")

        return self._finish_plot(show=show)

    def run_calculation(self, pile_tip_level):
        self.rcal = []
        kwargs = self.pile_calculation_kwargs.copy()
        for i in range(len(self.cpts)):
            kwargs['cpt'] = self.cpts[i]
            kwargs['layer_table'] = self.layer_tables[i]

            pc = self.pile_calculation(**kwargs)
            pc.run_calculation(pile_tip_level)
            self.rcal.append((pc.rb + pc.rs - pc.nk) * 1000)
        return self.rcal
