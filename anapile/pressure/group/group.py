from scipy import stats
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from anapile.pressure.compose import PileCalculationSettlementDriven
from anapile.plot import BasePlot


class PileGroupPlotter(BasePlot):
    coordinates = None
    vor = None
    rcal = None
    variation_coefficient = None

    def plot_group(self, show=True, voronoi=True, figsize=(6, 6), ax=None):
        """

        Parameters
        ----------
        show : bool
         Show the matplotlib plot.
        voronoi : bool
         Plot voronoi cells
        figsize : tuple
            Matplotlib figsize
        ax : matplotlib.axes
            If given plots will be made on this axis.

        Returns
        -------
        fig : matplotlib.pyplot.Figure

        """
        if ax is None:
            self._create_fig(figsize)
            ax = plt.gca()

        if voronoi:
            spatial.voronoi_plot_2d(self.vor, ax)
        else:
            ax.plot(self.coordinates[:, 0], self.coordinates[:, 1], "o")
        ax.set_xlabel("x-coordinates")
        ax.set_ylabel("y-coordinates")
        for i, p in enumerate(self.coordinates):
            ax.text(p[0], p[1], "#%d" % i, ha="center")

        return self._finish_plot(show=show)

    def plot_overview(self, show=True, figsize=(16, 8)):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        self.plot_group(show=False, voronoi=True, ax=ax[0])

        idx = np.arange(len(self.rcal))
        ax[1].scatter(idx, self.rcal)
        ax[1].plot([0, len(self.rcal)], np.ones(2) * np.mean(self.rcal))
        ax[1].set_xticks(idx)
        ax[1].grid()
        ax[1].set_ylabel('Rcal [kN]')
        ax[1].set_xlabel('#')

        for i, v in enumerate(self.mape):
            ax[1].text(idx[i], self.rcal[i], "{:0.2f}".format(v), rotation=45)

        ax[1].set_title(
            "Variation coefficient: {:.0f}%".format(self.variation_coefficient * 100)
        )
        self._finish_plot(show=show)


class PileGroup(PileGroupPlotter):
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
        self.variation_coefficient = None
        self.mape = None

    def run_calculation(self, pile_tip_level):
        self.rcal = []
        kwargs = self.pile_calculation_kwargs.copy()
        for i in range(len(self.cpts)):
            kwargs["cpt"] = self.cpts[i]
            kwargs["layer_table"] = self.layer_tables[i]

            pc = self.pile_calculation(**kwargs)
            pc.run_calculation(pile_tip_level)
            self.rcal.append((pc.rb + pc.rs - pc.nk) * 1000)

        self.variation_coefficient = stats.variation(self.rcal)
        self.mape = np.abs(self.rcal - np.mean(self.rcal)) / np.mean(self.rcal)
        return self.rcal
