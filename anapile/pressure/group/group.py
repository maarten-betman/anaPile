from scipy import stats
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from anapile.pressure.compose import PileCalculationSettlementDriven
from anapile.plot import BasePlot
from anapile.pressure.group.ec_params import xi_3, xi_4
from collections import defaultdict
import itertools
import matplotlib.colors as mcolors
import random
from sklearn import cluster
from sklearn.preprocessing import scale


class PileGroupPlotter(BasePlot):
    # don't wan't succeeding colors to look too much alike
    colors = list(mcolors.XKCD_COLORS.values())
    random.seed(1)
    random.shuffle(colors)

    def __init__(self):
        super().__init__()
        self.coordinates = None
        self.vor = None
        self.rcal = None
        self.mape = None
        self.groups = None

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
        colors : list[str]
            hex color values, used for group plotting.

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
            ax.text(
                p[0],
                p[1],
                f"#{i};G{self.groups[i]}",
                fontsize=10,
                ha="center",
                backgroundcolor=self.colors[self.groups[i]],
            )

        return self._finish_plot(show=show)

    def plot_overview(self, show=True, figsize=(16, 8)):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        plt.suptitle("N groups: {}".format(np.unique(self.groups).shape[0]))
        self.plot_group(show=False, voronoi=True, ax=ax[0])

        idx = np.arange(len(self.rcal))
        ax[1].scatter(idx, self.rcal)
        ax[1].plot(
            [0, len(self.rcal)], np.ones(2) * np.mean(self.rcal), label="Mean $R_{cal}$"
        )
        ax[1].set_xticks(idx)
        ax[1].grid()
        ax[1].set_ylabel("Rcal [kN]")
        ax[1].set_xlabel("#")

        for i, v in enumerate(self.mape):
            ax[1].text(
                idx[i],
                self.rcal[i],
                "{:0.2f}".format(v),
                backgroundcolor=self.colors[self.groups[i]],
                rotation=45,
            )
        plt.legend()
        plt.tight_layout(pad=1.5)
        self._finish_plot(show=show)


class PileGroupInPlane(PileGroupPlotter):
    def __init__(
        self,
        cpts,
        layer_tables,
        pile_calculation_kwargs=dict(soil_load=1, pile_load=750),
        pile_calculation=PileCalculationSettlementDriven,
    ):
        super().__init__()
        self.cpts = np.array(cpts)
        self.coordinates = np.array(list(map(lambda cpt: (cpt.x, cpt.y), cpts)))
        self.layer_tables = layer_tables
        self.pile_calculation_kwargs = pile_calculation_kwargs
        self.pile_calculation = pile_calculation

        self.vor = spatial.Voronoi(self.coordinates)

        # Start off with one group
        n = len(cpts)
        self.groups = np.zeros(n, dtype=int)
        # design value per cpt
        self.rc_k = np.zeros(n)
        self.variation_coefficients = np.zeros(n)

        self.neighbors = defaultdict(set)
        tri = spatial.Delaunay(self.coordinates)

        # idx are the indexes of the neighbouring cpts
        for idx in tri.simplices:
            # find the possible combinations of pairs.
            for i, j in itertools.combinations(idx, 2):
                region_i = set(self.vor.regions[self.vor.point_region[i]]) - {-1}
                region_j = set(self.vor.regions[self.vor.point_region[j]]) - {-1}

                # should at least share 2 points
                # then the edge is shared
                if len(region_i.intersection(region_j)) >= 2:
                    self.neighbors[i].add(j)
                    self.neighbors[j].add(i)

    def run_pile_calculations(self, pile_tip_level):
        self.rcal = np.zeros_like(self.cpts)
        kwargs = self.pile_calculation_kwargs.copy()
        for i in range(len(self.cpts)):
            kwargs["cpt"] = self.cpts[i]
            kwargs["layer_table"] = self.layer_tables[i]

            pc = self.pile_calculation(**kwargs)
            pc.run_calculation(pile_tip_level)
            self.rcal[i] = (pc.rb + pc.rs - pc.nk) * 1000

        self.mape = np.abs(self.rcal - np.mean(self.rcal)) / np.mean(self.rcal)

        return self.rcal

    def run_group_calculation(self, groups=None):
        """
        Determine the design values of the piles, the variation coefficients,
        and check if this group configuration is allowed
        (variation coefficients should be below 0.12)

        Returns
        -------
        results: tuple[np.array, np.array, boolean]
            (rc;k desing values, variation coefficients, allowed group)

        """
        if groups is not None:
            self.groups = groups

        for g in np.unique(self.groups):
            mask = self.groups == g

            # no. of cpts in group
            n = len(self.rcal[mask])
            xi3 = xi_3[n]
            xi4 = xi_4[n]

            self.rc_k[mask] = min(
                np.mean(self.rcal[mask]) / xi3, np.min(self.rcal[mask]) / xi4
            )

            self.variation_coefficients[mask] = stats.variation(self.rcal[mask])

        return (
            self.rc_k,
            self.variation_coefficients,
            np.all(self.variation_coefficients <= 0.12)
            and self._valid_group_configuration(),
        )

    def _valid_group_configuration(self):

        # Groups should have at least one neighbor from the same group,
        # unless it is the single member of the group
        for i in range(len(self.cpts)):
            # to which group do I belong?
            my_group = self.groups[i]

            # who are my group members?
            group_members = set(np.argwhere(self.groups == my_group).flatten()) - {i}

            # Being the only one in a group is allowed.
            if len(group_members) == 0:
                continue
            # Do I have at least on neighbor as group member
            if len(self.neighbors[i].intersection(group_members)) < 1:
                return False
        return True

    def optimize(self, seed=1):
        """
        Optimized
        Parameters
        ----------
        seed : int
            Fix random seed.

        Returns
        -------
        rc_k, variation_coefficients, valid : tuple[np.array, np.array, bool]
        """
        np.random.seed(seed)
        rc_k, variation_coefficients, valid = self.run_group_calculation()
        # add rcal twice so that extra weight, will be on those columns
        # and iterate over multiple strides of this x matrix
        x = scale(np.hstack([self.coordinates, self.rcal[:, None], self.rcal[:, None]]))
        n = 1
        while not valid:
            n += 1
            for m in [
                cluster.KMeans(n),
                cluster.AgglomerativeClustering(n, linkage="ward"),
                cluster.AgglomerativeClustering(n, linkage="average"),
                cluster.AgglomerativeClustering(n, linkage="single"),
            ]:
                for x_ in (x[:, :-2], x[:, :-1], x):
                    m.fit(x_)
                    self.groups = m.labels_
                    rc_k, variation_coefficients, valid = self.run_group_calculation()
                    if valid:
                        return rc_k, variation_coefficients, valid
        return rc_k, variation_coefficients, valid


class PileGroup(PileGroupInPlane):
    def __init__(
        self,
        cpts,
        layer_tables,
        pile_calculation_kwargs=dict(soil_load=1, pile_load=750),
        pile_calculation=PileCalculationSettlementDriven,
        pile_load=900,
    ):
        super().__init__(cpts, layer_tables, pile_calculation_kwargs, pile_calculation)
        self.pile_tip_level = None
        self.pile_load = pile_load
        self.allowed_depths = None
        self.proposal_depths_idx_ = None
        self.rcal = None  # (n_ptl, n_cpts)
        self.group_depths = np.zeros(len(cpts))

    def run_pile_calculations(self, pile_tip_level):
        self.rcal = np.zeros((len(pile_tip_level), len(self.cpts)))
        kwargs = self.pile_calculation_kwargs.copy()
        for i in range(len(self.cpts)):
            kwargs["cpt"] = self.cpts[i]
            kwargs["layer_table"] = self.layer_tables[i]

            pc = self.pile_calculation(**kwargs)
            pc.run_calculation(pile_tip_level)
            self.rcal[:, i] = (pc.rb_ + pc.rs_ - pc.nk_) * 1000

        mean_over_cpt = np.mean(self.rcal, axis=0)
        assert mean_over_cpt.shape[0] == len(self.cpts)
        self.mape = np.abs(self.rcal - mean_over_cpt) / mean_over_cpt
        self.pile_tip_level = pc.pile_tip_level_
        self.allowed_depths = self.rcal > self.pile_load
        self.proposal_depths_idx_ = np.argsort(self.allowed_depths.sum(1))[::-1]
        return self.rcal

    def run_group_calculation(self, groups=None):
        if groups is not None:
            self.groups = groups

        for g in np.unique(self.groups):
            mask = self.groups == g

            # no. of cpts in group
            n = len(self.cpts[mask])
            xi3 = xi_3[n]
            xi4 = xi_4[n]

            # rcal values of this group
            rcal = self.rcal[:, mask]

            r_ck = np.min(
                np.concatenate(
                    [
                        (np.mean(rcal, axis=1) / xi3)[:, None],
                        (np.min(rcal, axis=1) / xi4)[:, None],
                    ],
                    axis=1,
                ),
                axis=1,
            )

            variation_coefficients = stats.variation(rcal, axis=1)

            # find first depth that is valid
            strength_condition = (r_ck - self.pile_load) > 0
            variation_condition = variation_coefficients < 0.12
            cond = strength_condition & variation_condition

            if not np.any(cond):
                # No solution here
                return None, None, False
            idx = np.argmax(cond)

            self.group_depths[mask] = self.pile_tip_level[idx]
            self.rc_k[mask] = r_ck[idx]
            self.variation_coefficients[mask] = stats.variation(rcal[idx])

        return (
            self.rc_k,
            self.variation_coefficients,
            self._valid_group_configuration()
        )

    def optimize(self, seed=1, scale_geometry=3):
        rc_k, variation_coefficients, valid = self.run_group_calculation()

        x = scale(np.hstack([self.coordinates, self.rcal.T]))

        x[:, :2] = x[:, :2] * scale_geometry * 3
        n = 1
        while not valid:
            n += 1
            for m in [
                cluster.KMeans(n),
                cluster.AgglomerativeClustering(n, linkage="ward"),
                cluster.AgglomerativeClustering(n, linkage="average"),
                cluster.AgglomerativeClustering(n, linkage="single"),
            ]:
                m.fit(x)
                self.groups = m.labels_
                rc_k, variation_coefficients, valid = self.run_group_calculation()
                if valid:
                    return rc_k, variation_coefficients, valid

        return rc_k, variation_coefficients, valid

