from pygef import depth_to_nap, nap_to_depth
from anapile.pressure import bearing
from anapile.geo import soil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PileCalculation:
    def __init__(
        self,
        cpt,
        d_eq,
        circum,
        area,
        layer_table,
        alpha_s=0.01,
        gamma_m=1.0,
        alpha_p=0.7,
        beta_p=1.0,
        pile_factor_s=1.0,
    ):
        """

        Parameters
        ----------
        cpt : pygef.ParseGEF
            Parsed cpt gef file.
        d_eq : float
            Equivalent diameter. Is either the diameter of a circle or 1.13 * width of a square.
        circum : float
            Circumference of pile. Either 4 * width or pi * diameter.
        area: float
            Area of pile tip.
        layer_table : DataFrame
            Classification as defined in `tests/files/layer_table.csv`
        alpha_s : float
            Alpha s factor used in positive friction calculation.
        gamma_m : float
            Gamma m factor used in negative friction calculation.
        alpha_p : float
            Alpha p factor used in pile tip resistance calculation.
        beta_p : float
            Beta p factor used in pile tip resistance calculation.
        pile_factor_s : float
            Factor s used in pile tip resistance calculation.
        """
        self.cpt = cpt
        self.d_eq = d_eq
        self.circum = circum
        self.area = area
        self.gamma_m = gamma_m
        self.alpha_s = alpha_s
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.pile_factor_s = pile_factor_s
        self.layer_table = layer_table
        self.df = soil.join_cpt_with_classification(cpt, layer_table)
        # current pile tip level results
        self.pile_tip_level = None
        self._idx_ptl = None
        self.negative_friction_slice = None
        self.positive_friction_slice = None
        self.chamfered_qc = None
        self.rb = None
        self.qc1 = None
        self.qc2 = None
        self.qc3 = None
        self.rs = None
        self.rb = None
        self.nk = None
        self.pile_tip_level = None
        # arrays
        self.rb_ = None
        self.qc1_ = None
        self.qc2_ = None
        self.qc3_ = None
        self.rs_ = None
        self.rb_ = None
        self.nk_ = None
        self.pile_tip_level_ = None

    def _set_ptl(self, pile_tip_level):
        """
        Set pile tip level property.
        Parameters
        ----------
        pile_tip_level : float
            Depth value wrt NAP.
        Returns
        -------
        None
        """
        self.pile_tip_level = nap_to_depth(self.cpt.zid, pile_tip_level)
        # find the index of the pile tip
        self._idx_ptl = np.argmin(
            np.abs(self.cpt.df.depth.values - self.pile_tip_level)
        )

    @property
    def pile_tip_level_nap(self):
        if self.pile_tip_level:
            return depth_to_nap(self.pile_tip_level_nap, self.cpt.zid)

    def negative_friction(self, negative_friction_range=None, agg=True):
        """
        Determine negative friction.

        Parameters
        ----------
        negative_friction_range : np.array[float]
            Length == 2.
            Start and end depth of negative friction.

        agg : bool
            Influences return type.
                True: aggregates the friction values in total friction.
                False: Return friction values per cpt layer.

        Returns
        -------
        out : Union[float, np.array[float]]
            unit [MPa]
            See agg parameter
        """
        if negative_friction_range:
            self.negative_friction_slice = det_slice(
                    self.df.depth.values, negative_friction_range
                )
        negative_friction = bearing.negative_friction(
            depth=self.df.depth.values[self.negative_friction_slice],
            grain_pressure=self.df.grain_pressure.values[self.negative_friction_slice],
            circum=self.circum,
            phi=self.df.phi[self.negative_friction_slice],
            gamma_m=self.gamma_m,
        )
        self.nk = negative_friction.sum()
        if agg:
            return self.nk
        return negative_friction

    def positive_friction(
        self, positive_friction_range=None, agg=True, conservative=False
    ):
        """
        Determine shaft friction Rs.

        Parameters
        ----------
        positive_friction_range : np.array[float]
            Length == 2.
            Start and end depth of positive friction.

        agg : bool
            Influences return type.
                True: aggregates the friction values in total friction.
                False: Return friction values per cpt layer.

        conservative : bool

        Returns
        -------
        out : Union[float, np.array[float]]
            unit [MPa]
            See agg parameter
        """
        if positive_friction_range:
            self.positive_friction_slice = det_slice(
                    positive_friction_range, self.df.depth.values
                )
        self.chamfered_qc = bearing.chamfer_positive_friction(
            self.df.qc.values, self.cpt.df.depth.values
        )[self.positive_friction_slice]
        positive_friction = bearing.positive_friction(
            depth=self.df.depth.values[self.positive_friction_slice],
            chamfered_qc=self.chamfered_qc,
            circum=self.circum,
            alpha_s=self.alpha_s,
        )
        self.rs = positive_friction.sum()
        if agg:
            return self.rs
        return positive_friction

    def pile_tip_resistance(self, agg=True):
        """
        Determine pile tip resistance Rb.
        Parameters
        ----------
        agg : bool
            Influences return type.
                True: aggregates the qc_1, qc_2, and qc_3 to Rb
                False: Return R_b, qc_1, qc_2, qc_3

        Returns
        -------
        out : Union[float, tuple[float]]
        unit [MPa]
            See agg parameter
        """
        self.rb, self.qc1, self.qc2, self.qc3 = bearing.compute_pile_tip_resistance(
            ptl=self.pile_tip_level,
            qc=self.df.qc.values,
            depth=self.df.depth.values,
            d_eq=self.d_eq,
            alpha=self.alpha_p,
            beta=self.beta_p,
            s=self.pile_factor_s,
            area=self.area,
            return_q_components=True,
        )
        if agg:
            return self.rb
        else:
            return self.rb, self.qc1, self.qc2, self.qc3

    def plot_pile_calculation(
        self, pile_tip_level, show=True, figsize=(6, 10), **kwargs
    ):
        self.run_calculation(pile_tip_level)
        fig = self.cpt.plot(figsize=figsize, **kwargs, show=False)
        fig.axes[0].plot(
            self.df.qc.values[self.negative_friction_slice],
            self.df.depth.values[self.negative_friction_slice],
            color="red",
        )
        fig.axes[0].plot(
            self.chamfered_qc,
            self.df.depth[self.positive_friction_slice],
            color="lightgreen",
            lw=3,
        )
        fig.axes[0].hlines(self.pile_tip_level, 0, fig.axes[0].get_xlim()[1])

        factor_horizontal = 0.4

        fig.axes[0].text(
            factor_horizontal * fig.axes[0].get_xlim()[1],
            self.pile_tip_level * 1.1,
            "$R_b$: {:3.2f} kN\nptl: {:3.2f} m NAP".format(
                self.rb * 1000, depth_to_nap(self.pile_tip_level, self.cpt.zid)
            ),
        )

        fig.axes[0].text(
            factor_horizontal * fig.axes[0].get_xlim()[1],
            self.df.depth.values[self.negative_friction_slice].mean(),
            "$N_{{friction}}$: {:3.2f} kN".format(self.nk * 1000),
        )
        fig.axes[0].text(
            factor_horizontal * fig.axes[0].get_xlim()[1],
            self.df.depth.values[self.positive_friction_slice].mean(),
            "$R_s$: {:3.2f} kN".format(self.rs * 1000),
        )

        plt.suptitle(
            "$R_{{cal}}$: {:3.2f} kN".format(sum([self.rs, self.rb, -self.nk]) * 1000)
        )

        if show:
            plt.show()
        return fig

    def plot_pile_calculation_range(
        self, pile_tip_level, show=True, figsize=(9, 10), **kwargs
    ):
        self.run_calculation(pile_tip_level)

        fig = plt.figure(figsize=figsize, **kwargs)
        fig.add_subplot(1, 3, 1)
        plt.ylabel("Depth [m]")
        plt.xlabel("qc [MPa]")
        plt.plot(
            self.cpt.df.qc.values, self.cpt.df.elevation_with_respect_to_NAP.values
        )
        plt.grid()
        fig.add_subplot(1, 3, 2)
        plt.xlabel("friction number [%]")
        plt.plot(
            self.cpt.df.friction_number.values,
            self.cpt.df.elevation_with_respect_to_NAP.values,
        )
        plt.grid()
        fig.add_subplot(1, 3, 3)
        plt.plot(
            -self.nk_ * 1e3, self.pile_tip_level_, color="red", label=r"$N_{friction}$"
        )
        plt.plot(
            self.rs_ * 1e3, self.pile_tip_level_, color="lightgreen", label="$R_s$"
        )
        plt.plot(self.rb_ * 1e3, self.pile_tip_level_, color="darkgreen", label="$R_b$")
        plt.plot(
            (np.array(self.rb_) + np.array(self.rs_) - np.array(self.nk_)) * 1e3,
            self.pile_tip_level_,
            label=r"$R_{cal}$",
            lw=3
        )
        plt.xlabel("Force [kN]")
        plt.vlines(
            0,
            self.cpt.df.elevation_with_respect_to_NAP.values.max(),
            self.cpt.df.elevation_with_respect_to_NAP.values.min(),
        )
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.grid()

        if show:
            plt.show()
        return fig

    def calculation_result_table(self, pile_tip_level):
        self.run_calculation(pile_tip_level)
        return pd.DataFrame(
            {
                "pile_tip_level": self.pile_tip_level_,
                "negative_friction": self.nk_ * 1000,
                "Rs": self.rs_ * 1000,
                "Rb": self.rb_ * 1000,
                "Rcal": (self.rs_ + self.rb_ - self.nk_) * 1000,
                "qc1": self.qc1_,
                "qc2": self.qc2_,
                "qc3": self.qc3_,
            }
        )

    def run_calculation(self, pile_tip_level):
        if isinstance(pile_tip_level, (int, float)):
            pile_tip_level = [pile_tip_level]

        self.rb_ = []
        self.qc1_ = []
        self.qc2_ = []
        self.qc3_ = []
        self.rs_ = []
        self.rb_ = []
        self.nk_ = []
        self.pile_tip_level_ = np.array(pile_tip_level)

        for ptl in pile_tip_level:
            self._set_ptl(ptl)
            self.nk_.append(self.negative_friction())
            self.rs_.append(self.positive_friction())
            rb, qc1, qc2, qc3 = self.pile_tip_resistance(agg=False)
            self.rb_.append(rb)
            self.qc1_.append(qc1)
            self.qc2_.append(qc2)
            self.qc3_.append(qc3)
        self.rs_ = np.array(self.rs_)
        self.rb_ = np.array(self.rb_)
        self.nk_ = np.array(self.nk_)


class PileCalculationLowerBound(PileCalculation):
    """
    Pile calculation where pile tip level and friction ranges are determined naively.
    """

    def __init__(
        self,
        cpt,
        d_eq,
        circum,
        area,
        layer_table,
        alpha_s=0.01,
        gamma_m=1.0,
        alpha_p=0.7,
        beta_p=1.0,
        pile_factor_s=1.0,
    ):
        """

        Parameters
        ----------
        cpt : pygef.ParseGEF
            Parsed cpt gef file.
        d_eq : float
            Equivalent diameter. Is either the diameter of a circle or 1.13 * width of a square.
        circum : float
            Circumference of pile. Either 4 * width or pi * diameter.
        area: float
            Area of pile tip.
        layer_table : DataFrame
            Classification as defined in `tests/files/layer_table.csv`
        alpha_s : float
            Alpha s factor used in positive friction calculation.
        gamma_m : float
            Gamma m factor used in negative friction calculation.
        alpha_p : float
            Alpha p factor used in pile tip resistance calculation.
        beta_p : float
            Beta p factor used in pile tip resistance calculation.
        pile_factor_s : float
            Factor s used in pile tip resistance calculation.
        """
        super().__init__(
            cpt,
            d_eq,
            circum,
            area,
            layer_table,
            alpha_s,
            gamma_m,
            alpha_p,
            beta_p,
            pile_factor_s,
        )

    def negative_friction(self, negative_friction_range=None, agg=True):
        """
        Determine negative friction.

        Parameters
        ----------
        negative_friction_range : Union[np.array[float], None]
            Length == 2.
            Start and end depth of negative friction.
            If None given. Determines the negative friction range naively.

        agg : bool
            Influences return type.
                True: aggregates the friction values in total friction.
                False: Return friction values per cpt layer.

        Returns
        -------
        out : Union[float, np.array[float]]
            unit [MPa]
            See agg parameter
        """
        if negative_friction_range is None:
            ptl_slice = slice(0, self._idx_ptl)
            tipping_point = soil.find_last_negative_friction_tipping_point(
                self.df.depth.values[ptl_slice], self.df.soil_code.values[ptl_slice]
            )
            idx_tp = np.argmin(np.abs(self.df.depth.values - tipping_point))
            self.negative_friction_slice = slice(0, idx_tp)
        else:
            self.negative_friction_slice = det_slice(
                self.df.depth.values, negative_friction_range
            )

        return super().negative_friction(negative_friction_range, agg)

    def positive_friction(
        self, positive_friction_range=None, agg=True, conservative=False
    ):
        """
        Determine shaft friction Rs.

        Parameters
        ----------
        positive_friction_range : Union[np.array[float], None]
            Length == 2.
            Start and end depth of positive friction.
            If None given. Determines the positive friction range naively.

        agg : bool
            Influences return type.
                True: aggregates the friction values in total friction.
                False: Return friction values per cpt layer.

        conservative : bool

        Returns
        -------
        out : Union[float, np.array[float]]
            unit [MPa]
            See agg parameter
        """
        if positive_friction_range is None:
            ptl_slice = slice(0, self._idx_ptl)
            if conservative:
                f = soil.find_positive_friction_tipping_point
            else:
                f = soil.find_last_negative_friction_tipping_point
            tipping_point = f(
                self.df.depth.values[ptl_slice], self.df.soil_code.values[ptl_slice]
            )
            idx_tp = np.argmin(np.abs(self.df.depth.values - tipping_point))
            self.positive_friction_slice = slice(idx_tp, self._idx_ptl)

        return super().positive_friction(positive_friction_range, agg, conservative)


def det_slice(single_range, a):
    assert len(single_range) == 2
    idx_start = np.argmin(np.abs(a - single_range[0]))
    idx_end = np.argmin(np.abs(a - single_range[1]))
    return slice(idx_start, idx_end)


