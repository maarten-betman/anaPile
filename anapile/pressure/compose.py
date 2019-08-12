from pygef import depth_to_nap, nap_to_depth
from anapile.pressure import bearing
from anapile.geo import soil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from anapile import settlement
from anapile import geo


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
        # merged soil properties
        self.merged_soil_properties = soil.join_cpt_with_classification(
            cpt, layer_table
        )
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

    def _init_calculation(self, pile_tip_level):
        """
        reset calculation results and set parameters.

        Parameters
        ----------
        pile_tip_level : Union[np.array[float], float]
        """
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

    @property
    def pile_tip_level_nap(self):
        if self.pile_tip_level:
            return depth_to_nap(self.pile_tip_level, self.cpt.zid)

    @property
    def negative_friction_tipping_point_nap(self):
        """Tipping point for negative friction wrt. NAP"""
        if self.negative_friction_slice:
            return self.merged_soil_properties.elevation_with_respect_to_NAP[
                self.negative_friction_slice.stop
            ]

    @property
    def positive_friction_tipping_point_nap(self):
        if self.positive_friction_slice:
            return self.merged_soil_properties.elevation_with_respect_to_NAP[
                self.positive_friction_slice.start
            ]

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
                self.merged_soil_properties.depth.values, negative_friction_range
            )
        negative_friction = bearing.negative_friction(
            depth=self.merged_soil_properties.depth.values[
                self.negative_friction_slice
            ],
            grain_pressure=self.merged_soil_properties.grain_pressure.values[
                self.negative_friction_slice
            ],
            circum=self.circum,
            phi=self.merged_soil_properties.phi[self.negative_friction_slice],
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
                positive_friction_range, self.merged_soil_properties.depth.values
            )
        self.chamfered_qc = bearing.chamfer_positive_friction(
            self.merged_soil_properties.qc.values, self.cpt.df.depth.values
        )[self.positive_friction_slice]
        positive_friction = bearing.positive_friction(
            depth=self.merged_soil_properties.depth.values[
                self.positive_friction_slice
            ],
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
            qc=self.merged_soil_properties.qc.values,
            depth=self.merged_soil_properties.depth.values,
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

    def _plot_base(self, pile_tip_level, figsize, n_subplots=3, **kwargs):
        self.run_calculation(pile_tip_level)

        fig = plt.figure(figsize=figsize, **kwargs)
        fig.add_subplot(1, n_subplots, 1)
        plt.ylabel("Depth [m]")
        plt.xlabel("qc [MPa]")
        plt.plot(
            self.cpt.df.qc.values, self.cpt.df.elevation_with_respect_to_NAP.values
        )
        plt.grid()
        fig.add_subplot(1, n_subplots, 2)
        plt.xlabel("friction number [%]")
        plt.plot(
            self.cpt.df.friction_number.values,
            self.cpt.df.elevation_with_respect_to_NAP.values,
        )
        plt.grid()
        return fig

    def _plot_single(self, fig):
        fig.axes[0].plot(
            self.merged_soil_properties.qc.values[self.negative_friction_slice],
            self.merged_soil_properties.elevation_with_respect_to_NAP.values[
                self.negative_friction_slice
            ],
            color="red",
        )

        fig.axes[0].plot(
            self.chamfered_qc,
            self.merged_soil_properties.elevation_with_respect_to_NAP[
                self.positive_friction_slice
            ],
            color="lightgreen",
            lw=3,
        )
        fig.axes[0].hlines(self.pile_tip_level_nap, 0, fig.axes[0].get_xlim()[1])

        factor_horizontal = 0.4

        fig.axes[0].text(
            factor_horizontal * fig.axes[0].get_xlim()[1],
            self.pile_tip_level_nap * 1.1,
            "$R_b$: {:3.2f} kN\nptl: {:3.2f} m NAP".format(
                self.rb * 1000, depth_to_nap(self.pile_tip_level, self.cpt.zid)
            ),
        )

        fig.axes[0].text(
            factor_horizontal * fig.axes[0].get_xlim()[1],
            self.merged_soil_properties.elevation_with_respect_to_NAP.values[
                self.negative_friction_slice
            ].mean(),
            "$N_{{friction}}$: {:3.2f} kN".format(self.nk * 1000),
        )
        fig.axes[0].text(
            factor_horizontal * fig.axes[0].get_xlim()[1],
            self.merged_soil_properties.elevation_with_respect_to_NAP.values[
                self.positive_friction_slice
            ].mean(),
            "$R_s$: {:3.2f} kN".format(self.rs * 1000),
        )

        plt.suptitle(
            "$R_{{cal}}$: {:3.2f} kN".format(sum([self.rs, self.rb, -self.nk]) * 1000)
        )

    def _plot_range(self, fig, n_subplots):
        fig.add_subplot(1, n_subplots, 3)
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
            lw=3,
        )
        plt.xlabel("Force [kN]")
        plt.vlines(
            0,
            self.cpt.df.elevation_with_respect_to_NAP.values.max(),
            self.cpt.df.elevation_with_respect_to_NAP.values.min(),
        )
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.grid()

    def plot_pile_calculation(
        self, pile_tip_level, show=True, figsize=(6, 10), n_subplots=0, **kwargs
    ):
        """

        Parameters
        ----------
        pile_tip_level : Union[np.array[float], float]
            Pile tip level in [m NAP]
        show : bool
            call plt.show()
        figsize : tuple
            matplotlib figure size.
        n_subplots : int
            Number of extra subplots appended to the right of the figure.
        kwargs

        Returns
        -------

        """
        self.run_calculation(pile_tip_level)
        single_level = isinstance(pile_tip_level, (int, float))

        n_subplots = 2 + n_subplots if single_level else 3 + n_subplots

        fig = self._plot_base(pile_tip_level, figsize, n_subplots, **kwargs)

        if single_level:
            self._plot_single(fig)
        else:
            self._plot_range(fig, n_subplots)

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
        """
        Run a calculation and set result attributes.

        Parameters
        ----------
        pile_tip_level : Union[np.array[float], float]
            Pile tip level in [m NAP]
        """
        self._init_calculation(pile_tip_level)
        for ptl in self.pile_tip_level_:
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
                self.merged_soil_properties.depth.values[ptl_slice],
                self.merged_soil_properties.soil_code.values[ptl_slice],
            )
            idx_tp = np.argmin(
                np.abs(self.merged_soil_properties.depth.values - tipping_point)
            )
            self.negative_friction_slice = slice(0, idx_tp)
        else:
            self.negative_friction_slice = det_slice(
                self.merged_soil_properties.depth.values, negative_friction_range
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
                self.merged_soil_properties.depth.values[ptl_slice],
                self.merged_soil_properties.soil_code.values[ptl_slice],
            )
            idx_tp = np.argmin(
                np.abs(self.merged_soil_properties.depth.values - tipping_point)
            )
            self.positive_friction_slice = slice(idx_tp, self._idx_ptl)

        return super().positive_friction(positive_friction_range, agg, conservative)


def det_slice(single_range, a):
    assert len(single_range) == 2
    idx_start = np.argmin(np.abs(a - single_range[0]))
    idx_end = np.argmin(np.abs(a - single_range[1]))
    return slice(idx_start, idx_end)


class PileCalculationSettlementDriven(PileCalculationLowerBound):
    def __init__(
        self,
        cpt,
        d_eq,
        circum,
        area,
        layer_table,
        pile_load,
        soil_load,
        pile_system="soil-displacement",
        ocr=1.0,
        elastic_modulus_pile=30e3,
        settlement_time_in_days=int(10e3),
        alpha_s=0.00,
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
        pile_load : float
            Force on pile in SLS [kN]
            Used to determine settlement of pile w.r.t. soil.
        soil_load : float
            (Fictive) load in [kPa] on soil used to calculation soil settlement.
            This is required and used to determine settlement of pile w.r.t. soil.
        pile_system : str
            - 'soil-distplacement'
            - 'little-soil-displacement'
            - 'drilled pile'
        ocr : float
            Over consolidation ratio, used in soil settlement calculation.
        elastic_modulus_pile : float
            Elastic modulus of the pile, used in settlement calculation.
        settlement_time_in_days : int
            Time t in days, used in Koppejan settlement calculation.
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
        self.pile_load = pile_load
        self.soil_load = soil_load
        self.ocr = ocr
        self.settlement_time_in_days = settlement_time_in_days
        self.pile_system = pile_system
        self.elastic_modulus_pile = elastic_modulus_pile

        # results
        self.settlement_soil = None
        self.settlement_ptl = None

    def soil_settlement(self, grain_pressure=None):
        u2 = geo.soil.estimate_water_pressure(self.cpt, self.merged_soil_properties)[
            : self._idx_ptl
        ]

        if grain_pressure is None:
            grain_pressure = geo.soil.grain_pressure(
                self.merged_soil_properties.depth.values[: self._idx_ptl],
                self.merged_soil_properties.gamma_sat.values[: self._idx_ptl],
                self.merged_soil_properties.gamma.values[: self._idx_ptl],
                u2,
            )

        self.settlement_soil = settlement.soil.settlement_over_depth(
            self.merged_soil_properties.C_s.values[: self._idx_ptl],
            self.merged_soil_properties.C_p.values[: self._idx_ptl],
            self.merged_soil_properties.depth.values[: self._idx_ptl],
            grain_pressure,
            self.settlement_time_in_days,
            self.soil_load / 1e3,
            self.ocr,
        )
        return self.settlement_soil

    def pile_settlement_ptl(self):
        curve_types = {
            "soil-displacement": 1,
            "little-soil-displacement": 2,
            "drilled pile": 3,
        }
        self.settlement_ptl, _, _ = settlement.pile.pile_settlement(
            self.rb,
            self.rs,
            curve_types[self.pile_system],
            min(
                self.rs + self.rb - 0.001, self.pile_load / 1e3
            ),  # cannot apply higher load than capacity
            self.d_eq,
        )
        return self.settlement_ptl

    def _single_iter(
        self,
        positive_friction_parent,
        negative_friction_parent,
        depth,
        original_grain_pressure,
    ):
        # fill array with positive friction and negative friction values.
        # These arrays are used to determine the elastic elongation of the pile.

        # Determine positive friction
        positive_friction = np.zeros_like(depth)
        positive_friction[
            self.positive_friction_slice
        ] = positive_friction_parent.positive_friction(self, agg=False)

        # Determine negative friction
        negative_friction = np.zeros_like(depth)
        negative_friction[
            self.negative_friction_slice
        ] = negative_friction_parent.negative_friction(self, agg=False)

        elastic_elongation = settlement.pile.elastic_elongation(
            self.elastic_modulus_pile,
            self.area,
            self.pile_load / 1e3,
            depth,
            negative_friction,
            positive_friction,
        )

        # Settlement at pile tip level
        pile_settlement_ptl = self.pile_settlement_ptl()

        grain_pressure = original_grain_pressure + positive_friction - negative_friction
        soil_settlement = self.soil_settlement(grain_pressure)
        total_settlement_pile = np.cumsum(elastic_elongation) + pile_settlement_ptl
        return soil_settlement, total_settlement_pile

    def find_friction_tipping_point(self):
        # first iteration no rs is known, so we take the lower bound method.
        negative_friction_parent = PileCalculationLowerBound
        positive_friction_parent = PileCalculationLowerBound

        original_grain_pressure = self.merged_soil_properties.grain_pressure.values[
            : self._idx_ptl
        ]

        # slice until pile tip level
        depth = self.merged_soil_properties.depth[: self._idx_ptl]

        last_state = np.inf

        for i in range(10):
            soil_settlement, total_settlement_pile = self._single_iter(
                positive_friction_parent,
                negative_friction_parent,
                depth,
                original_grain_pressure,
            )

            if i == 0:
                negative_friction_parent = PileCalculation
                positive_friction_parent = PileCalculation

            signs = np.sign(total_settlement_pile - soil_settlement)
            idx = np.argwhere(np.diff(signs) > 0).flatten()
            if len(idx) > 0:
                tipping_idx = idx[0]
            else:
                tipping_idx = len(depth)
            self.positive_friction_slice = slice(tipping_idx, len(depth))
            self.negative_friction_slice = slice(0, tipping_idx)

            if last_state == tipping_idx:
                break
            last_state = tipping_idx
        return self._single_iter(
            positive_friction_parent,
            negative_friction_parent,
            depth,
            original_grain_pressure,
        )

    def run_calculation(self, pile_tip_level):
        """
        Run a calculation and set result attributes.
        Parameters
        ----------
        pile_tip_level : Union[np.array[float], float]
            Pile tip level in [m NAP]
        """
        self._init_calculation(pile_tip_level)
        for ptl in self.pile_tip_level_:
            self._set_ptl(ptl)
            rb, qc1, qc2, qc3 = self.pile_tip_resistance(agg=False)
            self.rb_.append(rb)
            self.qc1_.append(qc1)
            self.qc2_.append(qc2)
            self.qc3_.append(qc3)
            self.find_friction_tipping_point()
            self.nk_.append(self.nk)
            self.rs_.append(self.rs)
        self.rs_ = np.array(self.rs_)
        self.rb_ = np.array(self.rb_)
        self.nk_ = np.array(self.nk_)
