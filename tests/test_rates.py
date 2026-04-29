"""Tests for rates.py — bootstrapping, interpolation, and RateCurve queries.

The Treasury fetch itself is *not* tested here (it would require network
access and external data) — those tests would belong in a separate
integration suite.  Here we test the deterministic bits: BEY → continuous
conversion, recursive coupon-bond bootstrap, log-DF interpolation, and the
fallback / accessor helpers.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rates import (  # noqa: E402
    RateCurve,
    TENOR_LABELS,
    TENOR_YEARS,
    _format_tenor,
    _interp_log_df,
    bootstrap_zero_rates,
    build_curve_from_yields,
    build_curve_from_zero_rates,
    flat_curve,
    get_rate,
)


# ---------------------------------------------------------------------------
# BEY → continuous conversion (T-bill leg of the bootstrap)
# ---------------------------------------------------------------------------

class TestBEYConversion:
    """T-bill par yields are zero rates already; only the compounding
    convention changes from BEY (semi-annual) to continuous."""

    def test_3m_conversion_matches_formula(self):
        # 5 % BEY → r_cc = 2 · ln(1 + 0.025) ≈ 4.9385 %
        zeros = bootstrap_zero_rates({"3M": 5.0})
        T_3m = TENOR_YEARS["3M"]
        expected = 2.0 * np.log1p(0.025)
        assert abs(zeros[T_3m] - expected) < 1e-12

    def test_zero_yield_gives_zero_rate(self):
        zeros = bootstrap_zero_rates({"6M": 0.0})
        T_6m = TENOR_YEARS["6M"]
        assert abs(zeros[T_6m]) < 1e-12

    def test_bill_zero_rate_below_par_yield_for_positive_rates(self):
        """Continuous compounding produces a *lower* rate than semi-annual
        for the same growth — sanity check on the conversion direction."""
        bey = 4.5
        zeros = bootstrap_zero_rates({"1Y": bey})
        T_1y = TENOR_YEARS["1Y"]
        assert zeros[T_1y] < bey / 100.0
        # And by no more than ~6 bps for a 4.5 % rate.
        assert (bey / 100.0) - zeros[T_1y] < 0.001


# ---------------------------------------------------------------------------
# Coupon-bond bootstrap (T-note leg)
# ---------------------------------------------------------------------------

class TestCouponBootstrap:
    """The recursive bootstrap must price each par bond back to exactly 1.0
    when discounted using its own bootstrapped curve.  This is the
    self-consistency check that defines a correct bootstrap."""

    @staticmethod
    def _price_par_bond(T: float, y_par: float, zeros: dict[float, float]) -> float:
        """Price a semi-annual coupon bond at yield y_par using the curve."""
        coupon_times = np.arange(0.5, T + 1e-9, 0.5)
        Ts = np.array(sorted(zeros.keys()))
        rs = np.array([zeros[t] for t in Ts])
        pv = 0.0
        for t in coupon_times:
            r_t = _interp_log_df(t, Ts, rs)
            cf = (y_par / 2.0) + (1.0 if abs(t - T) < 1e-9 else 0.0)
            pv += cf * np.exp(-r_t * t)
        return pv

    def test_2y_par_bond_prices_to_par(self):
        yields = {"6M": 4.5, "1Y": 4.4, "2Y": 4.3}
        zeros = bootstrap_zero_rates(yields)
        pv = self._price_par_bond(2.0, yields["2Y"] / 100.0, zeros)
        assert abs(pv - 1.0) < 1e-9

    def test_5y_par_bond_prices_to_par(self):
        yields = {"6M": 4.5, "1Y": 4.4, "2Y": 4.3, "3Y": 4.25, "5Y": 4.2}
        zeros = bootstrap_zero_rates(yields)
        pv = self._price_par_bond(5.0, yields["5Y"] / 100.0, zeros)
        assert abs(pv - 1.0) < 1e-9

    def test_inverted_curve_prices_to_par(self):
        """Recent-history check: short end above long end."""
        yields = {"6M": 5.5, "1Y": 5.2, "2Y": 4.8, "3Y": 4.5, "5Y": 4.2}
        zeros = bootstrap_zero_rates(yields)
        for label, T in [("2Y", 2.0), ("3Y", 3.0), ("5Y", 5.0)]:
            pv = self._price_par_bond(T, yields[label] / 100.0, zeros)
            assert abs(pv - 1.0) < 1e-9, f"{label} not at par: {pv}"

    def test_zero_rate_close_to_par_yield_for_normal_curve(self):
        """For a smooth normal curve, zero rate ≈ par yield to within ~20 bps."""
        yields = {"6M": 4.5, "1Y": 4.4, "2Y": 4.3, "3Y": 4.25, "5Y": 4.2}
        zeros = bootstrap_zero_rates(yields)
        for label in ("2Y", "3Y", "5Y"):
            T = TENOR_YEARS[label]
            assert abs(zeros[T] - yields[label] / 100.0) < 0.002


# ---------------------------------------------------------------------------
# Log-DF interpolation
# ---------------------------------------------------------------------------

class TestLogDFInterp:
    tenors = (0.5, 1.0, 2.0, 5.0)
    rates = (0.04, 0.045, 0.043, 0.042)

    def test_returns_exact_rate_at_node(self):
        for T, r in zip(self.tenors, self.rates):
            assert abs(_interp_log_df(T, self.tenors, self.rates) - r) < 1e-12

    def test_below_min_returns_first_rate(self):
        assert _interp_log_df(0.001, self.tenors, self.rates) == self.rates[0]

    def test_above_max_returns_last_rate(self):
        assert _interp_log_df(10.0, self.tenors, self.rates) == self.rates[-1]

    def test_between_nodes_lies_in_log_DF_segment(self):
        """At T = 1.5 (midpoint of [1, 2]), log DF is the average of log DF(1)
        and log DF(2) — and the resulting zero rate sits between r(1) and r(2)."""
        T = 1.5
        r_q = _interp_log_df(T, self.tenors, self.rates)
        r1, r2 = self.rates[1], self.rates[2]
        T1, T2 = self.tenors[1], self.tenors[2]
        # Manual log-DF average:
        logDF_q = -r1 * T1 + 0.5 * (-r2 * T2 - (-r1 * T1))
        expected = -logDF_q / T
        assert abs(r_q - expected) < 1e-12
        # And the resulting rate is bracketed by r1 and r2.
        assert min(r1, r2) <= r_q <= max(r1, r2)

    def test_single_node_curve_is_flat(self):
        assert _interp_log_df(3.0, (1.0,), (0.05,)) == 0.05


# ---------------------------------------------------------------------------
# RateCurve container API
# ---------------------------------------------------------------------------

class TestRateCurve:
    @pytest.fixture
    def curve(self) -> RateCurve:
        yields = {"6M": 4.5, "1Y": 4.4, "2Y": 4.3, "5Y": 4.2}
        return build_curve_from_yields(yields, snap_date=date(2026, 4, 14))

    def test_zero_rate_at_node(self, curve):
        # 1Y is a T-bill so the zero rate is exactly 2·ln(1 + 0.022)
        T_1y = TENOR_YEARS["1Y"]
        expected = 2.0 * np.log1p(0.044 / 2.0)
        assert abs(curve.zero_rate(T_1y) - expected) < 1e-12

    def test_discount_factor_at_zero_is_one(self, curve):
        assert curve.discount_factor(0.0) == 1.0

    def test_discount_factor_decreasing_in_T(self, curve):
        Ts = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0]
        DFs = [curve.discount_factor(T) for T in Ts]
        for a, b in zip(DFs, DFs[1:]):
            assert b < a

    def test_extrapolates_flat_beyond_max(self, curve):
        r5 = curve.zero_rate(5.0)
        r6 = curve.zero_rate(6.0)
        r10 = curve.zero_rate(10.0)
        assert r5 == r6 == r10

    def test_source_and_snap_date_are_recorded(self, curve):
        assert curve.source == "treasury"
        assert curve.snap_date_iso == "2026-04-14"

    def test_as_table_has_expected_columns(self, curve):
        df = curve.as_table()
        assert list(df.columns) == ["Tenor", "Years", "Zero rate (%)"]
        # Zero rates in the table are expressed as percentages
        for r_decimal, r_pct in zip(curve.zero_rates, df["Zero rate (%)"]):
            assert abs(r_decimal * 100.0 - r_pct) < 1e-12


# ---------------------------------------------------------------------------
# Builders and helpers
# ---------------------------------------------------------------------------

class TestBuilders:
    def test_flat_curve_returns_constant_rate(self):
        c = flat_curve(0.045, snap_date=date(2026, 4, 14))
        for T in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            assert abs(c.zero_rate(T) - 0.045) < 1e-12
        assert c.source == "flat"

    def test_build_from_zero_rates_skips_BEY_conversion(self):
        """build_curve_from_zero_rates treats input as already continuous."""
        c = build_curve_from_zero_rates({"1Y": 4.4, "2Y": 4.3})
        # 1Y rate should be exactly 0.044 (no BEY conversion applied)
        assert abs(c.zero_rate(1.0) - 0.044) < 1e-12

    def test_build_from_zero_rates_preserves_only_known_tenors(self):
        """Unknown tenor labels are ignored; ordering is canonical."""
        c = build_curve_from_zero_rates({"1Y": 4.4, "FOO": 999.0})
        assert c.tenors_yr == (1.0,)
        assert abs(c.zero_rates[0] - 0.044) < 1e-12

    def test_build_from_yields_rejects_empty_input(self):
        with pytest.raises(ValueError):
            build_curve_from_yields({})


class TestGetRate:
    def test_get_rate_from_curve(self):
        c = flat_curve(0.05)
        assert abs(get_rate(c, 1.5) - 0.05) < 1e-12

    def test_get_rate_from_scalar(self):
        assert get_rate(0.04, 1.5) == 0.04

    def test_get_rate_accepts_int_zero(self):
        assert get_rate(0, 0.5) == 0.0


# ---------------------------------------------------------------------------
# Defensive / edge-case coverage flagged during the audit
# ---------------------------------------------------------------------------

class TestBootstrapCompleteness:
    """Every canonical tenor must appear in the curve.  Tenors not quoted by
    Treasury (notably 4Y, which CMT skips) are filled by log-DF
    interpolation from the surrounding bootstrapped knots — same scheme
    used at pricing time, so the row is consistent with its neighbours."""

    def test_full_curve_has_all_canonical_tenors(self):
        yields = {
            "1M": 5.50, "2M": 5.45, "3M": 5.40, "4M": 5.30, "6M": 5.20,
            "1Y": 5.00, "2Y": 4.50, "3Y": 4.30, "5Y": 4.20,
        }
        c = build_curve_from_yields(yields)
        from rates import TENOR_LABELS, TENOR_YEARS
        assert len(c.tenors_yr) == len(TENOR_LABELS)
        assert c.tenors_yr == tuple(sorted(c.tenors_yr))  # strictly increasing
        # Every canonical tenor must be represented, even those the CMT
        # schedule omits (4Y).
        for lbl in TENOR_LABELS:
            assert TENOR_YEARS[lbl] in c.tenors_yr

    def test_4y_filled_by_interpolation_when_treasury_omits_it(self):
        """Treasury CMT publishes 3Y and 5Y but not 4Y.  The curve must
        still expose a 4Y knot, with a value bracketed by 3Y and 5Y under
        a normally-shaped curve."""
        yields = {
            "1M": 5.50, "3M": 5.40, "6M": 5.20,
            "1Y": 5.00, "2Y": 4.50, "3Y": 4.30, "5Y": 4.20,
        }
        c = build_curve_from_yields(yields)
        i_3 = c.tenors_yr.index(3.0)
        i_4 = c.tenors_yr.index(4.0)
        i_5 = c.tenors_yr.index(5.0)
        # On a normal (downward-sloping at the long end) curve the
        # interpolated 4Y zero must sit strictly between 3Y and 5Y.
        assert min(c.zero_rates[i_3], c.zero_rates[i_5]) <= c.zero_rates[i_4]
        assert c.zero_rates[i_4] <= max(c.zero_rates[i_3], c.zero_rates[i_5])

    def test_partial_input_extrapolates_to_full_canonical_grid(self):
        """If the input is sparse (e.g. missing the front end), the
        bootstrap fills the canonical grid by interpolation / flat
        extrapolation rather than producing a partial curve."""
        yields = {"6M": 4.5, "1Y": 4.4, "5Y": 4.2}
        c = build_curve_from_yields(yields)
        from rates import TENOR_LABELS, TENOR_YEARS
        assert len(c.tenors_yr) == len(TENOR_LABELS)
        for lbl in TENOR_LABELS:
            assert TENOR_YEARS[lbl] in c.tenors_yr


class TestBootstrapNegativeRates:
    """Mildly negative rates (think Eurozone 2016-2021) should bootstrap fine."""

    def test_slightly_negative_curve_self_consistent(self):
        yields = {"6M": -0.10, "1Y": -0.05, "2Y": 0.05, "5Y": 0.50}
        zeros = bootstrap_zero_rates(yields)
        # 2y par bond should reprice to par under its own bootstrap.
        T = 2.0
        coupon_times = np.arange(0.5, T + 1e-9, 0.5)
        Ts = np.array(sorted(zeros.keys()))
        rs = np.array([zeros[t] for t in Ts])
        pv = 0.0
        for t in coupon_times:
            r_t = _interp_log_df(t, Ts, rs)
            cf = (yields["2Y"] / 100.0 / 2.0) + (1.0 if abs(t - T) < 1e-9 else 0.0)
            pv += cf * np.exp(-r_t * t)
        assert abs(pv - 1.0) < 1e-9

    def test_strongly_negative_curve_bracket_holds(self):
        """Regression for a real bug: the original bracket
            lo = max(r_centre − 0.05, −0.05),  hi = r_centre + 0.05
        produced lo > hi when r_centre dropped below ~−5 %, making brentq
        raise ValueError.  A −8 % par bond reprices the 2y point at
        ≈ −8.03 % and should bootstrap without crashing."""
        zeros = bootstrap_zero_rates({"6M": -7.5, "1Y": -8.0, "2Y": -8.0})
        T = 2.0
        coupon_times = np.arange(0.5, T + 1e-9, 0.5)
        Ts = np.array(sorted(zeros.keys()))
        rs = np.array([zeros[t] for t in Ts])
        pv = 0.0
        for t in coupon_times:
            r_t = _interp_log_df(t, Ts, rs)
            cf = (-0.08 / 2.0) + (1.0 if abs(t - T) < 1e-9 else 0.0)
            pv += cf * np.exp(-r_t * t)
        assert abs(pv - 1.0) < 1e-9


class TestDiscountFactorInterp:
    """Log-DF interpolation has the property: DF(midpoint) = √(DF(T0)·DF(T1))."""

    def test_geometric_mean_property(self):
        c = build_curve_from_zero_rates({"1Y": 4.0, "2Y": 5.0})
        df1 = c.discount_factor(1.0)
        df2 = c.discount_factor(2.0)
        df_mid = c.discount_factor(1.5)
        assert abs(df_mid - np.sqrt(df1 * df2)) < 1e-12


class TestRateCurveHashableForCache:
    """The Streamlit `_run_checks` cache reconstructs RateCurve from tuples;
    the curve must round-trip through the (tenors, rates) tuple representation
    without loss."""

    def test_roundtrip_through_tuples(self):
        c1 = build_curve_from_yields(
            {"6M": 4.5, "1Y": 4.4, "2Y": 4.3, "5Y": 4.2},
            snap_date=date(2026, 4, 14),
        )
        # Reconstruct via the same path the cache uses
        c2 = RateCurve(
            tenors_yr=tuple(c1.tenors_yr),
            zero_rates=tuple(c1.zero_rates),
            source="cached",
            snap_date_iso=c1.snap_date_iso,
        )
        for T in (0.25, 0.5, 1.0, 1.5, 2.0, 3.5, 5.0):
            assert abs(c1.zero_rate(T) - c2.zero_rate(T)) < 1e-15


class TestTenorLabelRoundTrip:
    """as_table must produce labels that round-trip through
    build_curve_from_zero_rates, otherwise user edits on rows whose tenor
    is not in TENOR_LABELS are silently dropped (regression: 1M, 2M, 4M
    were displayed as '0.0833Y', '0.1667Y', '0.3333Y' due to float-equality
    miss in the label lookup)."""

    def test_format_tenor_canonical_matches(self):
        """Each canonical TENORS entry must map back to its label, even for
        tenors like 1/12 that are not exactly representable in float."""
        for label, T in TENORS_CANONICAL():
            assert _format_tenor(T) == label, (
                f"Tenor {T!r} ({label}) did not round-trip — got "
                f"{_format_tenor(T)!r}"
            )

    def test_as_table_uses_only_canonical_labels(self):
        """For a curve built from the full canonical tenor set, every
        Tenor cell in the table must be a key recognised by
        build_curve_from_zero_rates."""
        yields = {
            "1M": 5.50, "2M": 5.45, "3M": 5.40, "4M": 5.30, "6M": 5.20,
            "1Y": 5.00, "2Y": 4.50, "3Y": 4.30, "5Y": 4.20,
        }
        c = build_curve_from_yields(yields)
        table = c.as_table()
        for label in table["Tenor"]:
            assert label in TENOR_LABELS, (
                f"Non-canonical tenor label {label!r} would be dropped "
                f"on reconstruction"
            )

    def test_full_round_trip_table_to_curve(self):
        """Build curve → as_table → reconstruct via build_curve_from_zero_rates
        must preserve every zero rate to machine precision."""
        yields = {
            "1M": 5.50, "2M": 5.45, "3M": 5.40, "4M": 5.30, "6M": 5.20,
            "1Y": 5.00, "2Y": 4.50, "3Y": 4.30, "5Y": 4.20,
        }
        c1 = build_curve_from_yields(yields)
        table = c1.as_table()
        rates_dict = {
            row["Tenor"]: row["Zero rate (%)"]
            for _, row in table.iterrows()
        }
        c2 = build_curve_from_zero_rates(rates_dict)
        assert c1.tenors_yr == c2.tenors_yr, (
            "Tenor set lost in round-trip"
        )
        for r1, r2 in zip(c1.zero_rates, c2.zero_rates):
            # Display rounds to 3 decimal places (0.001 % = 1 bp), so
            # round-trip precision is at most that.
            assert abs(r1 - r2) < 1e-12

    def test_format_tenor_fallback_units(self):
        """Non-canonical tenors should produce readable unit labels."""
        # 5 days
        assert _format_tenor(5 / 365.25) == "5D"
        # 3 weeks (21 days, between 14 and 60 — uses weeks)
        assert _format_tenor(21 / 365.25) == "3W"
        # 7 months
        assert _format_tenor(7 / 12) == "7M"
        # 4 years
        assert _format_tenor(4.0) == "4Y"


def TENORS_CANONICAL():
    """Helper — exposes TENORS via a callable so the test class can iterate."""
    from rates import TENORS
    return TENORS