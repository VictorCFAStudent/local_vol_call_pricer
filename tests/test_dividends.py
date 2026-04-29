"""Tests for dividends.py — implied dividend yield extraction & curve API."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dividends import (  # noqa: E402
    DivCurve,
    _interp_qT,
    build_curve_from_table,
    extract_implied,
    extract_implied_at_canonical_tenors,
    flat_div_curve,
    get_div_yield,
)
from rates import TENOR_LABELS, TENOR_YEARS, flat_curve  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_vol_df(spot: float, T_arr, q_arr, r_arr) -> pd.DataFrame:
    """Build a minimal vol_df with a forward consistent with given r and q.

    F(T) = S · exp((r − q) · T)
    """
    rows = []
    for T, q, r in zip(T_arr, q_arr, r_arr):
        F = spot * np.exp((r - q) * T)
        # Tag a couple of strikes per expiry for realism
        for m_pct in (90.0, 100.0, 110.0):
            rows.append({
                "expiry_label": f"T={T:.4f}y",
                "time_to_expiry": float(T),
                "moneyness_pct": float(m_pct),
                "forward_price": float(F),
                "implied_vol": 0.20,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parity-based extraction
# ---------------------------------------------------------------------------

class TestImpliedExtraction:

    def test_recovers_q_when_F_constructed_from_known_q(self):
        """If F is built from a known q under a known r, extract_implied
        must recover q to machine precision."""
        spot = 5000.0
        Ts = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
        qs_true = np.array([0.018, 0.019, 0.020, 0.021, 0.022])
        # Use a flat 4 % rate so the rate-curve lookup is trivial.
        rs = np.full_like(Ts, 0.04)
        df = _make_vol_df(spot, Ts, qs_true, rs)

        rate_curve = flat_curve(0.04)
        div_curve = extract_implied(df, spot, rate_curve, max_T=10.0)

        assert div_curve.tenors_yr == tuple(Ts)
        for got, want in zip(div_curve.div_yields, qs_true):
            assert abs(got - want) < 1e-12

    def test_consistent_with_rate_curve(self):
        """Extracting q with rate curve A vs B differs by exactly r_A − r_B
        at each T (the parity-extraction absorbs the rate-curve choice)."""
        spot = 100.0
        Ts = np.array([1.0, 2.0])
        qs_true = np.array([0.02, 0.025])
        rs = np.full_like(Ts, 0.05)
        df = _make_vol_df(spot, Ts, qs_true, rs)

        c_A = extract_implied(df, spot, flat_curve(0.05), max_T=10.0)
        c_B = extract_implied(df, spot, flat_curve(0.06), max_T=10.0)
        # 1 % higher r ⇒ extracted q must be 1 % higher (to keep F invariant)
        for q_a, q_b in zip(c_A.div_yields, c_B.div_yields):
            assert abs((q_b - q_a) - 0.01) < 1e-12

    def test_filters_by_max_T(self):
        spot = 100.0
        Ts = np.array([0.5, 2.0, 4.0, 6.0, 8.0])
        qs = np.full_like(Ts, 0.02)
        rs = np.full_like(Ts, 0.04)
        df = _make_vol_df(spot, Ts, qs, rs)

        c = extract_implied(df, spot, flat_curve(0.04), max_T=5.0)
        # Only T <= 5 should be kept
        assert len(c.tenors_yr) == 3
        assert max(c.tenors_yr) <= 5.0

    def test_raises_when_no_expiry_in_window(self):
        df = _make_vol_df(100.0, [10.0], [0.02], [0.04])
        with pytest.raises(ValueError):
            extract_implied(df, 100.0, flat_curve(0.04), max_T=5.0)

    def test_accepts_scalar_rate(self):
        """A bare float for `rate_curve` should be treated as a flat rate."""
        spot = 100.0
        Ts = np.array([1.0])
        qs = np.array([0.02])
        rs = np.array([0.04])
        df = _make_vol_df(spot, Ts, qs, rs)
        c = extract_implied(df, spot, 0.04, max_T=10.0)
        assert abs(c.div_yields[0] - 0.02) < 1e-12


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

class TestQTInterp:
    """`q · T` linear interpolation between knots."""

    tenors = (0.5, 1.0, 2.0, 5.0)
    yields = (0.020, 0.022, 0.024, 0.026)

    def test_returns_exact_yield_at_node(self):
        for T, q in zip(self.tenors, self.yields):
            assert abs(_interp_qT(T, self.tenors, self.yields) - q) < 1e-12

    def test_below_min_returns_first(self):
        assert _interp_qT(0.1, self.tenors, self.yields) == self.yields[0]

    def test_above_max_returns_last(self):
        assert _interp_qT(10.0, self.tenors, self.yields) == self.yields[-1]

    def test_qT_linear_property(self):
        """q(T)·T must be linear in T between knots."""
        T0, T1 = self.tenors[1], self.tenors[2]
        q0, q1 = self.yields[1], self.yields[2]
        T_mid = (T0 + T1) / 2
        q_mid = _interp_qT(T_mid, self.tenors, self.yields)
        # q_mid · T_mid should be the average of the endpoint q·T values
        assert abs(q_mid * T_mid - 0.5 * (q0 * T0 + q1 * T1)) < 1e-12

    def test_single_node_curve_is_flat(self):
        assert _interp_qT(3.0, (1.0,), (0.025,)) == 0.025

    def test_empty_curve_raises(self):
        with pytest.raises(ValueError):
            _interp_qT(1.0, (), ())


# ---------------------------------------------------------------------------
# Curve API + builders
# ---------------------------------------------------------------------------

class TestDivCurveAPI:

    @pytest.fixture
    def curve(self):
        return DivCurve(
            tenors_yr=(0.5, 1.0, 2.0, 5.0),
            div_yields=(0.018, 0.020, 0.022, 0.025),
            labels=("6M", "1Y", "2Y", "5Y"),
            source="implied",
            snap_date_iso="2026-04-14",
        )

    def test_div_yield_at_node(self, curve):
        assert abs(curve.div_yield(1.0) - 0.020) < 1e-12

    def test_extrapolates_flat_beyond(self, curve):
        assert curve.div_yield(0.1) == 0.018  # flat from first
        assert curve.div_yield(10.0) == 0.025  # flat from last

    def test_as_table_columns_and_pct(self, curve):
        df = curve.as_table()
        assert list(df.columns) == ["Tenor", "Years", "q (%)"]
        for q_decimal, q_pct in zip(curve.div_yields, df["q (%)"]):
            assert abs(q_decimal * 100 - q_pct) < 1e-12

    def test_round_trip_via_table(self, curve):
        """Build → as_table → build_curve_from_table preserves rates."""
        df = curve.as_table()
        c2 = build_curve_from_table(df)
        assert c2.tenors_yr == curve.tenors_yr
        for q1, q2 in zip(curve.div_yields, c2.div_yields):
            assert abs(q1 - q2) < 1e-12


class TestBuilders:

    def test_flat_curve_constant_yield(self):
        c = flat_div_curve(0.025)
        for T in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            assert abs(c.div_yield(T) - 0.025) < 1e-12
        assert c.source == "flat"

    def test_build_from_table_drops_nan(self):
        df = pd.DataFrame({
            "Tenor": ["A", "B", "C"],
            "Years":  [0.5, 1.0, 2.0],
            "q (%)":  [2.0, np.nan, 2.5],
        })
        c = build_curve_from_table(df)
        assert len(c.tenors_yr) == 2
        assert c.labels == ("A", "C")

    def test_build_from_empty_table_raises(self):
        df = pd.DataFrame({
            "Tenor": [], "Years": [], "q (%)": [],
        })
        with pytest.raises(ValueError):
            build_curve_from_table(df)


class TestGetDivYield:

    def test_from_curve(self):
        c = flat_div_curve(0.02)
        assert abs(get_div_yield(c, 1.5) - 0.02) < 1e-12

    def test_from_scalar(self):
        assert get_div_yield(0.018, 1.5) == 0.018

    def test_from_zero_scalar(self):
        assert get_div_yield(0, 0.5) == 0.0


# ---------------------------------------------------------------------------
# End-to-end consistency: extract → use to reproduce forward
# ---------------------------------------------------------------------------

class TestForwardSelfConsistency:

    def test_reproducing_F_from_extracted_q(self):
        """Plugging extracted q back into F = S·exp((r − q)·T) must reproduce
        the original Bloomberg forward exactly."""
        spot = 6500.0
        Ts = np.array([0.25, 0.5, 1.0, 2.5, 5.0])
        qs_true = np.array([0.0150, 0.0175, 0.0190, 0.0210, 0.0220])
        rs = np.array([0.045, 0.046, 0.047, 0.048, 0.049])
        df = _make_vol_df(spot, Ts, qs_true, rs)

        # Use an explicit curve with the matching r(T) for each expiry
        from rates import RateCurve
        r_curve = RateCurve(
            tenors_yr=tuple(Ts), zero_rates=tuple(rs),
            source="manual", snap_date_iso=None,
        )
        div_curve = extract_implied(df, spot, r_curve, max_T=10.0)

        # Reconstruct F from extracted q + same r
        for T, q in zip(div_curve.tenors_yr, div_curve.div_yields):
            r_T = r_curve.zero_rate(T)
            F_reconstructed = spot * np.exp((r_T - q) * T)
            F_market = df[df["time_to_expiry"] == T]["forward_price"].iloc[0]
            assert abs(F_reconstructed - F_market) < 1e-9


# ---------------------------------------------------------------------------
# Canonical-tenor extraction — what the sidebar actually shows
# ---------------------------------------------------------------------------

class TestCanonicalTenorExtraction:
    """`extract_implied_at_canonical_tenors` fits a straight line
    `q(T) = a + b·T` to all per-expiry implied-q points (via the OLS
    parametrisation `q·T = a·T + b·T²`) and evaluates that line at the
    canonical tenor grid.  This is what powers the sidebar table — it
    gives a clean, non-inverted term structure without overfitting
    short-T parity noise."""

    def test_labels_and_tenors_match_canonical_grid(self):
        """The returned curve has the canonical tenor labels and years,
        in the exact order of `rates.TENOR_LABELS`."""
        spot = 5000.0
        Ts = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 5.0])
        qs = np.array([0.012, 0.015, 0.018, 0.020, 0.022, 0.025])
        rs = np.full_like(Ts, 0.045)
        df = _make_vol_df(spot, Ts, qs, rs)

        c = extract_implied_at_canonical_tenors(df, spot, flat_curve(0.045))
        assert c.labels == TENOR_LABELS
        for lbl, T in zip(c.labels, c.tenors_yr):
            assert abs(T - TENOR_YEARS[lbl]) < 1e-12

    def test_recovers_constant_q(self):
        """If every per-expiry q is the same, the OLS fit must recover
        that constant exactly: `a = q_true`, `b = 0`, and every canonical
        tenor reads `q_true`."""
        spot = 5000.0
        Ts = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 5.0])
        q_true = 0.018
        qs = np.full_like(Ts, q_true)
        rs = np.full_like(Ts, 0.045)
        df = _make_vol_df(spot, Ts, qs, rs)

        c = extract_implied_at_canonical_tenors(df, spot, flat_curve(0.045))
        for q in c.div_yields:
            assert abs(q - q_true) < 1e-9

    def test_recovers_linear_q(self):
        """If per-expiry q lies exactly on a line `a + b·T`, the fit
        must recover (a, b) and reproduce that line at canonical tenors
        to machine precision."""
        spot = 5000.0
        a_true, b_true = 0.015, 0.002  # q rises from 1.5% by 0.2%/year
        Ts = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        qs = a_true + b_true * Ts
        rs = np.full_like(Ts, 0.045)
        df = _make_vol_df(spot, Ts, qs, rs)

        c = extract_implied_at_canonical_tenors(df, spot, flat_curve(0.045))
        for lbl, T, q in zip(c.labels, c.tenors_yr, c.div_yields):
            assert abs(q - (a_true + b_true * T)) < 1e-9

    def test_smooths_short_T_noise_into_monotone_curve(self):
        """When the underlying per-expiry q is bumpy at the front end
        (the realistic case from quote rounding + dividend timing), the
        canonical curve must come out *monotone* in T — the fit washes
        out the bumps."""
        spot = 5000.0
        # Underlying q is 1.5% at every long-T expiry but the front end
        # is corrupted with ±0.5 % zigzag noise (quote artefacts).
        Ts = np.array([
            0.02, 0.04, 0.06, 0.08, 0.10,
            0.25, 0.50, 1.0, 2.0, 3.0, 5.0,
        ])
        rng = np.random.default_rng(42)
        front_noise = np.array([+0.005, -0.005, +0.004, -0.003, +0.002])
        qs = np.concatenate([
            0.015 + front_noise,
            np.full(6, 0.015),
        ])
        rs = np.full_like(Ts, 0.045)
        df = _make_vol_df(spot, Ts, qs, rs)

        c = extract_implied_at_canonical_tenors(df, spot, flat_curve(0.045))
        # Term structure must be monotone (well-behaved fit), not bumpy
        diffs = np.diff(c.div_yields)
        assert np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12)
        # And every value should be within ~50 bp of the underlying 1.5%
        for q in c.div_yields:
            assert abs(q - 0.015) < 0.005

    def test_round_trip_through_sidebar_table(self):
        """Canonical-tenor curve → as_table → build_curve_from_table
        preserves the curve to within table-rounding precision."""
        spot = 5000.0
        Ts = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 5.0])
        qs = np.array([0.012, 0.015, 0.018, 0.020, 0.022, 0.025])
        rs = np.full_like(Ts, 0.045)
        df = _make_vol_df(spot, Ts, qs, rs)

        c = extract_implied_at_canonical_tenors(df, spot, flat_curve(0.045))
        c2 = build_curve_from_table(c.as_table())
        assert c2.labels == c.labels
        assert c2.tenors_yr == c.tenors_yr
        for q1, q2 in zip(c.div_yields, c2.div_yields):
            assert abs(q1 - q2) < 1e-5  # 3-decimal % rounding


# ---------------------------------------------------------------------------
# Round-trip through the sidebar table — guards the streamlit user-edit path
# ---------------------------------------------------------------------------

class TestSidebarRoundTrip:
    """Mirror of `test_full_round_trip_table_to_curve` (rates) — the editable
    dividend table must reconstruct the same curve when no edits are made."""

    def test_extract_then_table_then_build_preserves_q(self):
        spot = 5000.0
        Ts = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 5.0])
        qs = np.array([0.012, 0.015, 0.018, 0.020, 0.022, 0.025])
        rs = np.full_like(Ts, 0.045)
        df = _make_vol_df(spot, Ts, qs, rs)

        curve = extract_implied(df, spot, flat_curve(0.045), max_T=10.0)
        round_tripped = build_curve_from_table(curve.as_table())

        assert round_tripped.tenors_yr == curve.tenors_yr
        assert round_tripped.labels == curve.labels
        for q1, q2 in zip(curve.div_yields, round_tripped.div_yields):
            # 3-decimal % rounding in the table caps precision at 1e-5
            assert abs(q1 - q2) < 1e-5


# ---------------------------------------------------------------------------
# Heston calibration — documents that q is INERT under moneyness-quoted IVs
# ---------------------------------------------------------------------------

class TestHestonCalibrationQInvariance:
    """Heston/Black-76 prices and vegas are positively homogeneous of
    degree 1 in (F, K).  Bloomberg quotes IVs at fixed moneyness, so when
    we change q (or r), F shifts by `exp(-Δq·T)` and every K shifts by
    the same factor — both `C_market` and `C_model` rescale identically,
    vega rescales identically, and the vega-weighted residual is invariant.
    The DE optimum is therefore independent of `q`.  This is why we do
    *not* plumb the dividend curve into `heston.calibrate`: it would be a
    no-op masquerading as a feature.  This test guards against a future
    refactor that breaks the invariance silently."""

    def test_calibration_invariant_under_q(self):
        from heston import calibrate as heston_calibrate

        spot = 5000.0
        # Mild smile across 3 expiries — non-trivial enough that any real
        # change in the calibration objective would shift the params.
        rows = []
        for T in (0.25, 0.5, 1.0):
            F = spot * np.exp((0.045 - 0.020) * T)
            for m_pct, iv in [
                (80.0, 0.30), (90.0, 0.24), (95.0, 0.21),
                (100.0, 0.20), (105.0, 0.21), (110.0, 0.23), (120.0, 0.30),
            ]:
                rows.append({
                    "expiry_label": f"T={T:.2f}",
                    "time_to_expiry": float(T),
                    "moneyness_pct": float(m_pct),
                    "forward_price": float(F),
                    "implied_vol": float(iv),
                })
        df = pd.DataFrame(rows)
        r_curve = flat_curve(0.045)

        res_a = heston_calibrate(
            df, spot, r_curve, 0.000,
            max_slices=3, min_T=0.01, m_range=(70.0, 130.0),
        )
        res_b = heston_calibrate(
            df, spot, r_curve, 0.050,
            max_slices=3, min_T=0.01, m_range=(70.0, 130.0),
        )
        for p_a, p_b in zip(res_a.params, res_b.params):
            assert abs(p_a - p_b) < 1e-10