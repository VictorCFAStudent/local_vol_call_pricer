"""
dividends.py — implied dividend yield curve from Bloomberg forwards.

Theory
------
For an equity index, the no-arbitrage forward at time T satisfies

    F(T) = S · exp((r(T) − q(T)) · T)

Solving for the dividend yield gives

    q(T) = r(T) − ln(F(T) / S) / T

Every expiry in the loaded Bloomberg snapshot already provides `(T, F(T))`,
so combined with the bootstrapped rate curve the dividend curve is fully
determined by the data we already have — no extra data source needed.

This is internally consistent with the option market we're pricing off:
plugging our `(r, q)` back into Heston / Black-76 reproduces the Bloomberg
forward exactly, by construction.  If Bloomberg used a slightly different
rate curve than ours, the difference is absorbed entirely into the
extracted `q(T)`, and the model still prices vanillas correctly.

Caveat: SPX pays discrete dividends, not a continuous yield.  The curve
here is the *continuous-yield equivalent* that reproduces the forward —
exact for European pricing, a good approximation for path-dependent
products that don't reference individual ex-dividend dates.

Interpolation
-------------
Between knots we interpolate linearly in `q · T` (total dividend yield
over time), which corresponds to piecewise-flat instantaneous dividend
rates between knot points.  Beyond the curve range we extrapolate flat.
This is the dividend equivalent of the log-DF / piecewise-flat-forwards
scheme used for the rate curve.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd

# Canonical display tenors for the dividend curve.  Reused from `rates` so
# the rate and dividend sidebar tables share the exact same tenor grid —
# anything else would be visually inconsistent for no analytic gain.
from rates import TENOR_LABELS, TENOR_YEARS


# ---------------------------------------------------------------------------
# Curve container
# ---------------------------------------------------------------------------

class DivCurve(NamedTuple):
    """Implied (continuously-compounded) dividend yield curve.

    Attributes
    ----------
    tenors_yr : tuple of T values (years), strictly increasing.
    div_yields : tuple of continuously-compounded dividend yields aligned
                 with tenors_yr.
    labels : tuple of row identifier strings aligned with tenors_yr.
             For canonical-tenor curves these are tenor strings ("1M",
             "2M", …) matching `rates.TENOR_LABELS`; for raw per-expiry
             extraction they are expiry strings ("16 Jun 2026", …).
    source : provenance — 'implied' (Bloomberg forwards × rate curve),
             'manual' (user override), or 'flat' (degenerate fallback).
    snap_date_iso : ISO-format snap-date string, or None.
    """
    tenors_yr: tuple[float, ...]
    div_yields: tuple[float, ...]
    labels: tuple[str, ...]
    source: str
    snap_date_iso: str | None

    def div_yield(self, T: float) -> float:
        """Continuously-compounded dividend yield at maturity T (years).

        Linear in `q·T` between curve knots (piecewise-flat instantaneous
        dividend rates); flat extrapolation beyond the curve range.
        """
        return float(_interp_qT(T, self.tenors_yr, self.div_yields))

    def as_table(self) -> pd.DataFrame:
        """Return the curve as a tidy DataFrame (for display in the UI)."""
        return pd.DataFrame({
            "Tenor": list(self.labels),
            "Years": [float(T) for T in self.tenors_yr],
            "q (%)": [float(q) * 100.0 for q in self.div_yields],
        })


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _interp_qT(
    T_query: float,
    tenors_yr: tuple[float, ...] | list[float] | np.ndarray,
    div_yields: tuple[float, ...] | list[float] | np.ndarray,
) -> float:
    """Linear interpolation in `q·T` space (piecewise-flat instantaneous div)."""
    tenors = np.asarray(tenors_yr, dtype=float)
    yields = np.asarray(div_yields, dtype=float)
    if tenors.size == 0:
        raise ValueError("Cannot interpolate — empty dividend curve.")
    if tenors.size == 1:
        return float(yields[0])

    # Flat extrapolation at the boundaries.
    if T_query <= tenors[0]:
        return float(yields[0])
    if T_query >= tenors[-1]:
        return float(yields[-1])

    idx = int(np.searchsorted(tenors, T_query, side="right") - 1)
    T0, T1 = tenors[idx], tenors[idx + 1]
    q0, q1 = yields[idx], yields[idx + 1]

    qT0, qT1 = q0 * T0, q1 * T1
    qTq = qT0 + (T_query - T0) / (T1 - T0) * (qT1 - qT0)
    return float(qTq / T_query)


# ---------------------------------------------------------------------------
# Implied-q extraction
# ---------------------------------------------------------------------------

def extract_implied(
    vol_df: pd.DataFrame,
    spot: float,
    rate_curve,
    snap_date: date | None = None,
    max_T: float = 5.0,
) -> DivCurve:
    """Extract the implied dividend yield curve from Bloomberg forwards.

    For each unique expiry with `T ≤ max_T`:
        q(T) = r(T) − ln(F(T) / S) / T

    `rate_curve` may be a `RateCurve` instance or a plain float (treated
    as a flat rate — useful for tests).
    """
    by_exp = (
        vol_df.drop_duplicates("expiry_label")
        .sort_values("time_to_expiry")
        .reset_index(drop=True)
    )
    by_exp = by_exp[by_exp["time_to_expiry"] <= max_T].reset_index(drop=True)
    if by_exp.empty:
        raise ValueError(
            f"No expiries with T ≤ {max_T} years available — "
            f"cannot build an implied dividend curve."
        )

    Ts = by_exp["time_to_expiry"].values.astype(float)
    Fs = by_exp["forward_price"].values.astype(float)
    labels = by_exp["expiry_label"].values

    # Pull r(T) for each expiry.  The rate_curve duck-types: a RateCurve
    # has .zero_rate, a scalar is treated as flat.
    if hasattr(rate_curve, "zero_rate"):
        rs = np.array([rate_curve.zero_rate(float(T)) for T in Ts])
    else:
        rs = np.full_like(Ts, float(rate_curve))

    # q(T) = r(T) − ln(F/S) / T
    qs = rs - np.log(Fs / float(spot)) / Ts

    return DivCurve(
        tenors_yr=tuple(float(T) for T in Ts),
        div_yields=tuple(float(q) for q in qs),
        labels=tuple(str(l) for l in labels),
        source="implied",
        snap_date_iso=snap_date.isoformat() if snap_date else None,
    )


_FRONT_END_MIN_T_FOR_FIT: float = 1.0 / 52.0  # drop expiries shorter than ~1 week


def extract_implied_at_canonical_tenors(
    vol_df: pd.DataFrame,
    spot: float,
    rate_curve,
    snap_date: date | None = None,
    tenor_labels: tuple[str, ...] = TENOR_LABELS,
    tenor_years: dict[str, float] = TENOR_YEARS,
    min_T: float = _FRONT_END_MIN_T_FOR_FIT,
    max_T: float = 5.0,
) -> DivCurve:
    """Build a dividend curve at canonical tenors (1M, 2M, … 5Y).

    Why a fit and not a point-sample
    --------------------------------
    Per-expiry parity-implied `q = r(T) − ln(F(T)/S)/T` is *bumpy* in the
    front end and oscillates by tens of bps between adjacent expiries
    further out, because:

    * Bloomberg often quotes `F(T_earliest) ≡ S` to the cent, so the
      earliest one or two rows pin `q ≈ r` (a quote artefact).
    * The S&P pays dividends in discrete chunks; whether an ex-div date
      falls inside a given expiry window jolts the corresponding `F`
      and swings the implied `q` by ~30 bp at 1–3 month tenors.

    Point-sampling that noisy curve at 1M / 2M / … re-injects the noise
    into the displayed table and produces visually inconsistent term
    structures (often non-monotone, sometimes inverted).  Instead we fit
    a single straight line `q(T) = a + b·T` in the linearised form

        ln(F/S) = (r − q(T))·T   ⇒   q·T = r·T − ln(F/S) = a·T + b·T²

    by ordinary least squares on all expiries with `min_T ≤ T ≤ max_T`,
    weighting implicitly by `T` (the natural OLS weight in this
    parametrisation).  We then evaluate the line `q(T) = a + b·T` at each
    canonical tenor — including any below `min_T`, which the OLS line is
    well-defined at; in practice the smallest canonical tenor (1M ≈ 0.083 yr)
    is already above the 1/52 ≈ 0.019 yr filter, so this never produces a
    visible extrapolation.  The fit is dominated by long-end expiries where
    the dividend-timing noise has averaged out, so the resulting term
    structure is smooth and reflects the *secular* level of the implied
    dividend yield — exactly what a user expects to see for SPX.

    Caveat — rate-basis level
    -------------------------
    Bloomberg builds `F` from a SOFR / futures-implied rate, not from
    Treasury CMT.  Since `q_extracted = r_us_treasury − ln(F/S)/T`, the
    extracted level is shifted *down* by `(r_BB − r_treasury)`, which
    grows with `T` (~10 bp at 1M, ~50 bp at 5Y).  The result reproduces
    Bloomberg's forward exactly under the rate curve we *do* have, which
    is what matters for self-consistent option pricing — it just won't
    match Bloomberg's published "indicative dividend yield" by the
    SOFR-Treasury basis.
    """
    by_exp = (
        vol_df.drop_duplicates("expiry_label")
        .sort_values("time_to_expiry")
        .reset_index(drop=True)
    )
    by_exp = by_exp[
        (by_exp["time_to_expiry"] >= min_T)
        & (by_exp["time_to_expiry"] <= max_T)
    ].reset_index(drop=True)
    if by_exp.empty:
        raise ValueError(
            f"No expiries with {min_T} ≤ T ≤ {max_T} years available — "
            f"cannot fit a dividend curve."
        )

    Ts_data = by_exp["time_to_expiry"].values.astype(float)
    Fs_data = by_exp["forward_price"].values.astype(float)
    if hasattr(rate_curve, "zero_rate"):
        rs_data = np.array([rate_curve.zero_rate(float(T)) for T in Ts_data])
    else:
        rs_data = np.full_like(Ts_data, float(rate_curve))

    # Linearised parametrisation: q·T = a·T + b·T²,  q(T) = a + b·T.
    # Solving by OLS in (a, b) on the design matrix [T, T²] is equivalent
    # to T-weighted regression of q on T, which down-weights the noisy
    # short-T points naturally.
    qT = rs_data * Ts_data - np.log(Fs_data / float(spot))
    A = np.column_stack([Ts_data, Ts_data ** 2])
    if Ts_data.size >= 2:
        coef, *_ = np.linalg.lstsq(A, qT, rcond=None)
        a, b = float(coef[0]), float(coef[1])
    else:
        # Single point — degenerate; fall back to a flat fit at that q
        a, b = float(qT[0] / Ts_data[0]), 0.0

    Ts_canonical = tuple(float(tenor_years[lbl]) for lbl in tenor_labels)
    qs_canonical = tuple(a + b * T for T in Ts_canonical)
    return DivCurve(
        tenors_yr=Ts_canonical,
        div_yields=qs_canonical,
        labels=tuple(tenor_labels),
        source="implied",
        snap_date_iso=snap_date.isoformat() if snap_date else None,
    )


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_curve_from_table(
    edited_df: pd.DataFrame,
    snap_date: date | None = None,
    source: str = "manual",
) -> DivCurve:
    """Reconstruct a DivCurve from an edited sidebar table.

    Expects columns "Tenor", "Years", "q (%)" — produced by
    `DivCurve.as_table` and possibly edited in `st.data_editor`.  Rows
    with NaN in "q (%)" are dropped (defensive — the editor's NumberColumn
    constraints normally prevent this).
    """
    df = edited_df.dropna(subset=["q (%)"]).sort_values("Years").reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows in the dividend-curve table.")
    return DivCurve(
        tenors_yr=tuple(float(t) for t in df["Years"]),
        div_yields=tuple(float(q) / 100.0 for q in df["q (%)"]),
        labels=tuple(str(l) for l in df["Tenor"]),
        source=source,
        snap_date_iso=snap_date.isoformat() if snap_date else None,
    )


def flat_div_curve(
    yield_value: float,
    tenors_yr: tuple[float, ...] | list[float] = (0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0),
    snap_date: date | None = None,
) -> DivCurve:
    """Constant-yield curve at every tenor — used as a fallback / for tests."""
    Ts = tuple(float(T) for T in tenors_yr)
    return DivCurve(
        tenors_yr=Ts,
        div_yields=tuple(float(yield_value) for _ in Ts),
        labels=tuple(f"T={T:.4f}y" for T in Ts),
        source="flat",
        snap_date_iso=snap_date.isoformat() if snap_date else None,
    )


def get_div_yield(curve_or_scalar, T: float) -> float:
    """Convenience accessor: extract a div yield at maturity T from either
    a DivCurve or a plain float (treated as a flat curve).
    """
    if hasattr(curve_or_scalar, "div_yield"):
        return float(curve_or_scalar.div_yield(T))
    return float(curve_or_scalar)