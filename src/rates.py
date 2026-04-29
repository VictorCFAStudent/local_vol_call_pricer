"""
rates.py — US Treasury yield curve: fetch, bootstrap, interpolate.

Source
------
The U.S. Department of the Treasury publishes the Daily Treasury Par Yield
Curve Rates (CMT — Constant Maturity Treasury) as a CSV download keyed by
calendar year, with one row per business day and one column per tenor.
Endpoint:

    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/
    daily-treasury-rates.csv/{YYYY}/all
        ?type=daily_treasury_yield_curve
        &field_tdr_date_value={YYYY}
        &page&_format=csv

We fetch this CSV for the snapshot date's year, locate the row matching the
snapshot date (or the most recent business day on or before it if the snap
date falls on a weekend/holiday), and pull out the tenors we care about.

Bootstrapping
-------------
Treasury reports yields on a *bond-equivalent* basis (semi-annual
compounding).  For maturities ≤ 1y, the underlying instrument is a T-bill
(zero-coupon) so the par yield equals the zero rate; we just convert from
semi-annual to continuous compounding:

    r_cc = 2 · ln(1 + y_BEY / 2)

For 2y, 3y, 5y the published yield is a *par yield* on a coupon-bearing
T-note.  We bootstrap recursively (in increasing maturity order): for each
note, solve for the continuous zero rate r(T) such that the bond — with
semi-annual coupons of `y/2` and face value 1 — prices to par.  Coupon
dates that don't coincide with a previously bootstrapped node (e.g. the
1.5y coupon when bootstrapping 2y) are filled by linear interpolation in
log(DF) space, which is equivalent to assuming piecewise-constant forward
rates between knot points.  The single non-linear equation in r(T) is
solved by Brent's method.

Tenors
------
We keep only tenors at or below 5y, matching the Treasury-published set:

    1m, 2m, 3m, 4m, 6m, 1y, 2y, 3y, 5y     (9 tenors)

Interpolation
-------------
`RateCurve.zero_rate(T)` queries the curve at any maturity by linear
interpolation in log(DF) space (equivalent to piecewise-constant forward
rates between knot points).  Beyond the curve range we extrapolate flat
(the boundary rate).  This is the standard market convention for European
option pricing.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import NamedTuple

import io
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Tenor configuration
# ---------------------------------------------------------------------------

# Ordered tuple of (label, years).  Stop at 5y per project scope.
TENORS: tuple[tuple[str, float], ...] = (
    ("1M", 1 / 12),
    ("2M", 2 / 12),
    ("3M", 3 / 12),
    ("4M", 4 / 12),
    ("6M", 6 / 12),
    ("1Y", 1.0),
    ("2Y", 2.0),
    ("3Y", 3.0),
    ("4Y", 4.0),
    ("5Y", 5.0),
)
TENOR_LABELS: tuple[str, ...] = tuple(t[0] for t in TENORS)
TENOR_YEARS: dict[str, float] = dict(TENORS)

# T-bill tenors are zero-coupon — par yield = zero rate (after compounding fix).
_T_BILL_LABELS: tuple[str, ...] = ("1M", "2M", "3M", "4M", "6M", "1Y")
# T-note tenors require a real bootstrap.
_T_NOTE_LABELS: tuple[str, ...] = ("2Y", "3Y", "5Y")

# Treasury CSV column names.
_TREASURY_COL: dict[str, str] = {
    "1M": "1 Mo",
    "2M": "2 Mo",
    "3M": "3 Mo",
    "4M": "4 Mo",
    "6M": "6 Mo",
    "1Y": "1 Yr",
    "2Y": "2 Yr",
    "3Y": "3 Yr",
    "5Y": "5 Yr",
}

_TREASURY_URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}"
    "&page&_format=csv"
)


# ---------------------------------------------------------------------------
# Curve container
# ---------------------------------------------------------------------------

class RateCurve(NamedTuple):
    """Continuously-compounded zero curve.

    Attributes
    ----------
    tenors_yr : tuple of T values (in years), strictly increasing.
    zero_rates : tuple of continuously-compounded zero rates aligned with tenors_yr.
    source : provenance tag — 'treasury', 'manual', or 'flat'.
    snap_date_iso : ISO-format date string of the snapshot the curve corresponds to,
                    or None if the curve is not date-anchored (e.g. flat fallback).
    """
    tenors_yr: tuple[float, ...]
    zero_rates: tuple[float, ...]
    source: str
    snap_date_iso: str | None

    def zero_rate(self, T: float) -> float:
        """Continuous-compounded zero rate at maturity T (years).

        Linear in log(DF) between curve nodes; flat extrapolation beyond
        the curve range.
        """
        return float(_interp_log_df(T, self.tenors_yr, self.zero_rates))

    def discount_factor(self, T: float) -> float:
        """Discount factor `e^{-r(T)·T}` at maturity T."""
        if T <= 0.0:
            return 1.0
        return float(np.exp(-self.zero_rate(T) * T))

    def as_table(self) -> pd.DataFrame:
        """Return the curve as a tidy DataFrame (for display in the UI).

        The Tenor column always uses canonical Treasury labels ("1M", "2M",
        ..., "5Y") when the maturity matches one within float tolerance —
        critical because `build_curve_from_zero_rates` reconstructs the
        curve by looking up these labels in TENOR_LABELS, so any non-
        canonical label would cause that row's user edit to be silently
        dropped.  Non-canonical maturities (shouldn't occur with the
        current code paths, but kept for defensiveness) fall back to the
        most readable unit — days, weeks, months, or years.
        """
        rows = []
        for T, r in zip(self.tenors_yr, self.zero_rates):
            rows.append({
                "Tenor": _format_tenor(float(T)),
                "Years": float(T),
                "Zero rate (%)": float(r) * 100.0,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tenor formatting
# ---------------------------------------------------------------------------

def _format_tenor(T_years: float, tol: float = 1e-6) -> str:
    """Render a maturity in years as the most intuitive label.

    A canonical Treasury label ("1M", "2M", ..., "5Y") is returned whenever
    `T_years` matches one of `TENOR_YEARS` within `tol`.  This is essential
    for round-tripping through `build_curve_from_zero_rates`, which only
    recognises canonical labels — `1/12`, `2/12`, `4/12` are inexact in
    binary floating-point, so a naive dict lookup misses them and any user
    edit on those rows would otherwise be silently dropped.

    Non-canonical maturities (defensive fallback only — the current code
    paths never produce them) get a readable units label, picking days,
    weeks, months, or years depending on size.
    """
    for label, T_canonical in TENORS:
        if abs(T_years - T_canonical) < tol:
            return label
    days = T_years * 365.25
    if days < 14:
        return f"{int(round(days))}D"
    if days < 60:
        return f"{int(round(days / 7))}W"
    if T_years < 1.0:
        return f"{int(round(T_years * 12))}M"
    if abs(T_years - round(T_years)) < 0.01:
        return f"{int(round(T_years))}Y"
    return f"{T_years:.2f}Y"


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _interp_log_df(
    T_query: float,
    tenors_yr: tuple[float, ...] | list[float] | np.ndarray,
    zero_rates: tuple[float, ...] | list[float] | np.ndarray,
) -> float:
    """Linear interpolation in log(DF) space (piecewise-constant forward rates)."""
    tenors = np.asarray(tenors_yr, dtype=float)
    rates = np.asarray(zero_rates, dtype=float)
    if tenors.size == 0:
        raise ValueError("Cannot interpolate — empty curve.")
    if tenors.size == 1:
        return float(rates[0])

    # Flat extrapolation at the boundaries.
    if T_query <= tenors[0]:
        return float(rates[0])
    if T_query >= tenors[-1]:
        return float(rates[-1])

    # Locate the bracket.
    idx = int(np.searchsorted(tenors, T_query, side="right") - 1)
    T0, T1 = tenors[idx], tenors[idx + 1]
    r0, r1 = rates[idx], rates[idx + 1]

    # log DF(T) is linear in T between knots ⇒ piecewise-constant forwards.
    log_DF0 = -r0 * T0
    log_DF1 = -r1 * T1
    log_DFq = log_DF0 + (T_query - T0) / (T1 - T0) * (log_DF1 - log_DF0)
    return float(-log_DFq / T_query)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_zero_rates(par_yields_pct: dict[str, float]) -> dict[float, float]:
    """Bootstrap continuously-compounded zero rates from Treasury par yields.

    Parameters
    ----------
    par_yields_pct
        Mapping of tenor label → par yield in **percent** (e.g. 4.25 for 4.25 %).
        Tenors not present are simply skipped.

    Returns
    -------
    dict mapping `T` (years) → continuously-compounded zero rate (decimal).
    """
    zeros: dict[float, float] = {}

    # T-bills: par yield = zero rate (zero-coupon instruments).
    # Convert from BEY (semi-annual compounding) to continuous compounding.
    for label in _T_BILL_LABELS:
        if label not in par_yields_pct:
            continue
        T = TENOR_YEARS[label]
        y = float(par_yields_pct[label]) / 100.0
        zeros[T] = float(2.0 * np.log1p(y / 2.0))  # = 2 ln(1 + y/2)

    # T-notes: bootstrap recursively in increasing maturity order.
    for label in _T_NOTE_LABELS:
        if label not in par_yields_pct:
            continue
        T = TENOR_YEARS[label]
        y = float(par_yields_pct[label]) / 100.0
        zeros[T] = _bootstrap_note(T, y, zeros)

    return zeros


def _bootstrap_note(T: float, y_par: float, known_zeros: dict[float, float]) -> float:
    """Solve for r(T) so that a semi-annual coupon bond at par yield y_par prices to 1.

    The bond pays coupons of (y_par / 2) at times 0.5, 1.0, ..., T and a face
    value of 1 at time T.  Coupon dates that fall between previously
    bootstrapped knots and the unknown T are interpolated in log(DF) space
    using a *candidate* r(T) — the equation is then a 1-D root in r(T),
    solved via Brent's method.
    """
    coupon_times = np.arange(0.5, T + 1e-9, 0.5)

    # Capture known nodes as sorted arrays once for fast interp inside `equation`.
    Ts_known = np.array(sorted(known_zeros.keys()), dtype=float)
    rs_known = np.array([known_zeros[t] for t in Ts_known], dtype=float)

    def equation(r_T: float) -> float:
        Ts_aug = np.concatenate([Ts_known, [T]])
        rs_aug = np.concatenate([rs_known, [r_T]])
        # Sort (T should be the new max so this is essentially a no-op, but defensive).
        order = np.argsort(Ts_aug)
        Ts_aug = Ts_aug[order]
        rs_aug = rs_aug[order]

        pv = 0.0
        for t in coupon_times:
            if abs(t - T) < 1e-9:
                pv += (1.0 + y_par / 2.0) * np.exp(-r_T * T)
            else:
                r_t = _interp_log_df(t, Ts_aug, rs_aug)
                pv += (y_par / 2.0) * np.exp(-r_t * t)
        return pv - 1.0

    # Bracket the root in [r_centre − 20 %, r_centre + 20 %] around the
    # BEY-equivalent rate.  The par-zero spread is always small in well-
    # behaved curves (≤ 50 bps for US Treasury), so 20 bps would suffice in
    # practice — but a wider bracket is essentially free for Brent's method
    # (only ~2 extra iterations) and is the difference between handling
    # arbitrary stress scenarios robustly and crashing on edge cases.
    #
    # The earlier `lo = max(lo, -0.05)` floor was wrong: for negative par
    # yields below roughly −5 % it pushed `lo` above `hi`, making brentq
    # raise ValueError.  No guard is needed — equation(r_T) is monotone
    # decreasing in r_T (DFs shrink as r grows), so any bracket containing
    # the root is sufficient regardless of sign.
    r_centre = 2.0 * np.log1p(y_par / 2.0)
    lo, hi = r_centre - 0.20, r_centre + 0.20
    return float(brentq(equation, lo, hi, xtol=1e-10, maxiter=100))


# ---------------------------------------------------------------------------
# Treasury fetch
# ---------------------------------------------------------------------------

class TreasuryFetchError(RuntimeError):
    """Raised when the Treasury daily-yields CSV cannot be fetched / parsed
    for the requested snapshot date."""


def fetch_treasury_yields(snap_date: date, timeout: float = 15.0) -> dict[str, float]:
    """Download the Daily Treasury Par Yield Curve for `snap_date` (or the
    most recent business day on or before it) and return a mapping of
    tenor label → par yield in percent.

    Raises
    ------
    TreasuryFetchError if the network request fails, the snap-date year has
    no published data yet, or no row on or before the snap date is parseable.
    """
    url = _TREASURY_URL_TEMPLATE.format(year=snap_date.year)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TreasuryFetchError(
            f"Could not reach Treasury.gov for {snap_date.year}: {exc}"
        ) from exc

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise TreasuryFetchError(f"Could not parse Treasury CSV: {exc}") from exc

    if "Date" not in df.columns:
        raise TreasuryFetchError(
            f"Treasury CSV missing 'Date' column.  Columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    target = pd.Timestamp(snap_date)
    on_or_before = df[df["Date"] <= target].sort_values("Date")
    if on_or_before.empty:
        raise TreasuryFetchError(
            f"No Treasury yield-curve row on or before {snap_date}.  "
            f"Earliest available row in {snap_date.year}: "
            f"{df['Date'].min().date() if not df.empty else 'n/a'}."
        )

    row = on_or_before.iloc[-1]

    out: dict[str, float] = {}
    for label, col in _TREASURY_COL.items():
        if col not in df.columns:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        out[label] = float(val)

    if not out:
        raise TreasuryFetchError(
            f"Treasury row for {row['Date'].date()} has no parseable yields."
        )

    return out


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_curve_from_yields(
    par_yields_pct: dict[str, float],
    snap_date: date | None = None,
    source: str = "treasury",
) -> RateCurve:
    """Build a RateCurve from a tenor-label → par-yield (percent) dict.

    Canonical tenors that the bootstrap couldn't produce — typically 4Y,
    which the US Treasury CMT schedule omits — are filled by `log-DF`
    linear interpolation between the surrounding bootstrapped knots, the
    same scheme used at pricing time.  This guarantees every canonical
    tenor appears as an editable row in the sidebar, with a value
    consistent with its neighbours, while still letting the user override
    it.  Tenors below the smallest bootstrapped knot or above the largest
    are flat-extrapolated (matching `_interp_log_df`'s boundary rule).
    """
    zeros = bootstrap_zero_rates(par_yields_pct)
    if not zeros:
        raise ValueError("No zero rates produced from input par yields.")

    bootstrapped_Ts = np.array(sorted(zeros.keys()), dtype=float)
    bootstrapped_rs = np.array([zeros[T] for T in bootstrapped_Ts], dtype=float)
    for label in TENOR_LABELS:
        T = TENOR_YEARS[label]
        if T not in zeros:
            zeros[T] = float(_interp_log_df(T, bootstrapped_Ts, bootstrapped_rs))

    Ts = tuple(sorted(zeros.keys()))
    rs = tuple(zeros[T] for T in Ts)
    return RateCurve(
        tenors_yr=Ts,
        zero_rates=rs,
        source=source,
        snap_date_iso=snap_date.isoformat() if snap_date else None,
    )


def build_curve_from_zero_rates(
    zero_rates_pct_by_label: dict[str, float],
    snap_date: date | None = None,
    source: str = "manual",
) -> RateCurve:
    """Build a RateCurve from a tenor-label → continuous zero-rate (percent) dict.

    Used when the user types zero rates directly (skipping the bootstrap).
    """
    Ts: list[float] = []
    rs: list[float] = []
    for label in TENOR_LABELS:
        if label in zero_rates_pct_by_label:
            T = TENOR_YEARS[label]
            r = float(zero_rates_pct_by_label[label]) / 100.0
            Ts.append(T)
            rs.append(r)
    if not Ts:
        raise ValueError("No zero rates provided.")
    return RateCurve(
        tenors_yr=tuple(Ts),
        zero_rates=tuple(rs),
        source=source,
        snap_date_iso=snap_date.isoformat() if snap_date else None,
    )


def flat_curve(rate: float, snap_date: date | None = None) -> RateCurve:
    """Constant-rate curve at every tenor — used as a fallback / for tests."""
    Ts = tuple(TENOR_YEARS[lbl] for lbl in TENOR_LABELS)
    rs = tuple(float(rate) for _ in Ts)
    return RateCurve(
        tenors_yr=Ts,
        zero_rates=rs,
        source="flat",
        snap_date_iso=snap_date.isoformat() if snap_date else None,
    )


def get_rate(curve_or_scalar, T: float) -> float:
    """Convenience accessor: extract a zero rate at maturity T from either a
    RateCurve or a plain float (treated as a flat curve).

    Used by callers that want to support both for backwards compatibility.
    """
    if hasattr(curve_or_scalar, "zero_rate"):
        return float(curve_or_scalar.zero_rate(T))
    return float(curve_or_scalar)