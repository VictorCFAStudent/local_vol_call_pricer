"""
arbitrage_checks.py — Calendar, butterfly and vertical spread checks.

Each check is independent and returns a CheckResult namedtuple.

Mathematical foundations
------------------------
Calendar spread  : total variance w(K,T) = σ²·T must be non-decreasing in T
                   for every fixed moneyness K.  Violation: w(K,T₂) < w(K,T₁)
                   for T₁ < T₂.

Butterfly spread : the risk-neutral density must be non-negative, i.e.
                   d²C/dK² ≥ 0 (Breeden–Litzenberger).  Convexity of the
                   *implied vol* in K is neither necessary nor sufficient —
                   a perfectly arbitrage-free SPX put-skew is typically
                   concave in K on the put wing.  We therefore convert each
                   slice's IVs to call prices via Black-76 and check the
                   non-uniform finite second difference of C(K) directly.

Vertical spread  : call prices must be non-increasing in strike.
                   C(K₁,T) ≥ C(K₂,T) for K₁ < K₂.
                   Call prices are computed from implied vols via Black-76,
                   using a per-slice zero rate r(T) read off the rate curve
                   so that long-dated discounting reflects the term structure
                   (instead of using a single flat rate).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from rates import get_rate

# ---------------------------------------------------------------------------
CheckResult = namedtuple("CheckResult", ["name", "passed", "violations", "details"])

_BFLY_EPS   = 1e-4   # tolerance for butterfly check (accounts for bid-ask noise)
_VERT_EPS   = 1e-6   # tolerance for vertical check


# ---------------------------------------------------------------------------
# Black-76 helpers
# ---------------------------------------------------------------------------

def _black76_call(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """Discounted Black-76 European call price (multiplied by exp(-r·T)).

    Parameters
    ----------
    F     : forward price
    K     : strike price  (= moneyness_pct/100 · F)
    T     : time to expiry in years
    sigma : implied vol (decimal, e.g. 0.22)
    r     : continuously compounded risk-free rate
    """
    if T <= 1e-10 or sigma <= 0 or K <= 0 or F <= 0:
        return max(F - K, 0.0) * np.exp(-r * T)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


# ---------------------------------------------------------------------------
def check_calendar_spread(df: pd.DataFrame) -> CheckResult:
    """Calendar spread check: w(K,T) = σ²·T non-decreasing in T."""
    df = df.copy()
    df["total_var"] = df["implied_vol"] ** 2 * df["time_to_expiry"]
    violations: list[dict] = []

    for m_pct, grp in df.groupby("moneyness_pct"):
        g = grp.sort_values("time_to_expiry").reset_index(drop=True)
        for i in range(len(g) - 1):
            w_i    = g.loc[i,     "total_var"]
            w_next = g.loc[i + 1, "total_var"]
            delta_w = w_next - w_i
            if delta_w < -1e-8:
                violations.append(
                    {
                        "moneyness_pct":      m_pct,
                        "expiry_label":       g.loc[i,     "expiry_label"],
                        "expiry_label_next":  g.loc[i + 1, "expiry_label"],
                        "time_to_expiry":     g.loc[i,     "time_to_expiry"],
                        "time_to_expiry_next":g.loc[i + 1, "time_to_expiry"],
                        "w_short":            round(w_i,    6),
                        "w_long":             round(w_next, 6),
                        "delta_w":            round(delta_w, 6),
                        "severity":           round(abs(delta_w) / max(w_i, 1e-12), 4),
                        # For flag overlay
                        "iv":                 g.loc[i, "implied_vol"],
                    }
                )

    passed = len(violations) == 0
    n_exp = df["expiry_date"].nunique()
    n_k   = df["moneyness_pct"].nunique()
    details = (
        f"{len(violations)} violation(s) across {n_k} moneyness levels × {n_exp} expiry pairs"
        if violations
        else f"No calendar spread violations ({n_k} moneyness levels × {n_exp} expiries checked)."
    )
    return CheckResult(name="Calendar Spread", passed=passed, violations=violations, details=details)


# ---------------------------------------------------------------------------
def check_butterfly_spread(
    df: pd.DataFrame,
    r,
    eps: float = _BFLY_EPS,
) -> CheckResult:
    """Butterfly spread check: d²C/dK² ≥ 0 (Breeden–Litzenberger).

    The risk-neutral density f(K, T) = e^{rT}·d²C/dK² must be non-negative,
    so the second derivative of the *call price* in K must be ≥ 0.  We
    convert each slice's IVs to Black-76 call prices and apply a non-
    uniform centred finite difference — checking the IV smile's convexity
    directly is neither necessary nor sufficient for no-arbitrage.

    Parameters
    ----------
    r : float | RateCurve
        Per-slice zero rate is read off the curve (or the scalar is
        broadcast).  Discounting only rescales C uniformly within a slice,
        so the *sign* of d²C/dK² is r-independent — but we use the right
        rate anyway for severity normalisation.
    """
    violations: list[dict] = []

    for exp_label, grp in df.groupby("expiry_label"):
        g = grp.sort_values("moneyness_pct").reset_index(drop=True)
        if len(g) < 3:
            continue

        ks_pct = g["moneyness_pct"].values.astype(float)
        ivs    = g["implied_vol"].values.astype(float)
        T      = float(g["time_to_expiry"].iloc[0])
        F      = float(g["forward_price"].iloc[0])
        r_T    = get_rate(r, T)

        # Absolute strikes (Breeden–Litzenberger lives in K-space, not in K/F·100).
        Ks = ks_pct / 100.0 * F
        Cs = np.array([_black76_call(F, K, T, sig, r_T) for K, sig in zip(Ks, ivs)])

        for i in range(1, len(g) - 1):
            h1 = Ks[i]     - Ks[i - 1]
            h2 = Ks[i + 1] - Ks[i]
            if h1 <= 0 or h2 <= 0:
                continue
            # Non-uniform centred second derivative of C(K)
            d2C = (
                2.0 / (h1 + h2)
                * (Cs[i + 1] / h2 - Cs[i] * (1.0 / h1 + 1.0 / h2) + Cs[i - 1] / h1)
            )
            if d2C < -eps:
                violations.append(
                    {
                        "expiry_label":  exp_label,
                        "moneyness_pct": ks_pct[i],
                        "time_to_expiry": T,
                        "iv":            ivs[i],
                        "d2C":           round(d2C, 8),
                        "severity":      round(abs(d2C), 8),
                        "r_used":        round(r_T, 6),
                    }
                )

    passed = len(violations) == 0
    n_exp = df["expiry_label"].nunique()
    details = (
        f"{len(violations)} violation(s) across {n_exp} expiry slices"
        if violations
        else f"No butterfly spread violations ({n_exp} expiry slices checked)."
    )
    return CheckResult(name="Butterfly Spread", passed=passed, violations=violations, details=details)


# ---------------------------------------------------------------------------
def check_vertical_spread(df: pd.DataFrame, r) -> CheckResult:
    """Vertical spread check: call prices non-increasing in strike (Black-76).

    The rate argument may be either a scalar (interpreted as a flat rate) or
    a `RateCurve` object — when a curve is passed, each expiry slice is
    discounted with its own zero rate r(T) read off the curve.
    """
    violations: list[dict] = []
    used_rates: list[float] = []  # for the summary line

    for exp_label, grp in df.groupby("expiry_label"):
        g = grp.sort_values("moneyness_pct").reset_index(drop=True)
        if len(g) < 2:
            continue

        T = float(g["time_to_expiry"].iloc[0])
        F = float(g["forward_price"].iloc[0])
        r_T = get_rate(r, T)
        used_rates.append(r_T)

        # Compute call price for each strike K = (moneyness_pct/100) * F
        call_prices = [
            _black76_call(F, (row["moneyness_pct"] / 100.0) * F, T, row["implied_vol"], r_T)
            for _, row in g.iterrows()
        ]
        g = g.copy()
        g["call_price"] = call_prices

        # Reference ATM price for severity normalisation
        atm_rows   = g[g["moneyness_pct"] == 100.0]
        atm_price  = float(atm_rows["call_price"].iloc[0]) if len(atm_rows) > 0 else max(call_prices)
        atm_price  = max(atm_price, 1e-8)

        for i in range(len(g) - 1):
            c_low  = g.loc[i,     "call_price"]
            c_high = g.loc[i + 1, "call_price"]
            breach = c_high - c_low  # must be ≤ 0
            if breach > _VERT_EPS:
                violations.append(
                    {
                        "expiry_label":  exp_label,
                        "time_to_expiry": T,
                        "k1":            g.loc[i,     "moneyness_pct"],
                        "k2":            g.loc[i + 1, "moneyness_pct"],
                        "c1":            round(c_low,  6),
                        "c2":            round(c_high, 6),
                        "breach":        round(breach, 6),
                        "severity":      round(breach / atm_price, 4),
                        "iv":            g.loc[i, "implied_vol"],
                        "r_used":        round(r_T, 6),
                    }
                )

    passed = len(violations) == 0
    n_exp = df["expiry_label"].nunique()
    if used_rates:
        r_lo, r_hi = min(used_rates), max(used_rates)
        rate_str = (
            f"r={r_lo:.2%}" if abs(r_hi - r_lo) < 1e-6
            else f"r∈[{r_lo:.2%}, {r_hi:.2%}]"
        )
    else:
        rate_str = "no rate used"
    details = (
        f"{len(violations)} violation(s) across {n_exp} expiry slices ({rate_str})"
        if violations
        else f"No vertical spread violations ({n_exp} expiry slices, {rate_str})."
    )
    return CheckResult(name="Vertical Spread", passed=passed, violations=violations, details=details)


# ---------------------------------------------------------------------------
def run_all_checks(df: pd.DataFrame, r) -> list[CheckResult]:
    """Run all three checks and return results in order: calendar, butterfly, vertical.

    `r` is required and may be a scalar (flat rate) or a `RateCurve`
    (zero curve looked up per-slice in the butterfly and vertical-spread
    checks).  No silent default — discounting affects severity numbers and
    the caller must own the rate convention.
    """
    return [
        check_calendar_spread(df),
        check_butterfly_spread(df, r=r),
        check_vertical_spread(df, r=r),
    ]
