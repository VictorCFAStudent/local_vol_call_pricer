"""
arbitrage_checks.py — Calendar, butterfly and vertical spread checks.

Each check is independent and returns a CheckResult namedtuple.

Mathematical foundations
------------------------
Calendar spread  : total variance w(K,T) = σ²·T must be non-decreasing in T
                   for every fixed moneyness K.  Violation: w(K,T₂) < w(K,T₁)
                   for T₁ < T₂.

Butterfly spread : implied vol must be convex in moneyness — equivalently,
                   the second derivative d²IV/dK² ≥ 0 at every interior point.
                   Non-uniform finite differences are used because the
                   moneyness grid is not uniformly spaced.

Vertical spread  : call prices must be non-increasing in strike.
                   C(K₁,T) ≥ C(K₂,T) for K₁ < K₂.
                   Call prices are computed from implied vols via Black-76.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
CheckResult = namedtuple("CheckResult", ["name", "passed", "violations", "details"])

_BFLY_EPS   = 1e-4   # tolerance for butterfly check (accounts for bid-ask noise)
_VERT_EPS   = 1e-6   # tolerance for vertical check


# ---------------------------------------------------------------------------
# Black-76 helpers
# ---------------------------------------------------------------------------

def _black76_call(F: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    """Undiscounted Black-76 call price with discount factor exp(-r·T).

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
def check_butterfly_spread(df: pd.DataFrame, eps: float = _BFLY_EPS) -> CheckResult:
    """Butterfly spread check: d²IV/dK² ≥ 0 using non-uniform finite differences."""
    violations: list[dict] = []

    for exp_label, grp in df.groupby("expiry_label"):
        g = grp.sort_values("moneyness_pct").reset_index(drop=True)
        if len(g) < 3:
            continue

        ks  = g["moneyness_pct"].values.astype(float)
        ivs = g["implied_vol"].values.astype(float)
        T   = float(g["time_to_expiry"].iloc[0])

        for i in range(1, len(g) - 1):
            h1 = ks[i]     - ks[i - 1]
            h2 = ks[i + 1] - ks[i]
            if h1 <= 0 or h2 <= 0:
                continue
            # Non-uniform centred second derivative
            d2iv = (
                2.0 / (h1 + h2)
                * (ivs[i + 1] / h2 - ivs[i] * (1.0 / h1 + 1.0 / h2) + ivs[i - 1] / h1)
            )
            if d2iv < -eps:
                violations.append(
                    {
                        "expiry_label":  exp_label,
                        "moneyness_pct": ks[i],
                        "time_to_expiry": T,
                        "iv":            ivs[i],
                        "d2iv":          round(d2iv, 6),
                        "severity":      round(abs(d2iv), 6),
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
def check_vertical_spread(df: pd.DataFrame, r: float = 0.045) -> CheckResult:
    """Vertical spread check: call prices non-increasing in strike (Black-76)."""
    violations: list[dict] = []

    for exp_label, grp in df.groupby("expiry_label"):
        g = grp.sort_values("moneyness_pct").reset_index(drop=True)
        if len(g) < 2:
            continue

        T = float(g["time_to_expiry"].iloc[0])
        F = float(g["forward_price"].iloc[0])

        # Compute call price for each strike K = (moneyness_pct/100) * F
        call_prices = [
            _black76_call(F, (row["moneyness_pct"] / 100.0) * F, T, row["implied_vol"], r)
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
                    }
                )

    passed = len(violations) == 0
    n_exp = df["expiry_label"].nunique()
    details = (
        f"{len(violations)} violation(s) across {n_exp} expiry slices (r={r:.2%})"
        if violations
        else f"No vertical spread violations ({n_exp} expiry slices, r={r:.2%})."
    )
    return CheckResult(name="Vertical Spread", passed=passed, violations=violations, details=details)


# ---------------------------------------------------------------------------
def run_all_checks(df: pd.DataFrame, r: float = 0.045) -> list[CheckResult]:
    """Run all three checks and return results in order: calendar, butterfly, vertical."""
    return [
        check_calendar_spread(df),
        check_butterfly_spread(df),
        check_vertical_spread(df, r=r),
    ]
