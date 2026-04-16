"""
local_vol.py — Dupire local volatility surface via SVI analytical derivatives.

Theory
------
Dupire (1994) in Gatheral total-variance form  w(x,T) = σ²_IV(x,T)·T,
x = ln(K/F):

    σ²_loc(x, T) = (∂w/∂T) / g(x, T)

    g = 1 − (x/w)·(∂w/∂x)
          + ¼·(∂w/∂x)²·(−¼ − 1/w + x²/w²)
          + ½·(∂²w/∂x²)

Numerical strategy — why SVI?
------------------------------
With only ~12 moneyness quotes per expiry (60 %–140 % in the Bloomberg OVME
sheet), numerical second derivatives of the implied vol smile are highly
sensitive to the non-uniform strike spacing — the cluster around ATM
causes CubicSpline to produce wild d²w/dx² values.

Instead we:
  1. Fit Gatheral SVI per expiry slice.
     SVI gives a smooth, arbitrage-aware parametric smile.
  2. Evaluate w(x), dw/dx, d²w/dx² analytically from the SVI formula.
     For SVI:  w(x) = a + b·(ρ·z + √(z²+σ²))   where z = x−m
               dw/dx   = b·(ρ + z/d),             d = √(z²+σ²)
               d²w/dx² = b·σ²/d³                  (always ≥ 0 when b,σ > 0)
  3. Evaluate dw/dT per x column using PCHIP — shape-preserving monotone
     interpolation that guarantees dw/dT ≥ 0 where the data is monotone.
  4. Apply the Dupire formula on the fine (x, T) grid, then display.

Because d²w/dx² = b·σ²/d³ ≥ 0 by construction when the SVI fit is good, the
denominator g is much better-conditioned than with numerical derivatives.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
LocalVolGrid = namedtuple(
    "LocalVolGrid",
    ["K_grid", "T_grid", "LV_grid", "IV_grid", "moneyness", "expiries", "arb_mask"],
)

_LV_CAP       = 1.50    # 150 % hard ceiling — above this is a numerical artefact
_G_FLOOR      = 0.02    # denominator guard
_W_FLOOR      = 1e-8
_DWT_FLOOR    = 1e-6    # floor for dw/dT to avoid holes from numerical noise
_T_MIN_CUTOFF = 0.04    # ~2 weeks
_N_X_FINE     = 200     # fine x grid for SVI evaluation before T-derivative


# ---------------------------------------------------------------------------
# SVI helpers
# ---------------------------------------------------------------------------

def _svi_fit_slice(x: np.ndarray, w_obs: np.ndarray) -> dict | None:
    """Fit SVI to (x, w) pairs.  Returns param dict or None on failure.

    Uses relative squared-error as the objective so that each moneyness
    point contributes equally regardless of its absolute total-variance
    level.  Absolute MSE causes the extreme wing (60 % put with w >> ATM)
    to dominate the fit, collapsing the right wing toward zero.

    Multiple restarts with a data-driven initialisation: m is seeded at the
    observed smile minimum (which may be off-ATM in skewed markets), and a
    is seeded just below that minimum w value.
    """
    def svi_w(p, x):
        a, b, rho, m, sigma = p
        z = x - m
        return a + b * (rho * z + np.sqrt(z**2 + sigma**2))

    def obj(p):
        w_hat = svi_w(p, x)
        if np.any(w_hat <= 0):
            return 1e10
        return float(np.sum(((w_hat - w_obs) / w_obs) ** 2))   # relative MSE

    # Data-driven seed: anchor m at the observed smile minimum
    i_min  = int(np.argmin(w_obs))
    m_init = float(x[i_min])
    a_init = float(w_obs[i_min]) * 0.85

    starts = [
        [a_init, 0.10, -0.70, m_init, 0.10],
        [a_init, 0.15, -0.50, m_init, 0.15],
        [a_init, 0.20, -0.80, m_init, 0.20],
        [a_init, 0.10, -0.30, m_init, 0.05],
        [float(np.mean(w_obs)), 0.10, -0.50, 0.0, 0.10],
    ]
    bounds = [(-0.5, 2.0), (1e-4, 2.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 1.0)]

    best_res, best_fun = None, np.inf
    for x0 in starts:
        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-15})
        if res.fun < best_fun:
            best_fun = res.fun
            best_res = res

    # Reject if average relative error per point exceeds 20 %.
    # A poor SVI is worse than the numerical fallback.
    if best_res is None or best_fun / len(w_obs) > 0.04:
        return None

    a, b, rho, m, sigma = best_res.x
    if not np.all(svi_w(best_res.x, x) > 0):
        return None

    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}


def _svi_eval(p: dict, x: np.ndarray):
    """Return (w, dw_dx, d2w_dx2) analytically from SVI params."""
    a, b, rho, m, sigma = p["a"], p["b"], p["rho"], p["m"], p["sigma"]
    z = x - m
    d = np.sqrt(z**2 + sigma**2)
    w      = a + b * (rho * z + d)
    dw     = b * (rho + z / d)
    d2w    = b * sigma**2 / d**3
    return w, dw, d2w


# ---------------------------------------------------------------------------
def build_local_vol(
    df: pd.DataFrame,
    n_k: int = 80,
    n_t: int = 80,
) -> LocalVolGrid:
    """Compute the Dupire local vol surface.

    Parameters
    ----------
    df  : long-format DataFrame from data_loader
    n_k : moneyness points on the dense display grid
    n_t : expiry points on the dense display grid
    """
    df = df[df["time_to_expiry"] >= _T_MIN_CUTOFF].copy()
    if df.empty:
        raise ValueError("No data after T_MIN_CUTOFF filter.")

    m_raw = np.sort(df["moneyness_pct"].unique())
    T_raw = np.sort(df["time_to_expiry"].unique())
    x_raw = np.log(m_raw / 100.0)

    pivot = (
        df.pivot_table(index="time_to_expiry", columns="moneyness_pct",
                       values="implied_vol", aggfunc="mean")
        .reindex(index=T_raw, columns=m_raw)
    )
    filled = (
        pivot.interpolate(axis=1, limit_direction="both")
        .ffill().bfill()
        .fillna(float(pivot.mean().mean()))
    )
    IV_raw = filled.values                      # (N_T, 9)
    w_raw  = IV_raw**2 * T_raw[:, None]         # (N_T, 9)

    N_T = w_raw.shape[0]

    # ── Step 1: SVI fit per expiry → analytical x-derivatives on fine x grid ─
    x_fine  = np.linspace(x_raw.min(), x_raw.max(), _N_X_FINE)
    w_fine  = np.zeros((N_T, _N_X_FINE))
    dw_dx_fine   = np.zeros_like(w_fine)
    d2w_dx2_fine = np.zeros_like(w_fine)

    for i in range(N_T):
        p = _svi_fit_slice(x_raw, w_raw[i])
        if p is not None:
            w_f, dw_f, d2w_f = _svi_eval(p, x_fine)
            # Sanity-clip: SVI w must be positive
            if np.all(w_f > 0):
                w_fine[i]        = w_f
                dw_dx_fine[i]    = dw_f
                d2w_dx2_fine[i]  = d2w_f
                continue
        # Fallback: interpolate raw w to fine grid, use numerical derivatives
        w_interp = np.interp(x_fine, x_raw, w_raw[i])
        w_fine[i]       = np.maximum(w_interp, _W_FLOOR)
        dw_dx_fine[i]   = np.gradient(w_fine[i],   x_fine)
        d2w_dx2_fine[i] = np.gradient(dw_dx_fine[i], x_fine)

    # ── Step 2: PCHIP dw/dT per fine x column ────────────────────────────────
    # Prepend a synthetic T=0 anchor with w=0 (exact: total variance vanishes
    # at zero expiry).  This gives PCHIP a well-conditioned left boundary,
    # preventing spurious dw/dT estimates at the first real expiry.
    T_pchip = np.concatenate([[0.0], T_raw])
    dw_dT_fine = np.zeros_like(w_fine)

    for j in range(_N_X_FINE):
        col = np.concatenate([[0.0], w_fine[:, j]])
        # PCHIP is monotone-preserving: dw/dT ≥ 0 wherever w is non-decreasing
        pchip = PchipInterpolator(T_pchip, col)
        dw_dT_fine[:, j] = pchip(T_raw, 1)

    # ── Step 3: Gatheral g and local variance on fine (T, x_fine) grid ───────
    x_row  = x_fine[None, :]                    # (1, N_X_FINE)
    w_safe = np.maximum(w_fine, _W_FLOOR)

    g = (
        1.0
        - (x_row / w_safe) * dw_dx_fine
        + 0.25 * dw_dx_fine**2 * (-0.25 - 1.0/w_safe + x_row**2/w_safe**2)
        + 0.5  * d2w_dx2_fine
    )

    arb_mask = g <= _G_FLOOR

    dw_dT_safe = np.maximum(dw_dT_fine, _DWT_FLOOR)
    lv2 = np.where(arb_mask, np.nan, dw_dT_safe / np.where(arb_mask, 1.0, g))
    LV_fine = np.sqrt(np.where(lv2 > 0, lv2, np.nan))
    LV_fine = np.where(LV_fine <= _LV_CAP, LV_fine, np.nan)

    # ── Step 4: Interpolate onto dense display grid ───────────────────────────
    x_dense = np.linspace(x_fine.min(), x_fine.max(), n_k)
    # Log-spaced expiry axis: gives dense coverage at short maturities where
    # the surface changes fastest, and sparser points at the long end.
    t_dense = np.exp(np.linspace(np.log(T_raw.min()), np.log(T_raw.max()), n_t))
    m_dense = np.exp(x_dense) * 100.0

    # IV display: spline in log(T) on original raw grid
    log_T_raw = np.log(T_raw)
    spline_iv = RectBivariateSpline(log_T_raw, x_raw, IV_raw, kx=3, ky=3, s=0)
    IV_dense  = np.clip(spline_iv(np.log(t_dense), x_dense), 0.005, 2.0)

    # LV display: linear RegularGridInterpolator to avoid oscillation
    LV_for_interp = LV_fine.copy()
    for j in range(_N_X_FINE):
        col   = LV_for_interp[:, j]
        valid = ~np.isnan(col)
        if valid.sum() >= 2:
            col[~valid] = np.interp(T_raw[~valid], T_raw[valid], col[valid])
        elif valid.sum() == 1:
            col[~valid] = col[valid][0]
        else:
            col[:] = np.interp(x_fine[j], x_raw, IV_raw.mean(axis=0))

    # Light smoothing: reduces front-end spikiness and wing noise without
    # distorting the well-conditioned ATM region.
    # sigma=(T_axis, x_axis) — more smoothing across strikes than across maturities.
    LV_for_interp = gaussian_filter(LV_for_interp, sigma=(0.6, 1.5))
    LV_for_interp = np.clip(LV_for_interp, 0.005, _LV_CAP)

    rgi_lv = RegularGridInterpolator(
        (T_raw, x_fine), LV_for_interp,
        method="linear", bounds_error=False, fill_value=None,
    )
    rgi_arb = RegularGridInterpolator(
        (T_raw, x_fine), arb_mask.astype(float),
        method="linear", bounds_error=False, fill_value=0.0,
    )

    T_q, X_q = np.meshgrid(t_dense, x_dense, indexing="ij")
    LV_dense  = np.clip(rgi_lv((T_q, X_q)),  0.0, _LV_CAP)
    arb_dense = rgi_arb((T_q, X_q)) > 0.4
    LV_dense  = np.where(arb_dense, np.nan, LV_dense)

    K_grid, T_grid = np.meshgrid(m_dense, t_dense)

    return LocalVolGrid(
        K_grid=K_grid, T_grid=T_grid,
        LV_grid=LV_dense, IV_grid=IV_dense,
        moneyness=m_dense, expiries=t_dense,
        arb_mask=arb_dense,
    )


# ---------------------------------------------------------------------------
def atm_comparison(lv_grid: LocalVolGrid) -> pd.DataFrame:
    """ATM IV and ATM local vol per expiry T."""
    atm_idx = int(np.argmin(np.abs(lv_grid.moneyness - 100.0)))
    df = pd.DataFrame({
        "time_to_expiry": lv_grid.expiries,
        "iv_atm":  lv_grid.IV_grid[:, atm_idx],
        "lv_atm":  lv_grid.LV_grid[:, atm_idx],
    })
    df["lv_iv_ratio"] = df["lv_atm"] / df["iv_atm"]
    return df.dropna(subset=["lv_atm"])
