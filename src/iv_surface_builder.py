"""
iv_surface_builder.py — Grid interpolation and surface smoothing.

Takes the long-format vol DataFrame and produces a dense 2-D grid
(moneyness × time-to-expiry) suitable for surface plotting.

Interpolation uses RectBivariateSpline (bicubic, s=0) in log(T) × K space.
Working in log(T) is essential: the expiry axis spans 1 day to 6+ years, so
a linear T axis would compress the front-end where vol spikes the most.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
SurfaceGrid = namedtuple(
    "SurfaceGrid",
    ["K_grid", "T_grid", "IV_grid", "moneyness", "expiries", "expiry_labels"],
)

_IV_MIN = 0.005   # 0.5%  — hard floor for clipped output
_IV_MAX = 2.0     # 200%  — hard ceiling


# ---------------------------------------------------------------------------
def build_surface(
    df: pd.DataFrame,
    n_k: int = 60,
    n_t: int = 60,
) -> SurfaceGrid:
    """Interpolate the raw data onto a dense regular (K, T) grid.

    Parameters
    ----------
    df  : long-format DataFrame from data_loader
    n_k : number of moneyness grid points
    n_t : number of expiry grid points

    Returns
    -------
    SurfaceGrid namedtuple with K_grid, T_grid, IV_grid arrays plus the
    1-D axis vectors and expiry label strings.
    """
    moneyness_raw = np.sort(df["moneyness_pct"].unique())
    expiries_raw  = np.sort(df["time_to_expiry"].unique())

    # Build dense output axes
    k_dense = np.linspace(moneyness_raw.min(), moneyness_raw.max(), n_k)
    t_dense = np.linspace(expiries_raw.min(),  expiries_raw.max(),  n_t)

    IV_grid = _spline_interpolate(df, moneyness_raw, expiries_raw, k_dense, t_dense)

    K_grid, T_grid = np.meshgrid(k_dense, t_dense)

    # Expiry labels for the raw expiry axis (not the dense grid)
    label_map = (
        df.drop_duplicates("time_to_expiry")
        .set_index("time_to_expiry")["expiry_label"]
        .to_dict()
    )
    expiry_labels = [label_map.get(t, f"{t:.3f}y") for t in expiries_raw]

    return SurfaceGrid(
        K_grid=K_grid,
        T_grid=T_grid,
        IV_grid=IV_grid,
        moneyness=k_dense,
        expiries=t_dense,
        expiry_labels=expiry_labels,
    )


def _spline_interpolate(
    df: pd.DataFrame,
    moneyness_raw: np.ndarray,
    expiries_raw: np.ndarray,
    k_out: np.ndarray,
    t_out: np.ndarray,
) -> np.ndarray:
    """Bicubic spline on log(T) × K grid — exact fit, preserves front-end spike.

    Working in log(T) space is essential: the expiry axis spans 1 day to
    6+ years so bicubic spline in linear T would wildly under-represent the
    near-term skew.
    """
    pivot = df.pivot_table(
        index="time_to_expiry",
        columns="moneyness_pct",
        values="implied_vol",
        aggfunc="mean",
    ).reindex(index=expiries_raw, columns=moneyness_raw)

    filled = (
        pivot.interpolate(axis=1, limit_direction="both")
        .ffill()
        .bfill()
        .fillna(pivot.mean().mean())
    )

    log_T_raw = np.log(expiries_raw)
    log_T_out = np.log(t_out)

    try:
        # s=0 → exact interpolation through all data points
        spline = RectBivariateSpline(log_T_raw, moneyness_raw, filled.values, kx=3, ky=3, s=0)
        IV_grid = spline(log_T_out, k_out)
        return np.clip(IV_grid, _IV_MIN, _IV_MAX)
    except Exception:
        # Fallback: linear spline (kx=ky=1) — less smooth but always stable
        spline = RectBivariateSpline(log_T_raw, moneyness_raw, filled.values, kx=1, ky=1, s=0)
        IV_grid = spline(log_T_out, k_out)
        return np.clip(IV_grid, _IV_MIN, _IV_MAX)


# ---------------------------------------------------------------------------
def interpolate_slice(
    strikes: np.ndarray,
    vols: np.ndarray,
    k_new: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """1-D interpolation for a single expiry smile.

    Parameters
    ----------
    strikes : known moneyness values (sorted ascending)
    vols    : implied vols at those strikes (decimal)
    k_new   : query moneyness values
    method  : scipy interp1d kind ('cubic', 'linear', …)
    """
    f = interp1d(strikes, vols, kind=method, fill_value="extrapolate", bounds_error=False)
    return np.clip(f(k_new), _IV_MIN, _IV_MAX)


# ---------------------------------------------------------------------------
def svi_fit(
    strikes: np.ndarray,
    vols: np.ndarray,
    F: float,
    T: float,
) -> dict:
    """Fit Gatheral SVI parameters {a, b, ρ, m, σ} by least squares.

    Model: w(x) = a + b·(ρ·(x-m) + √((x-m)² + σ²))
    where x = ln(K/F) is log-moneyness and w = σ_IV² · T total variance.

    Returns
    -------
    dict with keys: a, b, rho, m, sigma, success, residual
    """
    x = np.log(np.asarray(strikes, dtype=float) / F)
    w_obs = np.asarray(vols, dtype=float) ** 2 * T

    def svi_w(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        return a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma**2))

    def objective(params: np.ndarray) -> float:
        w_hat = svi_w(params, x)
        if np.any(w_hat <= 0):
            return 1e10
        return float(np.sum((w_hat - w_obs) ** 2))

    x0 = [float(np.mean(w_obs)), 0.1, -0.5, 0.0, 0.1]
    bounds = [(-0.5, 1.0), (1e-4, 1.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 1.0)]
    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-14})
    a, b, rho, m, sigma = result.x
    return {
        "a": float(a), "b": float(b), "rho": float(rho),
        "m": float(m), "sigma": float(sigma),
        "success": bool(result.success),
        "residual": float(result.fun),
    }
