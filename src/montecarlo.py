"""
montecarlo.py — Monte Carlo pricing of European calls using the Dupire local vol surface.

Simulation scheme
-----------------
Log-Euler discretisation:

    log S_{t+Δt} = log S_t + (r − q − ½ σ²_loc(m_t, t)) · Δt + σ_loc · √Δt · Z_t

    where:
        m_t = (S_t / F_t) × 100       forward moneyness in %
        F_t = S_0 · exp((r − q) · t)  forward price at time t
        Z_t ~ N(0, 1)                  independent draw each step

Out-of-bounds handling
----------------------
When m_t or t falls outside the local vol surface domain, the nearest
boundary value is used (clamp).  The fraction of clamped steps is returned
as a diagnostic — a high value signals that the surface is too narrow for
the chosen strike or maturity.

Convergence criterion
---------------------
Paths are simulated in batches of 500.  After each batch, stop if:

    |P(n) − P(n − 500)| / max(P(n), _EPS_FLOOR) < eps   and   n ≥ 5 000

where P(n) is the running average discounted payoff after n paths.
Hard cap: 50 000 paths.  If the cap is reached, the result is still valid
but `converged` is set to False.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from local_vol import LocalVolGrid

# ---------------------------------------------------------------------------
MCResult = namedtuple(
    "MCResult",
    ["price", "std_error", "n_paths", "converged", "clamp_pct"],
)

_MIN_PATHS   = 5_000
_MAX_PATHS   = 50_000
_BATCH_SIZE  = 500    # also the convergence window size
_EPS_FLOOR   = 1e-8   # prevents div-by-zero when price ≈ 0


# ---------------------------------------------------------------------------
def _build_lv_interpolator(lv_grid: LocalVolGrid) -> RegularGridInterpolator:
    """Build a RegularGridInterpolator from the LocalVolGrid.

    NaN holes (arbitrage-masked points) are filled column-by-column via
    linear interpolation over the T axis before building the interpolator,
    so every grid node has a finite value.
    """
    LV = lv_grid.LV_grid.copy()

    # Fill NaN in the T direction, column by column
    for j in range(LV.shape[1]):
        col   = LV[:, j]
        valid = ~np.isnan(col)
        if valid.sum() >= 2:
            col[~valid] = np.interp(
                lv_grid.expiries[~valid],
                lv_grid.expiries[valid],
                col[valid],
            )
        elif valid.sum() == 1:
            col[:] = col[valid][0]
        LV[:, j] = col

    # Global fallback for any column that was entirely NaN
    fallback = float(np.nanmean(LV))
    LV = np.where(np.isnan(LV), fallback, LV)

    return RegularGridInterpolator(
        (lv_grid.expiries, lv_grid.moneyness),
        LV,
        method="linear",
        bounds_error=False,
        fill_value=None,  # clamping before query makes this unreachable
    )


# ---------------------------------------------------------------------------
def price_european_call(
    lv_grid:        LocalVolGrid,
    S0:             float,
    K:              float,
    T:              float,
    r:              float,
    q:              float = 0.0,
    steps_per_year: int   = 252,
    eps:            float = 0.001,
    seed:           int | None = None,
) -> MCResult:
    """Monte Carlo price of a European call using the Dupire local vol surface.

    Parameters
    ----------
    lv_grid        : LocalVolGrid from build_local_vol()
    S0             : current spot price
    K              : strike
    T              : option maturity in years
    r              : risk-free rate (continuous, annualised)
    q              : dividend yield (continuous, annualised)
    steps_per_year : Euler steps per year  (252 = daily, 52 = weekly)
    eps            : relative convergence threshold (default 0.001 = 0.1%)
    seed           : RNG seed for reproducibility (None = non-reproducible)

    Returns
    -------
    MCResult namedtuple:
        price      — discounted expected payoff
        std_error  — standard error of the Monte Carlo estimate
        n_paths    — total number of paths simulated
        converged  — True if convergence criterion was met before the cap
        clamp_pct  — % of simulation steps where spot or time was out of
                     the local vol surface range and was clamped
    """
    rng    = np.random.default_rng(seed)
    interp = _build_lv_interpolator(lv_grid)

    m_min = float(lv_grid.moneyness.min())
    m_max = float(lv_grid.moneyness.max())
    t_min = float(lv_grid.expiries.min())
    t_max = float(lv_grid.expiries.max())

    n_steps = max(1, int(round(T * steps_per_year)))
    dt      = T / n_steps
    sqrt_dt = np.sqrt(dt)
    disc    = np.exp(-r * T)

    # Pre-compute time grid (same for every path)
    t_grid = np.arange(n_steps) * dt          # shape (n_steps,)
    t_grid_clamped = np.clip(t_grid, t_min, t_max)
    t_steps_clamped = np.sum((t_grid < t_min) | (t_grid > t_max))

    # Running accumulators
    total_sum    = 0.0
    total_sq_sum = 0.0
    total_clamped_steps = 0
    total_steps_done    = 0
    n_paths_done        = 0
    converged           = False

    while n_paths_done < _MAX_PATHS:
        bs = min(_BATCH_SIZE, _MAX_PATHS - n_paths_done)

        # ── Vectorised simulation of bs paths ──────────────────────────────
        log_S          = np.full(bs, np.log(S0))   # (bs,)
        batch_clamped  = int(t_steps_clamped) * bs  # t-clamps: same for all paths

        for step_idx in range(n_steps):
            S_t = np.exp(log_S)                              # (bs,)
            t_i = t_grid[step_idx]
            F_t = S0 * np.exp((r - q) * t_i)                # forward price at time t
            m_t = (S_t / F_t) * 100.0                       # forward moneyness %

            m_q = np.clip(m_t, m_min, m_max)                # (bs,) — clamp moneyness
            t_q = float(t_grid_clamped[step_idx])            # scalar — clamp time

            # Count per-path moneyness clamps
            batch_clamped += int(np.sum((m_t < m_min) | (m_t > m_max)))

            # Batch local vol lookup: query shape (bs, 2)
            query = np.column_stack([np.full(bs, t_q), m_q])
            sigma = np.maximum(interp(query), 0.001)         # floor at 0.1%

            Z     = rng.standard_normal(bs)
            log_S += (r - q - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z

        # ── Payoffs ────────────────────────────────────────────────────────
        payoffs   = np.maximum(np.exp(log_S) - K, 0.0) * disc   # (bs,)
        batch_sum = float(payoffs.sum())

        total_sum    += batch_sum
        total_sq_sum += float((payoffs ** 2).sum())
        total_clamped_steps += batch_clamped
        total_steps_done    += bs * n_steps
        n_paths_done        += bs

        # ── Convergence check ──────────────────────────────────────────────
        # Compare P(n) to P(n − batch_size): did the last 500 paths move
        # the running average by more than eps (relative)?
        if n_paths_done >= _MIN_PATHS and n_paths_done > bs:
            P_n      = total_sum / n_paths_done
            P_n_prev = (total_sum - batch_sum) / (n_paths_done - bs)
            diff     = abs(P_n - P_n_prev)
            denom    = max(abs(P_n), _EPS_FLOOR)
            if diff / denom < eps:
                converged = True
                break

    # ── Final statistics ───────────────────────────────────────────────────
    price     = total_sum / n_paths_done
    variance  = total_sq_sum / n_paths_done - price ** 2
    std_error = np.sqrt(max(variance, 0.0) / n_paths_done)
    clamp_pct = 100.0 * total_clamped_steps / total_steps_done if total_steps_done > 0 else 0.0

    return MCResult(
        price=float(price),
        std_error=float(std_error),
        n_paths=n_paths_done,
        converged=converged,
        clamp_pct=float(clamp_pct),
    )
