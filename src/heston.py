"""
heston.py — Heston stochastic volatility model.

Theory
------
Heston (1993) dynamics under the risk-neutral measure:

    dS = (r − q) S dt  + √v · S dW^S
    dv = κ (θ − v) dt  + ξ √v  dW^v
    Corr(dW^S, dW^v) = ρ

Five parameters:
    v₀   initial variance          κ   mean-reversion speed
    θ    long-run variance          ξ   vol-of-vol
    ρ    spot–variance correlation

Numerical strategy
------------------
1. Semi-analytic European pricing via Fourier inversion of the Heston
   characteristic function (Albrecher et al. formulation — the "little
   Heston trap" fix — to avoid branch-cut discontinuities).
2. Calibration to a market IV surface: differential evolution (global)
   followed by L-BFGS-B polish.  Objective is vega-weighted price error
   (ΔC/vega ≈ Δσ), with maturity and moneyness filters to exclude
   noisy ultra-short and deep-OTM options.
3. Monte Carlo pricing via full-truncation Euler for arbitrary payoffs
   (vanilla, digital, barrier with American or European monitoring).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm

from montecarlo import MCResult
from rates import get_rate

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
HestonParams = namedtuple(
    "HestonParams", ["v0", "kappa", "theta", "xi", "rho"],
)

HestonCalibResult = namedtuple(
    "HestonCalibResult",
    ["params", "rmse_iv", "max_err_iv", "mean_err_iv",
     "n_points", "success", "detail_df"],
)


# ---------------------------------------------------------------------------
# Black-76 helpers (scalar + vectorised)
# ---------------------------------------------------------------------------

def _b76_call(F, K, T, r, sigma):
    """Black-76 European call price (scalar)."""
    if sigma <= 0 or T <= 0:
        return max(np.exp(-r * T) * (F - K), 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


def _b76_vega(F, K, T, r, sigma):
    """Black-76 vega (scalar)."""
    if sigma <= 0 or T <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return float(np.exp(-r * T) * F * norm.pdf(d1) * sqrt_T)


def _b76_call_vec(F, K, T, r, sigma):
    """Vectorised Black-76 call price."""
    F, K = np.asarray(F, float), np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), 1e-10)
    sigma = np.asarray(sigma, float)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _b76_vega_vec(F, K, T, r, sigma):
    """Vectorised Black-76 vega."""
    F, K = np.asarray(F, float), np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), 1e-10)
    sigma = np.asarray(sigma, float)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return np.exp(-r * T) * F * norm.pdf(d1) * sqrt_T


def _b76_iv(price, F, K, T, r, sigma0=0.3, tol=1e-8, max_iter=50):
    """Invert a call price to Black-76 IV via Newton-Raphson.

    Silent floor convention
    -----------------------
    When the input price is below intrinsic value (`price ≤ e^{-rT}·max(F-K, 0)`)
    or T ≤ 0, no real implied vol exists — Black-76 cannot reproduce a
    sub-intrinsic price.  Instead of raising, we return a sentinel value
    of `0.001` (10 bps).  This is a pragmatic choice for *display* paths
    (calibration reports, smile overlays) where a NaN would propagate
    into Plotly traces and break the chart, but it does mean a caller
    cannot tell apart "true 10-bp IV" from "intrinsic-violation point".
    Callers that need to flag intrinsic violations explicitly should
    test `price ≤ intrinsic` themselves before calling this.

    The same convention applies to the vectorised `_b76_iv_vec`.
    """
    if T <= 0:
        return 0.001
    intrinsic = np.exp(-r * T) * max(F - K, 0.0)
    if price <= intrinsic + 1e-12:
        return 0.001
    sigma = sigma0
    for _ in range(max_iter):
        p = _b76_call(F, K, T, r, sigma)
        v = _b76_vega(F, K, T, r, sigma)
        if v < 1e-14:
            break
        sigma -= (p - price) / v
        sigma = np.clip(sigma, 1e-4, 5.0)
        if abs(p - price) < tol:
            break
    return float(sigma)


def _b76_iv_vec(prices, F, K_arr, T, r, sigma0_arr=None, tol=1e-8, max_iter=50):
    """Vectorised Black-76 IV inversion — Newton-Raphson on arrays.

    ~50× faster than calling _b76_iv in a Python loop over strikes.
    """
    prices = np.asarray(prices, float)
    K_arr = np.asarray(K_arr, float)
    n = len(prices)

    if sigma0_arr is None:
        sigma = np.full(n, 0.3)
    else:
        sigma = np.asarray(sigma0_arr, float).copy()

    # Mask: which points still need iterating
    intrinsic = np.exp(-r * T) * np.maximum(F - K_arr, 0.0)
    active = prices > intrinsic + 1e-12

    for _ in range(max_iter):
        if not active.any():
            break
        p = _b76_call_vec(F, K_arr[active], T, r, sigma[active])
        v = _b76_vega_vec(F, K_arr[active], T, r, sigma[active])
        good = v > 1e-14
        update = np.zeros(active.sum())
        update[good] = (p[good] - prices[active][good]) / v[good]
        sigma_active = sigma[active] - update
        sigma_active = np.clip(sigma_active, 1e-4, 5.0)
        sigma[active] = sigma_active
        # Check convergence
        converged = np.abs(p - prices[active]) < tol
        idx_active = np.where(active)[0]
        active[idx_active[converged]] = False

    # Floor for points that never had enough premium
    sigma[~(prices > intrinsic + 1e-12)] = 0.001
    return sigma


# ---------------------------------------------------------------------------
# Heston characteristic function — Albrecher et al. formulation
# ---------------------------------------------------------------------------

def _heston_cf(phi, T, v0, kappa, theta, xi, rho, j):
    """
    Heston log-spot characteristic function f_j(φ), j ∈ {1, 2}.

    Uses the Albrecher et al. sign convention (the "little Heston trap"
    fix): g is defined with (b − ρξiφ − d) in the numerator, so that
    g·exp(−dT) → 0 for large d·T and no branch-cut jump occurs.

    Parameters
    ----------
    phi : 1-D ndarray   Integration variable (real, > 0).
    T   : float          Time to expiry.
    v0, kappa, theta, xi, rho : Heston model parameters.
    j   : int            1 or 2 (selects P₁ or P₂ in the pricing formula).
    """
    b = kappa - rho * xi if j == 1 else kappa
    u = 0.5 if j == 1 else -0.5
    a = kappa * theta

    d = np.sqrt(
        (rho * xi * 1j * phi - b) ** 2
        - xi ** 2 * (2 * u * 1j * phi - phi ** 2)
    )

    g = (b - rho * xi * 1j * phi - d) / (b - rho * xi * 1j * phi + d)
    exp_dT = np.exp(-d * T)

    D = ((b - rho * xi * 1j * phi - d) / xi ** 2) * (
        (1 - exp_dT) / (1 - g * exp_dT)
    )
    C = (a / xi ** 2) * (
        (b - rho * xi * 1j * phi - d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )

    return np.exp(C + D * v0)


# ---------------------------------------------------------------------------
# Semi-analytic Heston European call price
# ---------------------------------------------------------------------------

def heston_call(S, K, T, r, q, v0, kappa, theta, xi, rho, n_phi=1000):
    """
    European call price under Heston via Fourier inversion.

    C = S·e^{−qT}·P₁ − K·e^{−rT}·P₂

    where P_j = ½ + (1/π) ∫₀^∞ Re[e^{iφ(x − ln K)} f_j(φ)/(iφ)] dφ
    and  x = ln S + (r−q)T  =  ln F.
    """
    if T <= 0:
        return max(S - K, 0.0)

    phi = np.linspace(1e-8, 100, n_phi)
    x = np.log(S) + (r - q) * T  # ln(forward)

    price = 0.0
    for j in [1, 2]:
        cf = _heston_cf(phi, T, v0, kappa, theta, xi, rho, j)
        integrand = np.real(
            np.exp(1j * phi * (x - np.log(K))) * cf / (1j * phi)
        )
        P = np.clip(0.5 + np.trapezoid(integrand, phi) / np.pi, 0, 1)

        if j == 1:
            price += S * np.exp(-q * T) * P
        else:
            price -= K * np.exp(-r * T) * P

    return max(float(price), 0.0)


def _heston_call_fwd_batch(F, K_arr, T, r, v0, kappa, theta, xi, rho, phi):
    """
    Vectorised Heston call prices using the *market forward* F directly.

    C = e^{−rT} (F · P₁ − K · P₂)

    This avoids any spot/forward mismatch during calibration: the strikes K
    are computed from the Bloomberg forward, and the characteristic function
    sees x = ln(F) exactly.

    Parameters
    ----------
    F     : float        Market forward for this expiry.
    K_arr : 1-D ndarray  Strikes (same expiry).
    phi   : 1-D ndarray  Pre-allocated integration grid.
    """
    x = np.log(F)
    ln_K = np.log(K_arr)
    disc = np.exp(-r * T)

    prices = np.zeros(len(K_arr))
    for j in [1, 2]:
        cf = _heston_cf(phi, T, v0, kappa, theta, xi, rho, j)  # (n_phi,)
        # (n_K, n_phi) — vectorise over strikes
        phase = np.exp(1j * phi[None, :] * (x - ln_K[:, None]))
        integrands = np.real(phase * cf[None, :] / (1j * phi[None, :]))
        P = np.clip(0.5 + np.trapezoid(integrands, phi, axis=1) / np.pi, 0, 1)

        if j == 1:
            prices += F * disc * P
        else:
            prices -= K_arr * disc * P

    return np.maximum(prices, 0.0)


# ---------------------------------------------------------------------------
# Heston smile — for plotting the calibrated model against market quotes
# ---------------------------------------------------------------------------

def heston_smile(S, T, r, q, params, m_pct_grid, n_phi=1000):
    """
    Compute Heston implied vols on a moneyness-% grid for a single expiry.

    Returns 1-D ndarray of IVs (decimal) aligned with *m_pct_grid*.
    """
    v0, kappa, theta, xi, rho = params
    F = S * np.exp((r - q) * T)
    K_grid = m_pct_grid / 100.0 * F
    phi = np.linspace(1e-8, 100, n_phi)
    prices = _heston_call_fwd_batch(F, K_grid, T, r, v0, kappa, theta, xi, rho, phi)
    return _b76_iv_vec(prices, F, K_grid, T, r)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    vol_df: pd.DataFrame,
    S: float,
    r,
    q: float,
    max_slices: int = 12,
    min_T: float = 0.05,
    m_range: tuple[float, float] = (70.0, 130.0),
) -> HestonCalibResult:
    """
    Calibrate Heston parameters to the market implied-vol surface.

    Strategy
    --------
    1. Filter out ultra-short maturities (T < *min_T*) and extreme wings
       outside *m_range* — those are poorly captured by 5 Heston params.
    2. Select up to *max_slices* expiry slices (evenly spaced) for speed.
    3. Minimise  Σ (ΔC/vega)²  ≈  Σ (Δσ)²  via differential evolution
       (global) + L-BFGS-B polish.  Vega-weighting approximates IV-space
       fitting without costly Newton inversion in the inner loop.
    4. Re-compute per-point IV errors on the **full** surface for reporting.

    Parameters
    ----------
    r : float | RateCurve
        Risk-free rate.  May be a scalar (flat) or a `RateCurve` — when a
        curve is passed, each expiry slice is priced and discounted with
        its own zero rate r(T) read off the curve.
    q : float
        Accepted for API symmetry with the MC pricers, but **inert in
        calibration** — Heston/Black-76 prices and vegas under a
        moneyness-quoted IV surface are positively homogeneous of degree 1
        in (F, K), so re-scaling the forward by `exp(−Δq·T)` rescales
        every term in the vega-weighted residual by the same factor.  The
        DE optimum is therefore independent of `q` (and of the level of
        `r`).  The dividend curve does affect MC pricing where strikes
        are absolute, but not this calibration.
    """
    # ── 1. Filter and select representative expiries ─────────────────────
    df_filt = vol_df[
        (vol_df["time_to_expiry"] >= min_T)
        & (vol_df["moneyness_pct"] >= m_range[0])
        & (vol_df["moneyness_pct"] <= m_range[1])
    ].copy()
    T_all = np.sort(vol_df["time_to_expiry"].unique())
    T_filt = np.sort(df_filt["time_to_expiry"].unique())
    if len(T_filt) > max_slices:
        sel_idx = np.round(np.linspace(0, len(T_filt) - 1, max_slices)).astype(int)
        T_sel = T_filt[sel_idx]
    else:
        T_sel = T_filt
    df_cal = df_filt[df_filt["time_to_expiry"].isin(T_sel)].copy()

    # ── 2. Pre-compute market data ───────────────────────────────────────
    F_arr = df_cal["forward_price"].values.astype(float)
    m_arr = df_cal["moneyness_pct"].values.astype(float)
    K_arr = m_arr / 100.0 * F_arr
    T_arr = df_cal["time_to_expiry"].values.astype(float)
    iv_mkt = df_cal["implied_vol"].values.astype(float)

    # Group by T for batch Fourier evaluation
    unique_T = np.unique(T_arr)
    groups = []
    for T in unique_T:
        mask = T_arr == T
        groups.append((T, K_arr[mask], F_arr[mask][0],
                        iv_mkt[mask], np.where(mask)[0]))

    phi_cal = np.linspace(1e-8, 100, 300)

    # Resolve per-row rate from the curve (or broadcast a scalar) so that
    # vectorised B-76 prices and vegas use slice-appropriate discounting.
    r_arr = np.array([get_rate(r, float(t)) for t in T_arr])

    # Per-slice rate — picked once per group and reused inside the DE objective.
    r_by_T: dict[float, float] = {float(T): float(get_rate(r, float(T))) for T in unique_T}

    # Pre-compute market prices and vega for the objective
    C_mkt = _b76_call_vec(F_arr, K_arr, T_arr, r_arr, iv_mkt)
    vega  = _b76_vega_vec(F_arr, K_arr, T_arr, r_arr, iv_mkt)
    # Vega floor: single global 25th-percentile cap.  Caps the *weight*
    # `1/vega` from above, so deep-OTM points (tiny vega → huge weight)
    # don't dominate the residual — but every point still contributes.
    # A single global floor is simpler and more transparent than the
    # per-slice 5th-percentile rule it replaced; with the T / moneyness
    # filters already applied upstream, the resulting weighting is
    # well-behaved across slices.
    vega_floor = max(float(np.percentile(vega, 25)), 1e-2)
    w = 1.0 / np.maximum(vega, vega_floor)

    # ── 3. Objective — vega-weighted price errors ────────────────────────
    # With the T / moneyness filters already applied, this is fast and
    # well-behaved (no deep-OTM noise to distort the fit).
    def objective(p):
        v0, kappa, theta, xi, rho = p
        total = 0.0
        with np.errstate(all="ignore"):
            for T, K_g, F_g, iv_g, idx in groups:
                r_T = r_by_T[float(T)]
                C_mod = _heston_call_fwd_batch(
                    F_g, K_g, T, r_T, v0, kappa, theta, xi, rho, phi_cal,
                )
                if np.any(np.isnan(C_mod)):
                    return 1e12
                total += float(np.sum((w[idx] * (C_mod - C_mkt[idx])) ** 2))
        return total

    # ── Bounds (physically reasonable for equity indices) ────────────────
    bounds = [
        (0.001, 0.50),    # v0   — up to ~70% spot vol
        (0.1,  10.0),     # kappa
        (0.001, 0.50),    # theta — up to ~70% long-run vol
        (0.05,  3.0),     # xi
        (-0.99, 0.20),    # rho  — equity skew is always negative
    ]

    # Data-driven seed from ATM implied vols
    atm_cal = df_filt[df_filt["moneyness_pct"] == 100.0].sort_values("time_to_expiry")
    if atm_cal.empty:
        atm_cal = vol_df[vol_df["moneyness_pct"] == 100.0].sort_values("time_to_expiry")
    iv_front = float(atm_cal["implied_vol"].iloc[0]) if len(atm_cal) else 0.20
    iv_long = float(atm_cal["implied_vol"].iloc[-1]) if len(atm_cal) else 0.20
    x0 = [iv_front ** 2, 2.0, iv_long ** 2, 0.5, -0.70]

    # ── 4. Global optimisation ───────────────────────────────────────────
    result = differential_evolution(
        objective,
        bounds,
        x0=x0,
        maxiter=200,
        tol=1e-8,
        seed=42,
        polish=True,       # auto L-BFGS-B refinement
        popsize=10,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )

    v0, kappa, theta, xi, rho = result.x
    params = HestonParams(
        v0=float(v0), kappa=float(kappa), theta=float(theta),
        xi=float(xi), rho=float(rho),
    )

    # ── 5. Detailed fit report (all points, fine integration grid) ───────
    phi_fine = np.linspace(1e-8, 100, 1000)
    detail_rows = []

    for T_val in T_all:
        sl = vol_df[vol_df["time_to_expiry"] == T_val]
        F = float(sl["forward_price"].iloc[0])
        m_vals = sl["moneyness_pct"].values.astype(float)
        K_vals = m_vals / 100.0 * F
        iv_mkt_vals = sl["implied_vol"].values.astype(float)
        labels = sl["expiry_label"].values
        r_val = get_rate(r, float(T_val))

        c_heston = _heston_call_fwd_batch(
            F, K_vals, T_val, r_val, *params, phi_fine,
        )
        iv_h_arr = _b76_iv_vec(c_heston, F, K_vals, T_val, r_val, sigma0_arr=iv_mkt_vals)

        for i in range(len(K_vals)):
            detail_rows.append({
                "expiry_label": labels[i],
                "moneyness_pct": m_vals[i],
                "T": T_val,
                "iv_market": iv_mkt_vals[i],
                "iv_model": float(iv_h_arr[i]),
                "error_pp": (float(iv_h_arr[i]) - iv_mkt_vals[i]) * 100,
            })

    detail_df = pd.DataFrame(detail_rows)

    # Report quality on the calibrated region only (filtered by T and moneyness)
    cal_mask = (
        (detail_df["T"] >= min_T)
        & (detail_df["moneyness_pct"] >= m_range[0])
        & (detail_df["moneyness_pct"] <= m_range[1])
    )
    cal_errors = detail_df.loc[cal_mask, "error_pp"].values / 100

    return HestonCalibResult(
        params=params,
        rmse_iv=float(np.sqrt(np.mean(cal_errors ** 2))),
        max_err_iv=float(np.max(np.abs(cal_errors))),
        mean_err_iv=float(np.mean(np.abs(cal_errors))),
        n_points=int(cal_mask.sum()),
        success=result.success,
        detail_df=detail_df,
    )


# ---------------------------------------------------------------------------
# Monte Carlo under Heston
# ---------------------------------------------------------------------------

_MC_MIN_PATHS = 5_000
_MC_MAX_PATHS = 400_000   # Heston paths are cheap (no LV grid lookup) and
                          # digital / one-touch / far-OTM barrier payoffs
                          # have high variance — a larger cap is needed for
                          # the SE-based convergence rule to fire on those.
_MC_BATCH     = 500
_MC_EPS_FLOOR = 1e-8


def mc_heston(
    params: HestonParams,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    is_digital: bool = False,
    digital_payout: float = 1.0,
    barrier_type: str | None = None,
    barrier_level: float | None = None,
    barrier_style: str = "american",
    is_one_touch: bool = False,
    one_touch_direction: str | None = None,
    steps_per_year: int = 252,
    eps: float = 0.001,
    seed: int | None = None,
) -> MCResult:
    """
    Monte Carlo pricing under Heston dynamics.

    Discretisation: full-truncation Euler.
        v⁺ = max(v, 0) in both drift and diffusion — keeps S positive
        even when the Feller condition 2κθ > ξ² is violated.

    Supports vanilla / digital / barrier (call or put), and one-touch
    (pay-at-hit, FX market convention).  One-touch is mutually exclusive
    with the other modes; option_type and K are ignored when
    is_one_touch=True.

    Variance reduction — antithetic variates
    ----------------------------------------
    Each batch of `bs` Heston paths is drawn as `bs/2` independent
    `(Z₁, Z₂)` pre-rotation Gaussian trajectories plus their joint
    negations `(−Z₁, −Z₂)`.  Negating both pre-rotation drivers
    simultaneously preserves the correlation structure
    `(ρZ₁ + √(1−ρ²)Z₂)` while flipping the sign of the increments to
    both `dW^v` and `dW^S`.  The estimator stays unbiased; for vanilla
    payoffs the variance falls by ~30–50 % at no extra path cost.
    Smaller benefit on barrier / one-touch payoffs because hit
    indicators are not monotone in the driver, but never harmful.

    Discrete-monitoring bias
    ------------------------
    Barrier and one-touch hits are checked once per Euler step.  A
    Brownian path can cross a barrier *between* grid points and come
    back, so discrete monitoring systematically **under-detects hits**
    versus a continuously-monitored contract.  The bias scales as
    O(σ·√Δt) in barrier distance — knock-ins and one-touches are
    therefore under-priced and knock-outs over-priced, with the
    distortion largest for short-dated structures with the barrier close
    to spot.  No Brownian-bridge correction is applied; treat results
    near a tight barrier with caution and prefer the default
    `steps_per_year=252` over coarser settings.

    Drift uses a scalar `r` and `q`
    --------------------------------
    The simulation drift is `(r − q − ½v⁺)·dt` with `r` and `q`
    constants for the life of the path.  In production the caller passes
    `r = curve.zero_rate(T)` and `q = div_curve.div_yield(T)` — i.e. the
    *zero rates* read at the option's terminal maturity — so the
    discount factor `e^{−rT}` and the forward `S₀·e^{(r−q)T}` are
    consistent with the rate / dividend term structure at expiry.  But
    the *intermediate* forwards `S₀·e^{(r−q)t}` along the path are not
    — for a path-dependent payoff (barrier, one-touch) this is a
    standard simplification, accurate to first order in the slope of
    the forward curve.  If you need a fully term-structure-consistent
    drift, integrate `∫₀^t f(s) ds` from the instantaneous forward
    rates instead of using a single `r·t`.

    Convergence uses the same SE-based precision rule as the Dupire MC
    pricer (montecarlo.py): batches of 500 paths, minimum 5 000 before
    early exit, stop when `SE(n)/|P(n)| < eps`.  SE is computed from the
    variance of antithetic *pair averages* `Y_j = (X_a + X_b)/2`, which
    is the unbiased estimator of `Var(P̂)` — for monotone payoffs the
    pair correlation is strongly negative and the pair-aware SE is
    several × smaller than the marginal SE; for digital / OT payoffs
    the pair correlation is ≈ 0 and the two coincide (honestly
    reflecting that antithetic doesn't help on indicator-like payoffs).
    Hard cap 400 000 paths (higher than Dupire's 200 000 because Heston
    paths are cheaper per step — no grid interpolation — and digital /
    one-touch payoffs need the extra headroom under a true precision
    bound).
    """
    if is_one_touch:
        if is_digital:
            raise ValueError("is_one_touch is mutually exclusive with is_digital")
        if barrier_type is not None:
            raise ValueError("is_one_touch is mutually exclusive with barrier_type")
        if one_touch_direction not in ("up", "down"):
            raise ValueError(
                f"one_touch_direction must be 'up' or 'down' when is_one_touch=True, "
                f"got {one_touch_direction!r}"
            )
        if barrier_level is None or barrier_level <= 0:
            raise ValueError("barrier_level must be positive when is_one_touch=True")
        if digital_payout <= 0:
            raise ValueError("digital_payout (cash payout) must be positive for one-touch")

    v0, kappa, theta, xi, rho = params
    rng = np.random.default_rng(seed)

    N = max(1, int(T * steps_per_year))
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(1 - rho ** 2)
    disc = np.exp(-r * T)

    # Running accumulators (avoids rebuilding arrays each batch).
    # Pair-level sums power the antithetic-aware SE estimator — see
    # the docstring above for derivation.
    total_pair_sum    = 0.0
    total_pair_sq_sum = 0.0
    n_paths_done      = 0
    n_pairs_done      = 0
    converged         = False

    while n_paths_done < _MC_MAX_PATHS:
        bs = min(_MC_BATCH, _MC_MAX_PATHS - n_paths_done)

        # Antithetic variates: draw bs/2 independent (Z₁, Z₂) trajectories
        # and pair them with their joint negations.  Negating both drivers
        # simultaneously preserves the (ρ, √(1−ρ²)) rotation while flipping
        # the sign of every Brownian increment.  bs is always even
        # (_MC_BATCH = 500, _MC_MAX_PATHS a multiple of 500).
        half = bs // 2
        Z1_half = rng.standard_normal((half, N))
        Z2_half = rng.standard_normal((half, N))
        Z1 = np.concatenate([Z1_half, -Z1_half], axis=0)
        Z2 = np.concatenate([Z2_half, -Z2_half], axis=0)
        dW_v = Z1 * sqrt_dt
        dW_s = (rho * Z1 + sqrt_1_rho2 * Z2) * sqrt_dt

        S = np.full(bs, float(S0))
        v = np.full(bs, float(v0))
        hit = np.zeros(bs, dtype=bool)

        # One-touch hit-time tracking (per-path step index, -1 = never hit)
        if is_one_touch:
            first_hit_step = np.full(bs, -1, dtype=np.int32)
            if one_touch_direction == "up" and S0 >= barrier_level:
                first_hit_step[:] = 0
            elif one_touch_direction == "down" and S0 <= barrier_level:
                first_hit_step[:] = 0

        for i in range(N):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Log-Euler for S (exact drift correction)
            S *= np.exp(
                (r - q - 0.5 * v_pos) * dt + sqrt_v * dW_s[:, i]
            )
            # Full-truncation Euler for v
            v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * dW_v[:, i]

            # American barrier monitoring
            if barrier_type is not None and barrier_style == "american":
                if barrier_type in ("up_out", "up_in"):
                    hit |= S >= barrier_level
                else:
                    hit |= S <= barrier_level

            # One-touch monitoring (always American — only style supported).
            # Record first crossing step; subsequent crossings are ignored.
            if is_one_touch:
                if one_touch_direction == "up":
                    new_hit = (first_hit_step == -1) & (S >= barrier_level)
                else:
                    new_hit = (first_hit_step == -1) & (S <= barrier_level)
                first_hit_step[new_hit] = i + 1

        # European barrier (terminal only)
        if barrier_type is not None and barrier_style == "european":
            if barrier_type in ("up_out", "up_in"):
                hit = S >= barrier_level
            else:
                hit = S <= barrier_level

        # Payoff
        if is_one_touch:
            # Pay-at-hit: hit paths get digital_payout discounted from τ
            # back to today; un-hit paths get nothing.  Per-path discount
            # replaces the global `disc` here.
            hit_mask = first_hit_step >= 0
            tau      = first_hit_step.astype(float) * dt
            per_path_disc = np.where(hit_mask, np.exp(-r * tau), 0.0)
            payoffs = digital_payout * per_path_disc
        else:
            if option_type == "call":
                payoff = (
                    np.where(S > K, digital_payout, 0.0)
                    if is_digital
                    else np.maximum(S - K, 0.0)
                )
            else:
                payoff = (
                    np.where(S < K, digital_payout, 0.0)
                    if is_digital
                    else np.maximum(K - S, 0.0)
                )

            # Barrier knock-in / knock-out
            if barrier_type is not None:
                if barrier_type in ("up_out", "down_out"):
                    payoff = np.where(hit, 0.0, payoff)
                else:
                    payoff = np.where(hit, payoff, 0.0)

            payoffs = payoff * disc

        # Antithetic pair averages — paths [0..bs/2-1] are paired with
        # [bs/2..bs-1] (driven by joint (Z₁, Z₂) and (-Z₁, -Z₂)).
        half_bs = bs // 2
        pair_avg = 0.5 * (payoffs[:half_bs] + payoffs[half_bs:])

        total_pair_sum    += float(pair_avg.sum())
        total_pair_sq_sum += float((pair_avg ** 2).sum())
        n_paths_done      += bs
        n_pairs_done      += half_bs

        # Convergence check: relative standard error below `eps`.
        # SE from the variance of antithetic pair averages — for vanilla
        # Heston payoffs the pair correlation is strongly negative and
        # this is several × tighter than the marginal SE; for digital /
        # OT payoffs the two coincide.
        if n_paths_done >= _MC_MIN_PATHS:
            P_n      = total_pair_sum / n_pairs_done
            var_pair = max(total_pair_sq_sum / n_pairs_done - P_n ** 2, 0.0)
            se_n     = (var_pair / n_pairs_done) ** 0.5
            denom    = max(abs(P_n), _MC_EPS_FLOOR)
            if se_n / denom < eps:
                converged = True
                break

    price    = total_pair_sum / n_pairs_done
    var_pair = max(total_pair_sq_sum / n_pairs_done - price ** 2, 0.0)
    std_err  = (var_pair / n_pairs_done) ** 0.5

    return MCResult(
        price=float(price),
        std_error=float(std_err),
        n_paths=n_paths_done,
        converged=converged,
        clamp_pct=0.0,
    )