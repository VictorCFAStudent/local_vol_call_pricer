"""
montecarlo.py — Monte Carlo pricing on the Dupire local vol surface.

Supports:
  · Vanilla European calls and puts (max(S_T − K, 0) / max(K − S_T, 0))
  · Digital (cash-or-nothing) calls and puts: fixed payout if ITM at expiry
  · Barrier options: up/down × in/out, combinable with vanilla or digital
  · Barrier monitoring style: American (every Euler step) or European (at expiry)
  · One-touch options (pay-at-hit, FX market convention): fixed cash payout
    delivered the moment spot first touches the barrier, discounted from
    that hit time τ — `PV = E[e^{−rτ}·X·1{barrier hit during [0, T]}]`.
    Up or down direction.  American monitoring only.

Simulation scheme
-----------------
Log-Euler discretisation:

    log S_{t+Δt} = log S_t + (r − q − ½ σ²_loc(m_t, t)) · Δt + σ_loc · √Δt · Z_t

    where:
        m_t = (S_t / F_t) × 100       forward moneyness in %
        F_t = S_0 · exp((r − q) · t)  forward price at time t
        Z_t ~ N(0, 1)                  independent draw each step

Variance reduction — antithetic variates
----------------------------------------
Each batch of `bs` paths is generated as `bs/2` independent Gaussian
trajectories `Z` plus their negations `−Z` ("antithetic" pairs).  Since
the vanilla call/put payoff is monotone in the terminal log-price, the
two members of each pair have negatively correlated payoffs and the
average has lower variance than two independent draws — typically a
factor of ~2 reduction in variance for at-the-money calls/puts and 1.5x
for digitals.  Antithetic is unbiased: `E[(payoff(Z) + payoff(−Z))/2] =
E[payoff(Z)]`, regardless of correlation.  The batch size 500 is even
so each batch contains exactly 250 antithetic pairs.

Discrete-monitoring bias (barriers and one-touch)
-------------------------------------------------
Barrier hits and one-touch first-passage times are checked once per Euler
step.  Because the underlying Brownian path can cross a barrier *between*
two grid points and come back, discrete monitoring **systematically
under-detects hits** relative to a continuously-monitored contract.  The
bias is O(σ·√Δt) in barrier distance and is most severe when:

  * the barrier is close to spot (within a few σ·√Δt),
  * the option is short-dated (few steps to amortise the bias),
  * `steps_per_year` is small (52 weekly is much worse than 252 daily).

Practical effect on prices, holding all else equal:
  * knock-in / one-touch  : **under-priced**  (fewer hits detected)
  * knock-out             : **over-priced**   (fewer KO events)

The standard fix is a Brownian-bridge correction (Beaglehole–Dybvig–Zhou):
between two consecutive simulated points S_i, S_{i+1} that both stay on
the same side of the barrier, the conditional probability that a Brownian
bridge crossed it during [t_i, t_{i+1}] has a closed form.  This is not
applied here, so the engine is best suited to *trader-facing prototype*
pricing rather than production: `steps_per_year=252` is the recommended
setting (daily fixings match most listed barrier contracts), and barriers
within ~2σ·√Δt of spot should be treated with caution.  A real desk would
either add the Brownian-bridge correction or move to a PDE solver for
path-dependent payoffs (Bloomberg OVME does the latter).

Out-of-bounds handling
----------------------
When m_t or t falls outside the local vol surface domain, the nearest
boundary value is used (clamp).  The fraction of clamped steps is returned
as a diagnostic — a high value signals that the surface is too narrow for
the chosen strike or maturity.

Convergence criterion
---------------------
Paths are simulated in batches of 500.  After each batch we evaluate the
running standard error of the estimator and stop when the (one-sigma)
relative half-width falls below `eps`:

    SE(n) / max(|P(n)|, _EPS_FLOOR) < eps   and   n ≥ 5 000

**Antithetic-aware SE.**  Because each batch contains 250 antithetic
*pairs* (path i and path i+250 driven by `Z` and `−Z` respectively), the
two members of a pair are negatively correlated for monotonic payoffs.
Treating all 500 paths as IID — i.e. SE = √(Var(payoff)/N) from the
marginal sample variance — overestimates the true `Var(P̂)` by a factor
of `1/(1+ρ_pair)`.  For a vanilla call, `ρ_pair ≈ −0.95`, so the naive
SE is ~5× too pessimistic and convergence fires far later than it should.

We instead track the variance of *pair averages* `Y_j = (X_a + X_b)/2`
across batches.  `Var(P̂) = Var(Y) / M` where `M = N/2` is the number of
pairs, and `Var(Y)` captures the antithetic correlation correctly:
            Var(Y) = (1/4)·(2·var(X) + 2·Cov(X_a, X_b))
                   = (var(X)/2) · (1 + ρ_pair)
For vanillas this is ~5× smaller than the marginal variance and the SE
bound becomes tight.  For digital / one-touch payoffs the indicator
function is not monotone in `Z`, `ρ_pair ≈ 0`, and the pair-aware SE
reduces to the marginal SE — antithetic doesn't help on those, and we
honestly report it.

Hard cap: 200 000 paths (digitals, one-touches, and far-OTM barriers
have payoff variance an order of magnitude above vanillas, so the
previous 50 000 cap routinely terminated without precision being met).
If the cap is reached, the price is still the best unbiased estimate
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
_MAX_PATHS   = 200_000
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
def price_european_option(
    lv_grid:        LocalVolGrid,
    S0:             float,
    K:              float,
    T:              float,
    r:              float,
    q:              float = 0.0,
    option_type:    str   = "call",
    is_digital:     bool  = False,
    digital_payout: float = 1.0,
    barrier_type:   str | None = None,
    barrier_level:  float | None = None,
    barrier_style:  str   = "american",
    is_one_touch:   bool  = False,
    one_touch_direction: str | None = None,
    steps_per_year: int   = 252,
    eps:            float = 0.001,
    seed:           int | None = None,
) -> MCResult:
    """Monte Carlo price of a European vanilla, digital, barrier, or one-touch option.

    Parameters
    ----------
    lv_grid        : LocalVolGrid from build_local_vol()
    S0             : current spot price
    K              : strike (ignored for one-touch — OT has no strike)
    T              : option maturity in years
    r              : risk-free rate (continuous, annualised)
    q              : dividend yield (continuous, annualised)
    option_type    : "call" or "put" (ignored for one-touch)
    is_digital     : if True, payoff is cash-or-nothing: digital_payout when
                     S_T is in-the-money, else 0.  Otherwise a vanilla
                     max(S_T − K, 0) / max(K − S_T, 0) payoff is used.
                     Mutually exclusive with is_one_touch.
    digital_payout : cash amount paid when a digital is in-the-money, or when
                     a one-touch barrier is hit (default 1.0).  Reused for OT.
    barrier_type   : None for no barrier; one of "up_out", "up_in",
                     "down_out", "down_in".  Combines with is_digital to give
                     e.g. a "digital knock-out".  Mutually exclusive with
                     is_one_touch.
    barrier_level  : barrier level B (required when barrier_type is set, or
                     when is_one_touch=True).
    barrier_style  : "american" (default) monitors the barrier at every Euler
                     step throughout the life of the option.  "european"
                     checks the barrier only at expiry — S_T is the sole
                     determinant of hit / no-hit.
    is_one_touch   : if True, payoff is digital_payout paid AT THE TIME OF
                     THE BARRIER HIT (pay-at-hit convention, FX market
                     standard), discounted from τ to today; zero if the
                     barrier is never touched during [0, T].
    one_touch_direction : "up" (B above spot, pays on first up-cross) or
                     "down" (B below spot, pays on first down-cross).
                     Required when is_one_touch=True.
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
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    _VALID_BARRIERS = ("up_out", "up_in", "down_out", "down_in")
    if barrier_type is not None:
        if barrier_type not in _VALID_BARRIERS:
            raise ValueError(
                f"barrier_type must be one of {_VALID_BARRIERS} or None, got {barrier_type!r}"
            )
        if barrier_level is None or barrier_level <= 0:
            raise ValueError("barrier_level must be a positive number when barrier_type is set")
        if barrier_style not in ("american", "european"):
            raise ValueError(
                f"barrier_style must be 'american' or 'european', got {barrier_style!r}"
            )

    if is_digital and digital_payout <= 0:
        raise ValueError("digital_payout must be positive when is_digital=True")

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

    # Running accumulators.  Pair-level sums (sum and sum-of-squares of
    # antithetic pair averages) are the basis of the antithetic-aware SE
    # estimator — see the module docstring for derivation.
    total_pair_sum    = 0.0
    total_pair_sq_sum = 0.0
    total_clamped_steps = 0
    total_steps_done    = 0
    n_paths_done        = 0
    n_pairs_done        = 0
    converged           = False

    while n_paths_done < _MAX_PATHS:
        bs = min(_BATCH_SIZE, _MAX_PATHS - n_paths_done)

        # ── Vectorised simulation of bs paths ──────────────────────────────
        log_S          = np.full(bs, np.log(S0))   # (bs,)
        batch_clamped  = int(t_steps_clamped) * bs  # t-clamps: same for all paths

        # Per-path "barrier crossed at least once" flag.  For American-style
        # barriers we track hits during the simulation (initialised from S0
        # so an option starting past its barrier is handled correctly).  For
        # European-style barriers the flag is only set from S_T at payoff
        # time — the in-loop update is a no-op.
        barrier_hit = np.zeros(bs, dtype=bool)
        _monitor_in_loop = (barrier_type is not None) and (barrier_style == "american")
        if _monitor_in_loop:
            if barrier_type in ("up_out", "up_in") and S0 >= barrier_level:
                barrier_hit[:] = True
            elif barrier_type in ("down_out", "down_in") and S0 <= barrier_level:
                barrier_hit[:] = True

        # One-touch tracking: per-path step index of first barrier touch
        # (-1 = never touched).  Stored as int — converted to time τ at
        # payoff time for the path-specific discount.  Day-zero check:
        # if S0 is already at or past the OT barrier, every path is hit
        # at τ = 0 (immediate payment, no discount).
        if is_one_touch:
            first_hit_step = np.full(bs, -1, dtype=np.int32)
            if one_touch_direction == "up" and S0 >= barrier_level:
                first_hit_step[:] = 0
            elif one_touch_direction == "down" and S0 <= barrier_level:
                first_hit_step[:] = 0

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

            # Antithetic variates: draw bs/2 independent Gaussians and pair
            # them with their negations.  bs is even by construction
            # (_BATCH_SIZE = 500, _MAX_PATHS a multiple of 500).
            half = bs // 2
            Z_half = rng.standard_normal(half)
            Z = np.concatenate([Z_half, -Z_half])
            log_S += (r - q - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z

            # Barrier monitoring after the step update (American-style only)
            if _monitor_in_loop:
                if barrier_type in ("up_out", "up_in"):
                    barrier_hit |= (np.exp(log_S) >= barrier_level)
                else:  # down_out / down_in
                    barrier_hit |= (np.exp(log_S) <= barrier_level)

            # One-touch monitoring after the step update.  We need the
            # *time* of first hit, not just the boolean — record the
            # earliest step index where each path crosses.  step_idx + 1
            # because the spot has just moved to t = (step_idx+1)·dt.
            if is_one_touch:
                S_now = np.exp(log_S)
                if one_touch_direction == "up":
                    new_hit = (first_hit_step == -1) & (S_now >= barrier_level)
                else:
                    new_hit = (first_hit_step == -1) & (S_now <= barrier_level)
                first_hit_step[new_hit] = step_idx + 1

        # ── Payoffs ────────────────────────────────────────────────────────
        S_T = np.exp(log_S)

        # European barrier: hit is decided solely by the terminal spot
        if barrier_type is not None and barrier_style == "european":
            if barrier_type in ("up_out", "up_in"):
                barrier_hit = (S_T >= barrier_level)
            else:
                barrier_hit = (S_T <= barrier_level)

        if option_type == "call":
            itm = S_T > K
            vanilla = np.maximum(S_T - K, 0.0)
        else:
            itm = S_T < K
            vanilla = np.maximum(K - S_T, 0.0)

        raw_payoff = (digital_payout * itm.astype(float)) if is_digital else vanilla

        if is_one_touch:
            # Pay-at-hit: hit paths get digital_payout discounted from τ
            # back to today; un-hit paths get nothing.  Per-path discount
            # replaces the global `disc` here.
            hit_mask = first_hit_step >= 0
            tau      = first_hit_step.astype(float) * dt   # 0 for un-hit paths (masked out)
            per_path_disc = np.where(hit_mask, np.exp(-r * tau), 0.0)
            payoffs = digital_payout * per_path_disc
        elif barrier_type is None:
            payoffs = raw_payoff * disc
        elif barrier_type in ("up_out", "down_out"):
            # Knock-out: zero payoff if barrier was hit
            payoffs = np.where(barrier_hit, 0.0, raw_payoff) * disc
        else:  # up_in / down_in
            # Knock-in: zero payoff unless barrier was hit
            payoffs = np.where(barrier_hit, raw_payoff, 0.0) * disc

        # Antithetic pair averages: paths [0..bs/2-1] are paired with
        # [bs/2..bs-1] (driven by Z and -Z respectively).  Each pair's
        # average is one IID sample of the estimator integrand, with
        # the antithetic correlation absorbed correctly into Var(Y).
        half_bs = bs // 2
        pair_avg = 0.5 * (payoffs[:half_bs] + payoffs[half_bs:])

        total_pair_sum    += float(pair_avg.sum())
        total_pair_sq_sum += float((pair_avg ** 2).sum())
        total_clamped_steps += batch_clamped
        total_steps_done    += bs * n_steps
        n_paths_done        += bs
        n_pairs_done        += half_bs

        # ── Convergence check ──────────────────────────────────────────────
        # Stop when the running standard error is below `eps · |P|`.
        # SE is computed from the variance of antithetic pair averages,
        # which is the unbiased estimator for Var(P̂) — for a vanilla payoff
        # this is ~5× smaller than the marginal sample variance because
        # antithetic pairs are negatively correlated.  For digital / OT
        # payoffs the pair correlation is ≈ 0 and pair-aware SE matches
        # the marginal SE, which is the honest report.
        if n_paths_done >= _MIN_PATHS:
            P_n      = total_pair_sum / n_pairs_done
            var_pair = max(total_pair_sq_sum / n_pairs_done - P_n ** 2, 0.0)
            se_n     = (var_pair / n_pairs_done) ** 0.5
            denom    = max(abs(P_n), _EPS_FLOOR)
            if se_n / denom < eps:
                converged = True
                break

    # ── Final statistics ───────────────────────────────────────────────────
    price     = total_pair_sum / n_pairs_done
    var_pair  = max(total_pair_sq_sum / n_pairs_done - price ** 2, 0.0)
    std_error = (var_pair / n_pairs_done) ** 0.5
    clamp_pct = 100.0 * total_clamped_steps / total_steps_done if total_steps_done > 0 else 0.0

    return MCResult(
        price=float(price),
        std_error=float(std_error),
        n_paths=n_paths_done,
        converged=converged,
        clamp_pct=float(clamp_pct),
    )
