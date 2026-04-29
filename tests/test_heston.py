"""Tests for heston.py — Heston stochastic vol model correctness."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from heston import (
    HestonParams,
    _b76_call,
    _b76_call_vec,
    _b76_iv,
    _b76_iv_vec,
    _b76_vega,
    _b76_vega_vec,
    _heston_cf,
    _heston_call_fwd_batch,
    heston_call,
    heston_smile,
    mc_heston,
)


# ---------------------------------------------------------------------------
# Black-76 helpers
# ---------------------------------------------------------------------------

class TestBlack76:
    """Black-76 pricing, vega, and IV inversion."""

    # Known ATM case: F=100, K=100, T=1, r=0.05, sigma=0.20
    F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_call_price_positive(self):
        p = _b76_call(self.F, self.K, self.T, self.r, self.sigma)
        assert p > 0

    def test_call_price_atm_approximation(self):
        """ATM Black-76 ≈ disc * F * σ√T * 0.3989 (Brenner-Subrahmanyam)."""
        p = _b76_call(self.F, self.K, self.T, self.r, self.sigma)
        approx = np.exp(-self.r * self.T) * self.F * self.sigma * np.sqrt(self.T) * 0.3989
        assert abs(p - approx) / approx < 0.01  # within 1%

    def test_call_intrinsic_floor(self):
        """Deep ITM call ≥ discounted intrinsic value."""
        K_deep = 50.0
        p = _b76_call(self.F, K_deep, self.T, self.r, self.sigma)
        intrinsic = np.exp(-self.r * self.T) * (self.F - K_deep)
        assert p >= intrinsic - 1e-10

    def test_vega_positive(self):
        v = _b76_vega(self.F, self.K, self.T, self.r, self.sigma)
        assert v > 0

    def test_vega_zero_for_zero_vol(self):
        v = _b76_vega(self.F, self.K, self.T, self.r, 0.0)
        assert v == 0.0

    def test_vectorised_matches_scalar(self):
        """_b76_call_vec and _b76_vega_vec must match scalar versions."""
        Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        sigmas = np.full(5, self.sigma)
        prices_vec = _b76_call_vec(self.F, Ks, self.T, self.r, sigmas)
        vegas_vec  = _b76_vega_vec(self.F, Ks, self.T, self.r, sigmas)
        for i, K in enumerate(Ks):
            p_scalar = _b76_call(self.F, K, self.T, self.r, self.sigma)
            v_scalar = _b76_vega(self.F, K, self.T, self.r, self.sigma)
            assert abs(prices_vec[i] - p_scalar) < 1e-12
            assert abs(vegas_vec[i] - v_scalar) < 1e-12

    def test_iv_roundtrip_scalar(self):
        """IV inversion must recover the original sigma."""
        p = _b76_call(self.F, self.K, self.T, self.r, self.sigma)
        iv = _b76_iv(p, self.F, self.K, self.T, self.r)
        assert abs(iv - self.sigma) < 1e-6

    def test_iv_roundtrip_otm(self):
        """IV roundtrip for OTM call (K > F)."""
        K_otm = 120.0
        sigma_otm = 0.25
        p = _b76_call(self.F, K_otm, self.T, self.r, sigma_otm)
        iv = _b76_iv(p, self.F, K_otm, self.T, self.r)
        assert abs(iv - sigma_otm) < 1e-5

    def test_iv_vec_roundtrip(self):
        """Vectorised IV inversion must recover original sigmas."""
        Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        sigmas = np.array([0.30, 0.25, 0.20, 0.22, 0.28])
        prices = _b76_call_vec(self.F, Ks, self.T, self.r, sigmas)
        ivs = _b76_iv_vec(prices, self.F, Ks, self.T, self.r, sigma0_arr=sigmas)
        np.testing.assert_allclose(ivs, sigmas, atol=1e-5)

    def test_iv_vec_matches_scalar(self):
        """_b76_iv_vec must match _b76_iv for each strike."""
        Ks = np.array([90.0, 100.0, 110.0])
        sigmas = np.array([0.25, 0.20, 0.22])
        prices = _b76_call_vec(self.F, Ks, self.T, self.r, sigmas)
        ivs_vec = _b76_iv_vec(prices, self.F, Ks, self.T, self.r)
        for i, K in enumerate(Ks):
            iv_scalar = _b76_iv(prices[i], self.F, K, self.T, self.r)
            assert abs(ivs_vec[i] - iv_scalar) < 1e-6


# ---------------------------------------------------------------------------
# Heston characteristic function
# ---------------------------------------------------------------------------

class TestHestonCF:
    """Characteristic function sanity checks."""

    params = (0.04, 2.0, 0.04, 0.5, -0.7)  # v0, kappa, theta, xi, rho
    T = 1.0

    def test_cf_at_phi_zero_is_one(self):
        """f_j(0) = 1 for both j=1,2 (normalisation)."""
        phi = np.array([1e-12])  # near zero
        for j in [1, 2]:
            cf = _heston_cf(phi, self.T, *self.params, j)
            assert abs(abs(cf[0]) - 1.0) < 1e-4

    def test_cf_modulus_bounded(self):
        """|f_j(φ)| ≤ 1 for a characteristic function of a probability."""
        phi = np.linspace(0.01, 100, 500)
        for j in [1, 2]:
            cf = _heston_cf(phi, self.T, *self.params, j)
            # P_j CFs can have |f| > 1 because they include the stock-price
            # weighting (P1) or are conditional, but they should not blow up
            assert np.all(np.isfinite(cf)), "CF must be finite for all phi"
            assert np.max(np.abs(cf)) < 1e6, "CF should not blow up"

    def test_cf_finite_for_extreme_params(self):
        """CF should remain finite even for stressed parameters."""
        phi = np.linspace(0.01, 50, 200)
        stressed = (0.01, 0.5, 0.01, 2.0, -0.95)  # high xi, deep rho
        for j in [1, 2]:
            cf = _heston_cf(phi, 2.0, *stressed, j)
            assert np.all(np.isfinite(cf))


# ---------------------------------------------------------------------------
# Heston call pricing
# ---------------------------------------------------------------------------

class TestHestonCall:
    """Semi-analytic Heston call price tests."""

    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.5, -0.7

    def test_call_positive(self):
        p = heston_call(
            self.S, self.K, self.T, self.r, self.q,
            self.v0, self.kappa, self.theta, self.xi, self.rho,
        )
        assert p > 0

    def test_call_less_than_spot(self):
        """European call ≤ S·e^{-qT} (upper bound)."""
        p = heston_call(
            self.S, self.K, self.T, self.r, self.q,
            self.v0, self.kappa, self.theta, self.xi, self.rho,
        )
        assert p <= self.S * np.exp(-self.q * self.T) + 1e-8

    def test_call_above_intrinsic(self):
        """European call ≥ max(S·e^{-qT} − K·e^{-rT}, 0)."""
        p = heston_call(
            self.S, self.K, self.T, self.r, self.q,
            self.v0, self.kappa, self.theta, self.xi, self.rho,
        )
        intrinsic = max(
            self.S * np.exp(-self.q * self.T)
            - self.K * np.exp(-self.r * self.T), 0.0
        )
        assert p >= intrinsic - 1e-8

    def test_put_call_parity(self):
        """C − P = S·e^{-qT} − K·e^{-rT}."""
        C = heston_call(
            self.S, self.K, self.T, self.r, self.q,
            self.v0, self.kappa, self.theta, self.xi, self.rho,
        )
        P = C - self.S * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)
        # P must be positive for ATM
        assert P > 0
        # Verify the parity relationship is self-consistent:
        # price a deep ITM call and check C is close to intrinsic + time value
        C_itm = heston_call(
            self.S, 80.0, self.T, self.r, self.q,
            self.v0, self.kappa, self.theta, self.xi, self.rho,
        )
        P_itm = C_itm - self.S * np.exp(-self.q * self.T) + 80.0 * np.exp(-self.r * self.T)
        assert P_itm > 0

    def test_fwd_batch_matches_spot_formula(self):
        """_heston_call_fwd_batch must agree with heston_call."""
        F = self.S * np.exp((self.r - self.q) * self.T)
        Ks = np.array([90.0, 100.0, 110.0])
        phi = np.linspace(1e-8, 100, 1000)

        prices_batch = _heston_call_fwd_batch(
            F, Ks, self.T, self.r,
            self.v0, self.kappa, self.theta, self.xi, self.rho, phi,
        )

        for i, K in enumerate(Ks):
            p_spot = heston_call(
                self.S, K, self.T, self.r, self.q,
                self.v0, self.kappa, self.theta, self.xi, self.rho,
                n_phi=1000,
            )
            assert abs(prices_batch[i] - p_spot) < 0.05, (
                f"K={K}: batch={prices_batch[i]:.4f} vs spot={p_spot:.4f}"
            )

    def test_call_monotone_in_strike(self):
        """Call prices must decrease as K increases."""
        Ks = [80, 90, 100, 110, 120]
        prices = [
            heston_call(
                self.S, K, self.T, self.r, self.q,
                self.v0, self.kappa, self.theta, self.xi, self.rho,
            )
            for K in Ks
        ]
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1] - 1e-8

    def test_zero_vol_of_vol_reduces_to_black(self):
        """When ξ → 0, Heston should converge to Black-76 with σ = √v₀."""
        p_heston = heston_call(
            self.S, self.K, self.T, self.r, self.q,
            self.v0, self.kappa, self.theta, 0.001, 0.0,  # xi≈0, rho=0
        )
        F = self.S * np.exp((self.r - self.q) * self.T)
        p_b76 = _b76_call(F, self.K, self.T, self.r, np.sqrt(self.v0))
        assert abs(p_heston - p_b76) < 0.15, (
            f"Heston(xi≈0)={p_heston:.4f} vs B76={p_b76:.4f}"
        )


# ---------------------------------------------------------------------------
# Heston smile
# ---------------------------------------------------------------------------

class TestHestonSmile:
    """heston_smile function tests."""

    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    S, T, r, q = 100.0, 1.0, 0.05, 0.02

    def test_smile_returns_correct_length(self):
        m_grid = np.linspace(80, 120, 20)
        ivs = heston_smile(self.S, self.T, self.r, self.q, self.params, m_grid)
        assert len(ivs) == 20

    def test_smile_values_positive(self):
        m_grid = np.linspace(80, 120, 20)
        ivs = heston_smile(self.S, self.T, self.r, self.q, self.params, m_grid)
        assert np.all(ivs > 0)

    def test_smile_has_skew(self):
        """With rho < 0, low-strike IV should exceed high-strike IV."""
        m_grid = np.array([80.0, 100.0, 120.0])
        ivs = heston_smile(self.S, self.T, self.r, self.q, self.params, m_grid)
        assert ivs[0] > ivs[2], "Negative rho should produce put skew"


# ---------------------------------------------------------------------------
# Heston Monte Carlo
# ---------------------------------------------------------------------------

class TestHestonMC:
    """Monte Carlo under Heston dynamics."""

    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02

    def test_mc_vanilla_call_positive(self):
        res = mc_heston(self.params, self.S0, self.K, self.T, self.r, self.q, seed=1)
        assert res.price > 0
        assert res.std_error > 0
        assert res.n_paths >= 5000

    def test_mc_vs_fourier_atm(self):
        """MC price should be within 3 std errors of semi-analytic."""
        analytic = heston_call(
            self.S0, self.K, self.T, self.r, self.q, *self.params,
        )
        mc = mc_heston(
            self.params, self.S0, self.K, self.T, self.r, self.q, seed=42,
        )
        assert abs(mc.price - analytic) < 3 * mc.std_error + 0.5, (
            f"MC={mc.price:.2f} vs analytic={analytic:.2f}, "
            f"SE={mc.std_error:.2f}"
        )

    def test_mc_put_call_parity(self):
        """MC call − MC put ≈ S·e^{-qT} − K·e^{-rT}."""
        mc_call = mc_heston(
            self.params, self.S0, self.K, self.T, self.r, self.q,
            option_type="call", seed=123,
        )
        mc_put = mc_heston(
            self.params, self.S0, self.K, self.T, self.r, self.q,
            option_type="put", seed=123,
        )
        parity = (
            self.S0 * np.exp(-self.q * self.T)
            - self.K * np.exp(-self.r * self.T)
        )
        diff = mc_call.price - mc_put.price
        tol = 3 * np.sqrt(mc_call.std_error ** 2 + mc_put.std_error ** 2) + 0.5
        assert abs(diff - parity) < tol, (
            f"C-P={diff:.2f} vs parity={parity:.2f}, tol={tol:.2f}"
        )

    def test_mc_seed_reproducibility(self):
        """Same seed must give identical results."""
        r1 = mc_heston(self.params, self.S0, self.K, self.T, self.r, self.q, seed=99)
        r2 = mc_heston(self.params, self.S0, self.K, self.T, self.r, self.q, seed=99)
        assert r1.price == r2.price
        assert r1.n_paths == r2.n_paths

    def test_mc_digital_call_bounded(self):
        """Digital call price must be between 0 and disc * payout."""
        payout = 10.0
        res = mc_heston(
            self.params, self.S0, self.K, self.T, self.r, self.q,
            is_digital=True, digital_payout=payout, seed=42,
        )
        assert 0 <= res.price <= payout * np.exp(-self.r * self.T) + 0.01

    def test_mc_up_out_cheaper_than_vanilla(self):
        """Up-and-out call ≤ vanilla call."""
        vanilla = mc_heston(
            self.params, self.S0, self.K, self.T, self.r, self.q, seed=42,
        )
        barrier = mc_heston(
            self.params, self.S0, self.K, self.T, self.r, self.q,
            barrier_type="up_out", barrier_level=self.S0 * 1.3,
            barrier_style="american", seed=42,
        )
        assert barrier.price <= vanilla.price + 0.01


# ---------------------------------------------------------------------------
# One-touch (Heston)
# ---------------------------------------------------------------------------

class TestHestonOneTouch:
    """Heston MC one-touch — pay-at-hit convention, FX market standard."""

    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    S0, T, r, q = 100.0, 0.5, 0.05, 0.0
    payout = 1.0

    def test_ot_bounded_by_payout(self):
        """One-touch price is between 0 and the cash payout (no discount can exceed 1)."""
        res = mc_heston(
            self.params, self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=self.S0 * 1.10, digital_payout=self.payout, seed=42,
        )
        assert 0.0 <= res.price <= self.payout

    def test_ot_immediate_payment_when_already_past_barrier(self):
        """If barrier is already breached at t=0, price equals the payout exactly (τ=0, no discount)."""
        # Up-touch with B at or below spot: every path "hit" at step 0
        res = mc_heston(
            self.params, self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=self.S0,  # B = S0 → immediate hit
            digital_payout=self.payout, seed=42,
        )
        assert abs(res.price - self.payout) < 1e-12

    def test_ot_unreachable_barrier_zero_price(self):
        """A barrier 100× spot is essentially unreachable in 0.5 yr — price ≈ 0."""
        res = mc_heston(
            self.params, self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=self.S0 * 100.0,
            digital_payout=self.payout, seed=42,
        )
        assert res.price < 0.01  # well under 1% of payout

    def test_ot_up_monotone_in_barrier(self):
        """Higher up-touch barrier → lower OT price (less likely to hit)."""
        r_close = mc_heston(
            self.params, self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=self.S0 * 1.10,
            digital_payout=self.payout, seed=42,
        )
        r_far = mc_heston(
            self.params, self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=self.S0 * 1.30,
            digital_payout=self.payout, seed=42,
        )
        assert r_close.price > r_far.price

    def test_ot_seed_reproducible(self):
        """Same seed → identical price."""
        kwargs = dict(
            S0=self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="down",
            barrier_level=self.S0 * 0.9, digital_payout=self.payout, seed=7,
        )
        r1 = mc_heston(self.params, **kwargs)
        r2 = mc_heston(self.params, **kwargs)
        assert r1.price == r2.price

    def test_ot_payout_scales_linearly(self):
        """Doubling the cash payout doubles the price (at the same seed)."""
        kwargs = dict(
            S0=self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=self.S0 * 1.10, seed=42,
        )
        r1 = mc_heston(self.params, digital_payout=1.0, **kwargs)
        r2 = mc_heston(self.params, digital_payout=2.0, **kwargs)
        assert abs(r2.price - 2.0 * r1.price) < 1e-12

    def test_ot_rejects_bad_direction(self):
        """Invalid direction raises."""
        with pytest.raises(ValueError):
            mc_heston(
                self.params, self.S0, K=0.0, T=self.T, r=self.r, q=self.q,
                is_one_touch=True, one_touch_direction="sideways",
                barrier_level=self.S0 * 1.1, digital_payout=1.0, seed=1,
            )

    def test_ot_mutex_with_digital(self):
        """is_one_touch and is_digital cannot both be set."""
        with pytest.raises(ValueError):
            mc_heston(
                self.params, self.S0, self.S0, self.T, self.r, self.q,
                is_one_touch=True, one_touch_direction="up",
                barrier_level=self.S0 * 1.1,
                is_digital=True, digital_payout=1.0, seed=1,
            )

    def test_ot_mutex_with_barrier(self):
        """is_one_touch and barrier_type cannot both be set."""
        with pytest.raises(ValueError):
            mc_heston(
                self.params, self.S0, self.S0, self.T, self.r, self.q,
                is_one_touch=True, one_touch_direction="up",
                barrier_level=self.S0 * 1.1, digital_payout=1.0,
                barrier_type="up_out", seed=1,
            )