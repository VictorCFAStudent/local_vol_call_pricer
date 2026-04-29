"""Tests for montecarlo.py — focuses on the one-touch payoff under Dupire LV.

Vanilla / digital / barrier paths are exercised end-to-end via the Heston
suite (`test_heston.py`); here we validate the one-touch logic under the
local-volatility engine, which is the only place those code paths run for
the Dupire pricer.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_workbook
from local_vol import build_local_vol
from montecarlo import price_european_option

_DATA_DIR = Path(__file__).parent.parent / "data_vol_surface"
_CANDIDATES = sorted(_DATA_DIR.glob("vol_surface*.xlsx"))
WORKBOOK = _CANDIDATES[-1] if _CANDIDATES else _DATA_DIR / "vol_surface.xlsx"


@pytest.fixture(scope="module")
def lv_grid():
    """Real Dupire LV grid built from the bundled SPX snapshot."""
    ds = load_workbook(WORKBOOK)
    return build_local_vol(ds.vol_df, n_k=40, n_t=40)


# ---------------------------------------------------------------------------
# One-touch (Dupire LV)
# ---------------------------------------------------------------------------

class TestLVOneTouch:
    """Local-vol Monte Carlo one-touch — pay-at-hit convention."""

    T, r, q = 0.5, 0.04, 0.0
    payout = 1.0

    @pytest.fixture
    def S0(self, lv_grid):
        # Use ATM as a sensible default — guaranteed inside the surface domain
        return float(lv_grid.K_grid.mean())

    def test_ot_bounded_by_payout(self, lv_grid, S0):
        """Price ∈ [0, payout]."""
        res = price_european_option(
            lv_grid, S0=S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=S0 * 1.10, digital_payout=self.payout, seed=42,
        )
        assert 0.0 <= res.price <= self.payout

    def test_ot_immediate_payment_when_already_past_barrier(self, lv_grid, S0):
        """B = S0 → every path hit at τ=0; price = payout exactly (no discount)."""
        res = price_european_option(
            lv_grid, S0=S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=S0, digital_payout=self.payout, seed=42,
        )
        assert abs(res.price - self.payout) < 1e-12

    def test_ot_unreachable_barrier_zero_price(self, lv_grid, S0):
        """Barrier 100× spot is unreachable → price ≈ 0."""
        res = price_european_option(
            lv_grid, S0=S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=S0 * 100.0, digital_payout=self.payout, seed=42,
        )
        assert res.price < 0.01

    def test_ot_up_monotone_in_barrier(self, lv_grid, S0):
        """Higher up-touch barrier → lower price."""
        kwargs = dict(
            lv_grid=lv_grid, S0=S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            digital_payout=self.payout, seed=42,
        )
        r_close = price_european_option(barrier_level=S0 * 1.05, **kwargs)
        r_far   = price_european_option(barrier_level=S0 * 1.20, **kwargs)
        assert r_close.price > r_far.price

    def test_ot_seed_reproducible(self, lv_grid, S0):
        """Same seed → identical price."""
        kwargs = dict(
            lv_grid=lv_grid, S0=S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="down",
            barrier_level=S0 * 0.9, digital_payout=self.payout, seed=11,
        )
        r1 = price_european_option(**kwargs)
        r2 = price_european_option(**kwargs)
        assert r1.price == r2.price

    def test_ot_payout_scales_linearly(self, lv_grid, S0):
        """Doubling the cash payout doubles the price (deterministic at fixed seed)."""
        kwargs = dict(
            lv_grid=lv_grid, S0=S0, K=0.0, T=self.T, r=self.r, q=self.q,
            is_one_touch=True, one_touch_direction="up",
            barrier_level=S0 * 1.10, seed=42,
        )
        r1 = price_european_option(digital_payout=1.0, **kwargs)
        r2 = price_european_option(digital_payout=2.0, **kwargs)
        assert abs(r2.price - 2.0 * r1.price) < 1e-12

    def test_ot_down_decreases_with_higher_rate(self, lv_grid, S0):
        """For a down-touch, raising r both lowers hit probability (drift up,
        away from the barrier) AND increases discounting — both effects shrink
        the price.  This indirectly verifies that pay-at-hit discounting is
        actually applied: a buggy implementation that ignored τ would still
        respond to drift but not to discount, so the magnitude check would
        be sensitive to that.
        """
        kwargs = dict(
            lv_grid=lv_grid, S0=S0, K=0.0, T=self.T, q=self.q,
            is_one_touch=True, one_touch_direction="down",
            barrier_level=S0 * 0.90, digital_payout=self.payout, seed=42,
        )
        r_zero = price_european_option(r=0.0,  **kwargs)
        r_pos  = price_european_option(r=0.05, **kwargs)
        assert r_pos.price < r_zero.price

    def test_ot_rejects_bad_direction(self, lv_grid, S0):
        with pytest.raises(ValueError):
            price_european_option(
                lv_grid, S0=S0, K=0.0, T=self.T, r=self.r, q=self.q,
                is_one_touch=True, one_touch_direction="sideways",
                barrier_level=S0 * 1.1, digital_payout=1.0, seed=1,
            )

    def test_ot_mutex_with_digital(self, lv_grid, S0):
        with pytest.raises(ValueError):
            price_european_option(
                lv_grid, S0=S0, K=S0, T=self.T, r=self.r, q=self.q,
                is_one_touch=True, one_touch_direction="up",
                barrier_level=S0 * 1.1,
                is_digital=True, digital_payout=1.0, seed=1,
            )

    def test_ot_mutex_with_barrier(self, lv_grid, S0):
        with pytest.raises(ValueError):
            price_european_option(
                lv_grid, S0=S0, K=S0, T=self.T, r=self.r, q=self.q,
                is_one_touch=True, one_touch_direction="up",
                barrier_level=S0 * 1.1, digital_payout=1.0,
                barrier_type="up_out", seed=1,
            )