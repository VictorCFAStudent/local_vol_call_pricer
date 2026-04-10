"""Tests for arbitrage_checks.py — economic correctness of each check."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from arbitrage_checks import (
    check_butterfly_spread,
    check_calendar_spread,
    check_vertical_spread,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_df(expiries, moneyness, vols_matrix, fwd=100.0):
    """Build a long-format DataFrame from expiry list, moneyness list, and
    a (len(expiries) × len(moneyness)) matrix of decimal implied vols."""
    snap = date(2026, 1, 1)
    rows = []
    for i, (exp_label, T) in enumerate(expiries):
        for j, m in enumerate(moneyness):
            rows.append({
                "expiry_date":    snap,
                "expiry_label":   exp_label,
                "time_to_expiry": T,
                "moneyness_pct":  m,
                "forward_price":  fwd,
                "implied_vol":    vols_matrix[i][j],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calendar spread
# ---------------------------------------------------------------------------

def test_calendar_no_violation():
    """Monotone total variance → no violations."""
    # w = σ²T: T1=0.25 σ=0.20 → w=0.01; T2=0.5 σ=0.22 → w=0.0242 > 0.01 ✓
    df = _make_df(
        expiries=[("3M", 0.25), ("6M", 0.50)],
        moneyness=[100.0],
        vols_matrix=[[0.20], [0.22]],
    )
    result = check_calendar_spread(df)
    assert result.passed
    assert result.violations == []


def test_calendar_violation_detected():
    """Injected inversion of total variance must be flagged."""
    # T1=0.25 σ=0.30 → w=0.0225; T2=0.50 σ=0.10 → w=0.005 < 0.0225 ✗
    df = _make_df(
        expiries=[("3M", 0.25), ("6M", 0.50)],
        moneyness=[100.0],
        vols_matrix=[[0.30], [0.10]],
    )
    result = check_calendar_spread(df)
    assert not result.passed
    assert len(result.violations) == 1
    assert result.violations[0]["delta_w"] < 0


# ---------------------------------------------------------------------------
# Butterfly spread
# ---------------------------------------------------------------------------

def test_butterfly_no_violation():
    """Standard downward-sloping put skew — d²IV/dK² > 0 everywhere."""
    df = _make_df(
        expiries=[("1Y", 1.0)],
        moneyness=[80., 90., 95., 100., 105., 110., 120.],
        vols_matrix=[[0.40, 0.30, 0.25, 0.20, 0.22, 0.26, 0.35]],
    )
    result = check_butterfly_spread(df)
    assert result.passed


def test_butterfly_violation_detected():
    """A local concavity in the smile must be flagged."""
    # ATM vol (100%) artificially high → concave around it
    df = _make_df(
        expiries=[("1Y", 1.0)],
        moneyness=[80., 90., 100., 110., 120.],
        vols_matrix=[[0.20, 0.18, 0.25, 0.18, 0.20]],
    )
    result = check_butterfly_spread(df)
    assert not result.passed
    assert any(v["moneyness_pct"] == 100.0 for v in result.violations)


# ---------------------------------------------------------------------------
# Vertical spread
# ---------------------------------------------------------------------------

def test_vertical_no_violation():
    """Monotone decreasing call prices → no violations."""
    # Standard put skew: higher vol for lower strikes → call prices decrease as K rises
    df = _make_df(
        expiries=[("1Y", 1.0)],
        moneyness=[80., 90., 100., 110., 120.],
        vols_matrix=[[0.40, 0.30, 0.20, 0.18, 0.17]],
    )
    result = check_vertical_spread(df, r=0.045)
    assert result.passed


def test_vertical_violation_detected():
    """If call price at K2 > call price at K1 < K2, flag it."""
    # Force a smile shape where the call at 110% is priced above 100% call.
    # Use a large upward vol spike at 110% to invert prices.
    df = _make_df(
        expiries=[("1Y", 1.0)],
        moneyness=[100., 110.],
        vols_matrix=[[0.10, 0.90]],   # extreme spike at 110%
    )
    result = check_vertical_spread(df, r=0.0)
    assert not result.passed
    assert result.violations[0]["breach"] > 0


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

def test_run_all_checks_returns_three():
    df = _make_df(
        expiries=[("1Y", 1.0)],
        moneyness=[90., 100., 110.],
        vols_matrix=[[0.25, 0.20, 0.22]],
    )
    results = run_all_checks(df, r=0.045)
    assert len(results) == 3
    names = {r.name for r in results}
    assert names == {"Calendar Spread", "Butterfly Spread", "Vertical Spread"}
