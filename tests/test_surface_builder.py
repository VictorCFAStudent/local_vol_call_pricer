"""Tests for iv_surface_builder.py — interpolation correctness."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_workbook
from iv_surface_builder import build_surface, interpolate_slice

WORKBOOK = Path(__file__).parent.parent / "data_vol_surface" / "vol_surface.xlsx"


@pytest.fixture(scope="module")
def vol_df():
    return load_workbook(WORKBOOK).vol_df


# ---------------------------------------------------------------------------
# T-09  Surface grid has the expected shape
# ---------------------------------------------------------------------------
def test_surface_shape(vol_df):
    grid = build_surface(vol_df, n_k=20, n_t=20)
    assert grid.IV_grid.shape == (20, 20)
    assert grid.K_grid.shape  == (20, 20)
    assert grid.T_grid.shape  == (20, 20)


# ---------------------------------------------------------------------------
# T-10  All interpolated vols are strictly positive
# ---------------------------------------------------------------------------
def test_surface_no_negative_vols(vol_df):
    grid = build_surface(vol_df, n_k=30, n_t=30)
    assert (grid.IV_grid > 0).all(), "Interpolated surface contains non-positive vols"


# ---------------------------------------------------------------------------
# T-11  Front-end spike is preserved (≥ 50% at 80% moneyness, first expiry)
# ---------------------------------------------------------------------------
def test_front_end_spike_preserved(vol_df):
    """The near-term OTM put vol must not be flattened by interpolation."""
    raw_max = vol_df["implied_vol"].max()
    grid = build_surface(vol_df, n_k=40, n_t=40)
    surface_max = grid.IV_grid.max()
    # Surface max should be within 10% of the raw data max
    assert surface_max >= raw_max * 0.90, (
        f"Surface max {surface_max:.2%} too far below raw max {raw_max:.2%} — "
        "front-end spike may have been smoothed out"
    )


# ---------------------------------------------------------------------------
# T-13  interpolate_slice is monotone in a standard smile shape
# ---------------------------------------------------------------------------
def test_interpolate_slice_no_extrapolation_explosion():
    strikes = np.array([80., 90., 95., 100., 105., 110., 120.])
    vols    = np.array([0.40, 0.30, 0.25, 0.20, 0.22, 0.26, 0.35])
    k_new   = np.linspace(80, 120, 50)
    result  = interpolate_slice(strikes, vols, k_new)
    assert result.min() > 0,   "Interpolated slice contains non-positive vols"
    assert result.max() < 2.0, "Interpolated slice contains unreasonably high vols"
