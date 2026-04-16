"""Tests for data_loader.py — schema validation and parsing correctness."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_workbook, validate

_DATA_DIR = Path(__file__).parent.parent / "data_vol_surface"
_CANDIDATES = sorted(_DATA_DIR.glob("vol_surface*.xlsx"))
WORKBOOK = _CANDIDATES[-1] if _CANDIDATES else _DATA_DIR / "vol_surface.xlsx"


# ---------------------------------------------------------------------------
# T-01  Load valid workbook
# ---------------------------------------------------------------------------
def test_load_valid_workbook():
    ds = load_workbook(WORKBOOK)
    assert len(ds.vol_df) > 0, "DataFrame must not be empty"
    assert ds.metadata["n_expiries"] > 0
    assert ds.metadata["n_strikes"] > 0


# ---------------------------------------------------------------------------
# T-02  Expected columns are present
# ---------------------------------------------------------------------------
def test_columns_present():
    ds = load_workbook(WORKBOOK)
    required = {"expiry_date", "expiry_label", "time_to_expiry",
                "moneyness_pct", "forward_price", "implied_vol"}
    assert required.issubset(ds.vol_df.columns)


# ---------------------------------------------------------------------------
# T-03  Implied vols are all positive and in a sensible range
# ---------------------------------------------------------------------------
def test_implied_vol_range():
    ds = load_workbook(WORKBOOK)
    iv = ds.vol_df["implied_vol"]
    assert (iv > 0).all(),   "All implied vols must be positive"
    assert (iv < 2.0).all(), "Implied vols must be below 200%"


# ---------------------------------------------------------------------------
# T-04  Time to expiry is positive for all rows
# ---------------------------------------------------------------------------
def test_time_to_expiry_positive():
    ds = load_workbook(WORKBOOK)
    assert (ds.vol_df["time_to_expiry"] > 0).all()


# ---------------------------------------------------------------------------
# T-05  validate() returns empty list on clean data
# ---------------------------------------------------------------------------
def test_validate_clean():
    ds = load_workbook(WORKBOOK)
    errors = validate(ds.vol_df)
    assert errors == [], f"Unexpected validation errors: {errors}"


# ---------------------------------------------------------------------------
# T-06  validate() catches negative implied vols
# ---------------------------------------------------------------------------
def test_validate_negative_vol():
    ds = load_workbook(WORKBOOK)
    bad_df = ds.vol_df.copy()
    bad_df.loc[0, "implied_vol"] = -0.1
    errors = validate(bad_df)
    assert any("negative" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# T-07  Snap date controls which rows are included
# ---------------------------------------------------------------------------
def test_snap_date_filters_expired():
    # All rows are expired by 2099 → validate() should raise ValueError
    # (an empty surface is not a usable dataset, so raising is the correct behaviour)
    with pytest.raises(ValueError, match="No data rows"):
        load_workbook(WORKBOOK, snap_date=date(2099, 1, 1))


# ---------------------------------------------------------------------------
# T-08  FileNotFoundError on missing workbook
# ---------------------------------------------------------------------------
def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_workbook(Path("nonexistent_file.xlsx"))
