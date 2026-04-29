"""
data_loader.py — Excel ingestion and validation.

Reads the Bloomberg OVME wide-format vol surface sheet and returns a tidy
long-format DataFrame plus a metadata dict.

Actual sheet layout (VolSurface):
  Row 1 : headers  → "Exp Date" | "ImpFwd" | "60.0%" | "80.0%" | … | "140.0%"
  Row 2 : absolute strike prices for each moneyness level (reference only)
  Row 3+: data     → date str  | fwd      | IV%     | IV%     | … | IV%

"ImpFwd" = forward price for that expiry (grows with T, consistent with a
           risk-neutral forward F = S·exp((r-q)·T)).
Moneyness columns are K/F × 100  (so 100.0% = ATM-forward).
Vol values are quoted in percentage points (22.03 → σ = 0.2203).
"""

from __future__ import annotations

from collections import namedtuple
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
VolDataset = namedtuple("VolDataset", ["vol_df", "metadata"])

_SHEET = "VolSurface"
_DATE_COL = 0   # column index in raw sheet
_FWD_COL  = 1
_VOL_START = 2  # first moneyness column index


# ---------------------------------------------------------------------------
def load_workbook(path: str | Path, snap_date: date | None = None) -> VolDataset:
    """Load vol_surface.xlsx and return a validated VolDataset.

    Parameters
    ----------
    path : path to the workbook
    snap_date : date used to compute time-to-expiry; defaults to today.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    snap = snap_date or date.today()
    raw = pd.read_excel(path, sheet_name=_SHEET, header=None)

    vol_df   = _parse_vol_surface(raw, snap)
    metadata = _extract_metadata(raw, vol_df, snap)

    errors = validate(vol_df)
    if errors:
        raise ValueError("Validation errors:\n" + "\n".join(f"  · {e}" for e in errors))

    return VolDataset(vol_df=vol_df, metadata=metadata)


# ---------------------------------------------------------------------------
def _parse_vol_surface(raw: pd.DataFrame, snap: date) -> pd.DataFrame:
    """Melt the wide-format sheet into a tidy long DataFrame."""
    header_row = raw.iloc[0]
    # Collect moneyness labels from columns C onwards (e.g. "60.0%", "80.0%" …)
    # Strip optional trailing "%" before converting to float.
    moneyness_pcts: list[float] = []
    for cell in header_row.iloc[_VOL_START:]:
        if pd.isna(cell):
            break
        moneyness_pcts.append(float(str(cell).replace("%", "")))

    # Row index 1 contains absolute strike prices (reference only) — skip it.
    records: list[dict] = []
    for _, row in raw.iloc[2:].iterrows():
        raw_date = row.iloc[_DATE_COL]
        if pd.isna(raw_date):
            continue

        exp_date = pd.to_datetime(str(raw_date), dayfirst=True).date()
        tte = (exp_date - snap).days / 365.25
        if tte <= 0:
            continue  # skip expired / same-day options

        fwd = float(row.iloc[_FWD_COL])

        for i, m_pct in enumerate(moneyness_pcts):
            vol_val = row.iloc[_VOL_START + i]
            if pd.isna(vol_val):
                continue
            records.append(
                {
                    "expiry_date":    exp_date,
                    "expiry_label":   exp_date.strftime("%d %b %Y"),
                    "time_to_expiry": round(tte, 6),
                    "moneyness_pct":  m_pct,
                    "forward_price":  fwd,
                    "implied_vol":    float(vol_val) / 100.0,  # % → decimal
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Deduplicate rows with identical (expiry_date, moneyness_pct) — keep mean
    df = (
        df.groupby(["expiry_date", "expiry_label", "time_to_expiry", "moneyness_pct"], as_index=False)
        .agg({"forward_price": "first", "implied_vol": "mean"})
    )
    df = df.sort_values(["time_to_expiry", "moneyness_pct"]).reset_index(drop=True)
    return df


def _extract_spot_from_strike_grid(raw: pd.DataFrame) -> float | None:
    """Back-derive the spot price from row 1's absolute-strike grid.

    Bloomberg writes the moneyness percentages in row 0 ("60.0%", "80.0%",
    ..., "140.0%") and the corresponding absolute strikes in row 1
    ("4263.8", "5685.1", ...).  Every cell satisfies `K = m_pct × S` for
    the *same* `S`, so each column independently recovers the spot up to
    per-cell rounding (≈ 5 cents).  Taking the median across columns is
    robust to a stray rounding outlier or a missing column.

    This is the right number to use as `S` in the parity formula `q(T) =
    r(T) − ln(F(T)/S) / T` — Bloomberg's actual reference spot, not the
    earliest forward (which is `S·exp((r−q)·T_earliest)` and so differs
    by ≈ 1 bp of price).  Returns `None` if the row is unusable.
    """
    # Catch the *specific* failure modes of the row-1 strike grid: missing /
    # short rows (IndexError, KeyError), non-numeric cells (ValueError,
    # TypeError).  A bare `except Exception` would also swallow upstream
    # bugs (typos, refactor mistakes) silently.
    try:
        header_row = raw.iloc[0]
        m_pcts: list[float] = []
        for cell in header_row.iloc[_VOL_START:]:
            if pd.isna(cell):
                break
            m_pcts.append(float(str(cell).replace("%", "")))
        if not m_pcts:
            return None
        strike_row = raw.iloc[1, _VOL_START:_VOL_START + len(m_pcts)].values
        strikes = np.asarray(strike_row, dtype=float)
        m_pcts_arr = np.asarray(m_pcts, dtype=float)
        if np.any(m_pcts_arr <= 0) or np.any(np.isnan(strikes)):
            return None
        per_col_spot = strikes / (m_pcts_arr / 100.0)
        return float(np.nanmedian(per_col_spot))
    except (IndexError, KeyError, ValueError, TypeError):
        return None


def _extract_metadata(raw: pd.DataFrame, vol_df: pd.DataFrame, snap: date) -> dict:
    title = "Implied Volatility Surface"

    # Prefer Bloomberg's reference spot encoded in the strike grid (row 1).
    # Falls back to the earliest expiry's forward only if that row is
    # missing or unparseable — the forward is `S·exp((r−q)·T_earliest)`,
    # off by ~1 bp of price for SPX, which is wrong but bounded.
    spot: float | None = _extract_spot_from_strike_grid(raw)
    if spot is None and not vol_df.empty:
        earliest = vol_df.loc[vol_df["time_to_expiry"].idxmin()]
        spot = float(earliest["forward_price"])

    return {
        "title":      title,
        "ticker":     "SPX",
        "currency":   "USD",
        "spot_price": spot,
        "snap_date":  snap.isoformat(),
        "n_expiries": int(vol_df["expiry_date"].nunique()) if not vol_df.empty else 0,
        "n_strikes":  int(vol_df["moneyness_pct"].nunique()) if not vol_df.empty else 0,
    }


# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error strings (empty list = OK)."""
    errors: list[str] = []

    required = ["expiry_date", "time_to_expiry", "moneyness_pct", "implied_vol", "forward_price"]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if df.empty:
        errors.append("No data rows parsed — check sheet name and format.")
        return errors

    if "implied_vol" in df.columns:
        n_neg = int((df["implied_vol"] < 0).sum())
        if n_neg:
            errors.append(f"{n_neg} negative implied_vol value(s)")
        if df["implied_vol"].isna().any():
            errors.append("NaN values found in implied_vol")

    if "time_to_expiry" in df.columns:
        n_bad = int((df["time_to_expiry"] <= 0).sum())
        if n_bad:
            errors.append(f"{n_bad} non-positive time_to_expiry value(s)")

    return errors
