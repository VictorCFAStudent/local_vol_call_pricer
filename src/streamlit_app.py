"""
streamlit_app.py — UI entry-point for the Volatility Surface Explorer.

Run with:
    uv run streamlit run src/streamlit_app.py

UI layout
---------
Sidebar  : file uploader · daily file selector (data_vol_surface/) ·
           rate-curve editor (10 canonical tenors, bootstrapped from US
           Treasury CMT for the snap date) · dividend-curve editor
           (parity-implied at the same canonical tenors, OLS-fitted) ·
           moneyness range slider
Main     : metric cards · 5 tabs (IV Surface [3-D / Smiles / Term Structure /
           Heatmap] / LV Surface / LV MC Pricer / Heston Pricer /
           Arbitrage Checks)
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

# Allow imports from the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from arbitrage_checks import run_all_checks, CheckResult
from data_loader import load_workbook, VolDataset
from local_vol import build_local_vol, LocalVolGrid
from heston import calibrate as heston_calibrate, mc_heston, heston_smile, heston_call
from montecarlo import price_european_option, MCResult
from iv_surface_builder import svi_fit, build_surface, SurfaceGrid
import rates as _rates_mod
from rates import (
    RateCurve,
    TENOR_LABELS,
    TENOR_YEARS,
    build_curve_from_yields,
    build_curve_from_zero_rates,
    flat_curve,
)
from dividends import (
    build_curve_from_table as build_div_from_table,
    extract_implied_at_canonical_tenors as extract_div_canonical,
    flat_div_curve,
    get_div_yield,
)
from plots import (
    plot_arbitrage_flags,
    plot_atm_term_structure,
    plot_heatmap,
    plot_local_vol_3d,
    plot_lv_vs_iv_slice,
    plot_smile_slices,
    plot_surface_3d,
    plot_term_structure,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vol Surface Explorer & Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme / typography
# ---------------------------------------------------------------------------
# Modern-fintech aesthetic: navy primary, slate text, restrained accent
# colours.  System font stacks (no CDN) — SF Pro / Segoe UI render very
# cleanly on macOS / Windows; SF Mono / Consolas / ui-monospace cover
# the numeric monospace.
#
# IMPORTANT: the CSS string MUST start at column 0 with `<style>` and
# every line MUST be flush-left.  Streamlit's markdown renderer applies
# the standard "4-space indent = code block" rule, so any leading
# whitespace turns the entire block into a literal code block (the CSS
# would render as text instead of being applied).
_APP_CSS = """<style>
:root {
  --navy-900:  #0a2540;
  --navy-700:  #1e3a5f;
  --navy-100:  #e8eef5;
  --slate-800: #1e293b;
  --slate-700: #334155;
  --slate-600: #475569;
  --slate-500: #64748b;
  --slate-400: #94a3b8;
  --slate-300: #cbd5e1;
  --slate-200: #e2e8f0;
  --slate-100: #f1f5f9;
  /* Canvas a hair off-white — enough to differentiate cards / inputs as
     "raised panels" without the whole app feeling washed out. */
  --canvas:    #f7f9fc;
  --card-bg:   #ffffff;
  --accent-pos: #0f766e;
  --accent-neg: #b91c1c;
  --accent-warn:#b45309;
  --radius:    8px;
  --font-ui:   -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  --font-mono: ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace;
}

/* Base typography — soft off-white canvas; cards / inputs / pills are
   pure white so they pop against it.  All grey panel components got
   bumped from slate-100 (which used to differentiate against pure white)
   to white-with-stronger-border, which differentiates against slate. */
html, body, [class*="stApp"] {
  font-family: var(--font-ui);
  color: var(--slate-700);
  background: var(--canvas);
}
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-ui);
  color: var(--navy-900);
  font-weight: 600;
  letter-spacing: -0.01em;
}
h1 { font-size: 1.65rem; }
h2 { font-size: 1.3rem; }
h3 { font-size: 1.1rem; }

/* Numeric values monospaced */
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
.stDataFrame, .stTable,
code, kbd, samp {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

/* Tighter top padding */
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 4rem;
  max-width: 1500px;
}

/* Metric cards: flat white panels on the off-white canvas.  Tighter
   vertical padding + slightly smaller value font — the header status
   row reads as a compact summary strip rather than a chunky block. */
[data-testid="stMetric"] {
  background: var(--card-bg);
  border: 1px solid var(--slate-200);
  border-radius: var(--radius);
  padding: 8px 18px;
}
[data-testid="stMetricLabel"] {
  font-size: 0.7rem !important;
  font-weight: 500 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--slate-500) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.35rem !important;
  font-weight: 600 !important;
  color: var(--navy-900) !important;
  line-height: 1.15;
}

/* Tabs — flat with a clear navy underline + heavier weight on the
   active tab (was previously navy-100 fill which blends with the
   off-white canvas). */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid var(--slate-200);
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  flex: 1 1 0;
  justify-content: center;
  font-family: var(--font-ui);
  font-size: 0.95rem;
  font-weight: 500;
  padding: 22px 18px;
  color: var(--slate-500);
  letter-spacing: 0;
  border-right: 1px solid var(--slate-200);
  background: var(--card-bg);
}
.stTabs [data-baseweb="tab"]:first-child {
  border-left: 1px solid var(--slate-200);
}
.stTabs .stTabs [data-baseweb="tab"] {
  flex: 0 0 auto;
  justify-content: flex-start;
  font-size: 0.875rem;
  padding: 10px 22px;
  border-right: none;
  background: transparent;
}
.stTabs .stTabs [data-baseweb="tab"]:first-child {
  border-left: none;
}
.stTabs [aria-selected="true"] {
  color: var(--navy-900) !important;
  font-weight: 600 !important;
  background-color: var(--card-bg);
  border-bottom: 3px solid var(--navy-900) !important;
  box-shadow: inset 0 -1px 0 var(--card-bg);
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--navy-900);
  background-color: var(--slate-100);
}

/* Buttons: navy primary, restrained */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background: var(--navy-900);
  color: #ffffff;
  border: 1px solid var(--navy-900);
  border-radius: var(--radius);
  font-family: var(--font-ui);
  font-weight: 500;
  letter-spacing: 0.01em;
  padding: 8px 18px;
}
.stButton > button[kind="primary"]:hover {
  background: var(--navy-700);
  border-color: var(--navy-700);
  color: #ffffff;
}
.stButton > button:not([kind="primary"]) {
  border: 1px solid var(--slate-300);
  background: var(--card-bg);
  color: var(--slate-700);
  border-radius: var(--radius);
  font-weight: 500;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--navy-900);
  color: var(--navy-900);
  background: var(--card-bg);
}

/* Inputs / selects — explicit white background so the field stands out
   against the off-white canvas (Streamlit defaults to a slate-100
   wrapper which now blends in).  Stronger slate-300 border instead of
   slate-200 for the same reason. */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {
  border-radius: var(--radius) !important;
  border-color: var(--slate-300) !important;
  background: var(--card-bg) !important;
}
[data-baseweb="input"] input,
[data-baseweb="select"] input,
[data-baseweb="textarea"] textarea {
  background: var(--card-bg) !important;
  color: var(--slate-800) !important;
}
[data-baseweb="input"]:focus-within > div,
[data-baseweb="select"]:focus-within > div {
  border-color: var(--navy-900) !important;
  box-shadow: 0 0 0 2px var(--navy-100) !important;
}
[data-baseweb="input"] input[type="number"] {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}
/* Number-input +/- buttons */
[data-testid="stNumberInputContainer"] button {
  background: var(--card-bg) !important;
  border-color: var(--slate-300) !important;
  color: var(--slate-700) !important;
}

/* File uploader (drag-and-drop area) — was disappearing on the canvas */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card-bg) !important;
  border: 1px dashed var(--slate-300) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--navy-900) !important;
  background: var(--card-bg) !important;
}

/* Range slider — the slider's *primary* colour (filled track + value
   label) is set globally via .streamlit/config.toml `primaryColor`,
   because reaching the filled-track div from CSS requires brittle
   deep-child selectors that also catch the min/max label boxes
   (Streamlit renders them in the same `[data-baseweb="slider"] > div`
   subtree).
   Here we only restyle the thumb (stable selector via `role="slider"`)
   to ensure it picks up navy in case the theme colour is overridden,
   and gently recolour the unfilled-track portion to a neutral slate
   so it sits cleanly against the off-white canvas. */
[data-testid="stSlider"] [role="slider"] {
  background-color: var(--navy-900) !important;
  border-color: var(--navy-900) !important;
  box-shadow: none !important;
}
/* Unfilled track (right of thumb) — Streamlit emits this as the second
   sibling div under the thumb's grandparent.  Targeting just this
   shape avoids hitting the min/max label containers. */
[data-testid="stSlider"] [role="slider"] ~ div:last-child {
  background: var(--slate-200) !important;
}

/* Sidebar — pure white panel against the off-white canvas.  Caption
   text bumped to slate-700 with a slightly larger size and tighter
   line-height so the long FRED-link paragraph stays readable.  Selectors
   are deliberately broad so they catch both `st.caption()` (Streamlit
   keeps moving the data-testid) and inline `st.markdown` paragraphs. */
section[data-testid="stSidebar"] {
  border-right: 1px solid var(--slate-200);
  background: var(--card-bg);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  color: var(--navy-900);
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
  font-size: 0.85rem !important;
  color: var(--slate-700) !important;
  line-height: 1.55 !important;
}
section[data-testid="stSidebar"] a {
  color: var(--navy-900) !important;
  text-decoration: underline;
  text-decoration-color: var(--slate-400);
  text-underline-offset: 2px;
  font-weight: 500;
}
section[data-testid="stSidebar"] a:hover {
  text-decoration-color: var(--navy-900);
}
/* Tighten the gap between the file uploader and the next sidebar
   section — Streamlit's default puts ~3rem of empty space below
   `200MB per file - XLSX`. */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  margin-bottom: 0.4rem;
}
section[data-testid="stSidebar"] [data-testid="stDivider"] {
  margin: 1rem 0 !important;
}

/* Data tables */
.stDataFrame thead tr th, .stTable thead tr th {
  background: var(--slate-100) !important;
  color: var(--navy-900) !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

/* Dividers */
hr, [data-testid="stDivider"] {
  border-color: var(--slate-200) !important;
}

/* Status pill helper class (used in header) — navy-tinted background
   so all four pills (Underlying / CCY / Snap / Source) share the look
   that only the `pill-accent` UNDERLYING pill had previously.  The
   label gets a muted navy tone and the value (SPX, USD, …) keeps the
   visual weight via the navy-900 base colour. */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--navy-100);
  color: var(--navy-900);
  border: 1px solid var(--navy-100);
  font-family: var(--font-mono);
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}
.pill .pill-label {
  color: var(--navy-700);
  font-family: var(--font-ui);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.7rem;
  font-weight: 600;
  opacity: 0.85;
}
/* Accent variant — currently a no-op so all four header pills look
   identical; kept as a hook for future "alert" pills (e.g. could
   reintroduce a stronger border or a different fill colour to
   highlight a specific pill). */
.pill-accent {
  /* intentionally empty — same as base .pill */
}

/* Header block — title, subtitle and status pills stack vertically so
   each line stays readable and the title doesn't get clipped against
   any flex baselines. */
.app-header {
  padding: 12px 0 14px 0;
  border-bottom: 1px solid var(--slate-200);
  margin-bottom: 18px;
}
.app-header .title {
  font-family: var(--font-ui);
  font-size: 1.55rem;
  font-weight: 600;
  color: var(--navy-900);
  letter-spacing: -0.015em;
  line-height: 1.4;
}
.app-header .subtitle {
  font-family: var(--font-ui);
  font-size: 0.88rem;
  color: var(--slate-600);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.app-header .status-row {
  display: flex;
  gap: 8px;
  flex-wrap: nowrap;
  margin-top: 12px;
  overflow-x: auto;
}
.pill .pill-label::after { content: ":"; }

/* Heston calibration result panel — custom grid (not st.metric) so the
   Greek symbols (v₀ / κ / θ / ξ / ρ) render in their actual lowercase
   form.  The default st.metric label has text-transform: uppercase, which
   would convert lowercase Greek letters into their uppercase forms
   (κ → Κ, ξ → Ξ, ρ → Ρ) — visually wrong for a model display. */
.heston-result {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin: 6px 0 16px 0;
}
.heston-param-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
}
.heston-param {
  background: var(--card-bg);
  border: 1px solid var(--slate-200);
  border-radius: var(--radius);
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.heston-param-symbol {
  font-family: var(--font-ui);
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--slate-500);
  letter-spacing: 0;
  text-transform: none;
}
.heston-param-value {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--navy-900);
  line-height: 1.15;
}
.heston-param-meta {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: 0.74rem;
  color: var(--slate-500);
  background: var(--slate-100);
  padding: 2px 7px;
  border-radius: 4px;
  align-self: flex-start;
  letter-spacing: 0;
  margin-top: 2px;
}
.heston-quality-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr)) auto;
  gap: 10px;
  align-items: stretch;
}
.heston-quality-stat {
  background: var(--card-bg);
  border: 1px solid var(--slate-200);
  border-radius: var(--radius);
  padding: 10px 14px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.heston-quality-label {
  font-family: var(--font-ui);
  font-size: 0.7rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--slate-500);
  display: flex;
  align-items: center;
  gap: 6px;
}
/* Tiny circled "?" — visual cue that the card carries a hover tooltip
   (the actual tooltip text lives on the parent card's `title` attribute,
   so hovering anywhere on the card surfaces it). */
.help-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--slate-100);
  color: var(--slate-500);
  font-family: var(--font-ui);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: none;
  line-height: 1;
  cursor: help;
}
.heston-quality-value {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--navy-900);
}
.heston-badges {
  display: flex;
  gap: 6px;
  align-items: center;
  padding: 0 4px;
}
.heston-badge {
  display: inline-flex;
  align-items: center;
  padding: 5px 12px;
  border-radius: 999px;
  font-family: var(--font-ui);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  border: 1px solid transparent;
}
.heston-badge.ok {
  background: #ecfdf5;
  color: #065f46;
  border-color: #a7f3d0;
}
.heston-badge.warn {
  background: #fef3c7;
  color: #78350f;
  border-color: #fde68a;
}
</style>"""

st.markdown(_APP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached loader — keyed on file content + snap date + interpolation method
# so Streamlit only re-computes when something actually changes.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_and_build(
    file_bytes: bytes,
    snap_date_iso: str,
) -> tuple[VolDataset, SurfaceGrid]:
    snap = date.fromisoformat(snap_date_iso)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        dataset = load_workbook(tmp_path, snap_date=snap)
        grid    = build_surface(dataset.vol_df)
        return dataset, grid
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner=False)
def _run_checks(
    file_bytes: bytes,
    snap_date_iso: str,
    curve_tenors: tuple,
    curve_rates: tuple,
) -> list[CheckResult]:
    """Cache keyed on the raw file + snap date + curve fingerprint.

    The curve is reconstructed inside (cache key uses the immutable tuple
    representation so a user override invalidates correctly)."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        from datetime import date as _date
        ds = load_workbook(tmp_path, snap_date=_date.fromisoformat(snap_date_iso))
        curve = RateCurve(
            tenors_yr=curve_tenors,
            zero_rates=curve_rates,
            source="cached",
            snap_date_iso=snap_date_iso,
        )
        return run_all_checks(ds.vol_df, r=curve)
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner=False)
def _build_lv(file_bytes: bytes, snap_date_iso: str, n: int) -> LocalVolGrid:
    """Compute Dupire local vol — cached at top level so the decorator is
    registered once, not re-registered on every tab render."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        ds = load_workbook(tmp_path, snap_date=date.fromisoformat(snap_date_iso))
        return build_local_vol(ds.vol_df, n_k=n, n_t=n)
    finally:
        os.unlink(tmp_path)


def _export_violations(checks: list[CheckResult]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for check in checks:
            if check.violations:
                sheet = check.name.replace(" ", "")[:31]
                pd.DataFrame(check.violations).to_excel(writer, sheet_name=sheet, index=False)
    buf.seek(0)
    return buf.read()


def _badge(passed: bool) -> str:
    if passed:
        return '<span style="background:#d4edda;color:#155724;padding:2px 10px;border-radius:4px;font-weight:600;">PASS</span>'
    return '<span style="background:#f8d7da;color:#721c24;padding:2px 10px;border-radius:4px;font-weight:600;">FAIL</span>'


@st.cache_data(show_spinner=False)
def _cached_treasury_yields(snap_date_iso: str) -> tuple[dict | None, str | None]:
    """Cached Treasury fetch.  Returns (par_yields_dict, None) on success, or
    (None, error_message) on failure.  Cache key is the ISO snap-date so the
    same date is never re-fetched within a session."""
    try:
        from rates import fetch_treasury_yields, TreasuryFetchError
        snap_d = date.fromisoformat(snap_date_iso)
        return fetch_treasury_yields(snap_d), None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data_vol_surface"


def _file_date(p: Path) -> tuple[int, int, int]:
    """Sort key: extract (yyyy, mm, dd) from vol_surface_dd_mm_yyyy.xlsx.
    Returns (0, 0, 0) for files that don't match the pattern."""
    import re
    match = re.search(r"(\d{2})_(\d{2})_(\d{4})", p.stem)
    if match:
        d, mo, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return (y, mo, d)
    return (0, 0, 0)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Controls")

    uploaded = st.file_uploader(
        "Upload a workbook",
        type=["xlsx"],
        help="Upload any Bloomberg OVME wide-format workbook directly",
    )

    st.divider()
    st.subheader("Data file")

    _available = sorted(_DATA_DIR.glob("*.xlsx"), key=_file_date, reverse=True)
    if _available:
        _file_options = {f.name: f for f in _available}
        _selected_name = st.selectbox(
            "Select daily file",
            options=list(_file_options.keys()),
            index=0,
            help="Files from data_vol_surface/ — most recent first",
            disabled=(uploaded is not None),
        )
        if uploaded is None:
            st.caption(f"{len(_available)} file(s) available")
        else:
            st.caption("File selector disabled — using uploaded file above")
    else:
        _selected_name = None
        st.caption("No files found in data_vol_surface/")


    # The rate-curve editor is added later (after snap_date is resolved)
    # via this placeholder, since the curve is keyed on the snapshot date.
    rate_curve_placeholder = st.empty()

    # Dividend-curve editor — populated after vol_df + rate_curve are built,
    # since q is implied from F (Bloomberg) and r (rate curve).
    div_curve_placeholder = st.empty()

    # Placeholder for moneyness range — populated once data loads
    range_slider_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Resolve data source
# ---------------------------------------------------------------------------
def _date_from_filename(name: str) -> date:
    """Extract snapshot date from vol_surface_dd_mm_yyyy.xlsx; fall back to today."""
    import re
    m = re.search(r"(\d{2})_(\d{2})_(\d{4})", name)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(y, mo, d)
    return date.today()


# ---------------------------------------------------------------------------
# Shared MC pricer form (LV-MC tab + Heston-MC tab)
# ---------------------------------------------------------------------------
# Both tabs render the same trade-and-engine input form (payoff selector,
# option type / OT direction, optional barrier, S₀ / K / T / r / q, steps,
# ε) and run the same per-batch sanity checks on barrier placement.  Only
# the actual pricing call and the result-display block differ — the
# Heston tab additionally renders an analytical reference for vanillas.
# Keeping the form in one helper avoids two ~270-line copies drifting
# apart silently when one is edited and the other isn't.
# ---------------------------------------------------------------------------
from typing import NamedTuple as _NamedTuple


class _PricerFormInputs(_NamedTuple):
    """Resolved inputs from `_render_mc_pricer_form`."""
    S0: float
    K: float
    T: float
    r: float
    q: float
    option_type: str
    is_digital: bool
    is_one_touch: bool
    one_touch_direction: str | None
    digital_payout: float
    barrier_type: str | None
    barrier_level: float | None
    barrier_style: str
    steps_per_year: int
    eps: float


def _render_mc_pricer_form(
    *,
    prefix: str,
    spot_default: float,
    T_max: float,
    T_help_suffix: str,
    rate_curve,
    div_curve,
    eps_help: str,
) -> _PricerFormInputs:
    """Render the Monte Carlo pricer input form and return resolved inputs.

    Used by both the LV-MC tab (`prefix='mc'`) and the Heston-MC tab
    (`prefix='heston'`); all Streamlit widget keys are namespaced under
    `prefix` so the two forms keep independent session state.

    Side effects: warnings for barrier-on-the-wrong-side-of-spot and
    one-touch-already-past-the-barrier are emitted from inside this helper
    (they are common to both tabs).  Tab-specific warnings — e.g. the
    LV-MC tab's "strike outside surface moneyness range" warning — are
    the caller's responsibility, after the form has been rendered.
    """
    # ── Payoff selector ──────────────────────────────────────────────────
    _payoff = st.segmented_control(
        "Payoff",
        options=["European Vanilla / Barrier", "Digital", "One-Touch"],
        default="European Vanilla / Barrier",
        selection_mode="single",
        key=f"{prefix}_payoff_seg",
        help=(
            "European Vanilla / Barrier — max(S_T − K, 0) for call / "
            "max(K − S_T, 0) for put, optionally combined with a "
            "knock-in or knock-out barrier.  European-exercise only "
            "(no early exercise modelled).  "
            "Digital — fixed cash payout if S_T is ITM at expiry, else 0.  "
            "One-Touch — cash payout the moment spot first touches a "
            "barrier (pay-at-hit, FX market convention, "
            "American-monitored).  No strike."
        ),
    ) or "European Vanilla / Barrier"

    is_digital   = (_payoff == "Digital")
    is_one_touch = (_payoff == "One-Touch")

    # ── Option type / OT direction (own row, directly under Payoff) ──────
    if is_one_touch:
        _ot_dir = st.segmented_control(
            "Direction",
            options=["Up", "Down"],
            default="Up",
            selection_mode="single",
            key=f"{prefix}_ot_dir_seg",
            help=(
                "Up — barrier above spot, pays on first up-cross.  "
                "Down — barrier below spot, pays on first down-cross."
            ),
        )
        one_touch_direction = (_ot_dir or "Up").lower()
        option_type = "call"  # unused for OT
    else:
        _t = st.segmented_control(
            "Option type",
            options=["Call", "Put"],
            default="Call",
            selection_mode="single",
            key=f"{prefix}_type_seg",
        )
        option_type = (_t or "Call").lower()
        one_touch_direction = None

    digital_payout = 1.0
    if is_digital or is_one_touch:
        _label = (
            "Cash payout (paid on touch)" if is_one_touch
            else "Cash payout (paid at expiry if ITM)"
        )
        digital_payout = st.number_input(
            _label,
            min_value=0.0001, value=1.0, step=0.5, format="%.4f",
            key=f"{prefix}_payout",
            help=(
                "Cash amount delivered when the option pays out.  "
                "Common conventions: 1.0 (unit), or the option notional."
            ),
        )

    # ── Barrier section ──────────────────────────────────────────────────
    # Vanilla:  selector full width  → if a barrier is picked, two paired
    #           controls on a single row [Monitoring | Level].
    # Digital:  no barrier (caption explains).
    # One-Touch: OT *is* the barrier product; the level lives in the
    #           Strike column of the numeric grid below.
    barrier_type  = None
    barrier_level = None
    barrier_style = "american"

    if is_one_touch:
        pass  # barrier level rendered in the Strike column below
    elif is_digital:
        st.caption("Barrier options are disabled for digitals.")
    else:
        _BARRIER_LABELS = {
            "None":         None,
            "Up-and-Out":   "up_out",
            "Up-and-In":    "up_in",
            "Down-and-Out": "down_out",
            "Down-and-In":  "down_in",
        }
        _b_choice = st.segmented_control(
            "Barrier",
            options=list(_BARRIER_LABELS.keys()),
            default="None",
            selection_mode="single",
            key=f"{prefix}_barrier_seg",
            help=(
                "Up = barrier above spot, Down = below.  "
                "Out = option dies if barrier is touched.  "
                "In = option only activates once barrier is touched."
            ),
        )
        barrier_type = _BARRIER_LABELS[_b_choice or "None"]

        if barrier_type is not None:
            _b_mon_col, _b_level_col = st.columns(2)
            with _b_mon_col:
                _mon = st.segmented_control(
                    "Barrier monitoring",
                    options=["American", "European"],
                    default="American",
                    selection_mode="single",
                    key=f"{prefix}_monitor_seg",
                    help=(
                        "American: barrier monitored at every Euler step throughout the life of the option "
                        "(continuous-style, slight upper bound on knock-out price under daily monitoring).  "
                        "European: barrier checked only at expiry — S_T alone decides hit / no-hit, "
                        "i.e. the option behaves as a vanilla extinguished only if the terminal spot breaches B."
                    ),
                )
                barrier_style = (_mon or "American").lower()
            with _b_level_col:
                _is_up = barrier_type.startswith("up")
                _default_B = spot_default * (1.10 if _is_up else 0.90)
                barrier_level = st.number_input(
                    "Barrier level B",
                    min_value=0.01, value=float(_default_B), step=10.0,
                    key=f"{prefix}_barrier_level",
                    help=(
                        "Up barriers require B > S₀; down barriers require B < S₀.  "
                        "An option already past its barrier at inception is degenerate "
                        "(knock-out → 0, knock-in → vanilla)."
                    ),
                )

    # ── Numeric inputs ───────────────────────────────────────────────────
    # Row 1: Spot | Strike (or barrier for OT) | Maturity.
    # Row 2: Risk-free rate | Dividend yield (both T-dependent, curve / manual).
    # Row 3: Steps per year | Convergence ε (MC engine settings).
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        S0 = st.number_input(
            "Spot S₀",
            min_value=1.0, value=spot_default, step=10.0,
            key=f"{prefix}_S0",
            help="Current spot price (pre-filled from the loaded dataset)",
        )
    with _c2:
        if is_one_touch:
            _ot_default_B = spot_default * (1.10 if one_touch_direction == "up" else 0.90)
            barrier_level = st.number_input(
                "Barrier level B",
                min_value=0.01, value=float(_ot_default_B), step=10.0,
                key=f"{prefix}_ot_barrier_level",
                help=(
                    "Up direction requires B > S₀; down direction requires B < S₀.  "
                    "If B is already past the spot at inception, payoff is delivered immediately "
                    "(price = cash payout, no discount)."
                ),
            )
            K = 0.0  # unused for OT
        else:
            K = st.number_input(
                "Strike K",
                min_value=1.0, value=spot_default, step=10.0,
                key=f"{prefix}_K",
                help="Option strike (default = ATM)",
            )
    with _c3:
        T = st.number_input(
            "Maturity T (years)",
            min_value=0.001, max_value=float(T_max),
            value=1.000, step=0.0001, format="%.4f",
            key=f"{prefix}_T",
            help=(
                f"Option maturity in years ({T_help_suffix}).  "
                "Any 4-decimal value is accepted — e.g. 0.2500 (3 m), 0.1781 (65 d)."
            ),
        )

    _rate_col, _div_col = st.columns(2)
    with _rate_col:
        _r_use_curve = st.checkbox(
            "Use curve r at T",
            value=True,
            key=f"{prefix}_r_use_curve",
            help=(
                "When ticked, the risk-free rate is read from the "
                "sidebar rate curve at the current maturity T and "
                "updates live as T changes.  Untick to enter a flat r "
                "manually (e.g. for stress-testing rate sensitivity)."
            ),
        )
        if _r_use_curve:
            r = float(rate_curve.zero_rate(float(T)))
            st.markdown(f"**Risk-free rate r (%)** &nbsp; `{r * 100:.3f}`")
            st.caption(f"📊 curve (T = {T:.4f} yr)")
        else:
            r = st.number_input(
                "Risk-free rate r (%)",
                min_value=-50.0, max_value=20.0,
                value=4.500, step=0.001, format="%.3f",
                key=f"{prefix}_r_manual",
                help="Continuously-compounded risk-free rate in %.",
            ) / 100.0
            st.caption("✏️ manual override")

    with _div_col:
        _q_use_curve = st.checkbox(
            "Use curve q at T",
            value=True,
            key=f"{prefix}_q_use_curve",
            help=(
                "When ticked, the dividend yield is read from the "
                "sidebar curve at the current maturity T and updates "
                "live as T changes.  Untick to enter q manually."
            ),
        )
        if _q_use_curve:
            q = float(get_div_yield(div_curve, float(T)))
            st.markdown(f"**Dividend yield q (%)** &nbsp; `{q * 100:.3f}`")
            st.caption(f"📊 implied (T = {T:.4f} yr)")
        else:
            q = st.number_input(
                "Dividend yield q (%)",
                min_value=-50.0, max_value=20.0,
                value=0.000, step=0.001, format="%.3f",
                key=f"{prefix}_q_manual",
                help="Continuous dividend yield in %.",
            ) / 100.0
            st.caption("✏️ manual override")

    _steps_col, _eps_col = st.columns(2)
    with _steps_col:
        steps_per_year = st.selectbox(
            "Steps per year",
            options=[252, 52],
            index=0,
            format_func=lambda x: f"{x}  ({'daily' if x == 252 else 'weekly'})",
            key=f"{prefix}_steps",
            help="Number of Euler time steps per year",
        )
    with _eps_col:
        eps = st.number_input(
            "Convergence ε",
            min_value=1e-6, max_value=0.05,
            value=0.001, step=0.0005, format="%.4f",
            key=f"{prefix}_eps",
            help=eps_help,
        )

    # ── Common sanity warnings (barrier and OT placement vs spot) ────────
    if barrier_type is not None:
        if barrier_type.startswith("up") and barrier_level <= S0:
            st.warning(
                f"Up barrier B={barrier_level:g} is not above spot S₀={S0:g}. "
                "The option is knocked out / activated at inception."
            )
        elif barrier_type.startswith("down") and barrier_level >= S0:
            st.warning(
                f"Down barrier B={barrier_level:g} is not below spot S₀={S0:g}. "
                "The option is knocked out / activated at inception."
            )

    if is_one_touch:
        if one_touch_direction == "up" and barrier_level <= S0:
            st.warning(
                f"Up one-touch barrier B={barrier_level:g} is not above spot "
                f"S₀={S0:g}.  Payoff is delivered immediately at inception "
                f"(price ≈ cash payout)."
            )
        elif one_touch_direction == "down" and barrier_level >= S0:
            st.warning(
                f"Down one-touch barrier B={barrier_level:g} is not below spot "
                f"S₀={S0:g}.  Payoff is delivered immediately at inception "
                f"(price ≈ cash payout)."
            )

    return _PricerFormInputs(
        S0=S0, K=K, T=T, r=r, q=q,
        option_type=option_type,
        is_digital=is_digital,
        is_one_touch=is_one_touch,
        one_touch_direction=one_touch_direction,
        digital_payout=digital_payout,
        barrier_type=barrier_type,
        barrier_level=barrier_level,
        barrier_style=barrier_style,
        steps_per_year=int(steps_per_year),
        eps=float(eps),
    )


if uploaded is not None:
    file_bytes   = uploaded.read()
    source_label = f"Uploaded: {uploaded.name}"
    snap_date    = _date_from_filename(uploaded.name)
elif _selected_name is not None:
    file_bytes   = _file_options[_selected_name].read_bytes()
    source_label = f"data_vol_surface/{_selected_name}"
    snap_date    = _date_from_filename(_selected_name)
else:
    st.info("Add .xlsx files to the data_vol_surface/ folder or upload one in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading and interpolating surface…"):
    try:
        dataset, grid = _load_and_build(file_bytes, snap_date.isoformat())
    except Exception as exc:
        st.error(f"**Failed to load data:** {exc}")
        st.stop()

vol_df = dataset.vol_df
meta   = dataset.metadata

# ---------------------------------------------------------------------------
# Rate curve — fetch + bootstrap + editable override
# ---------------------------------------------------------------------------
# Treasury fetch is cached on snap_date.  The user sees the bootstrapped
# continuous-compounded zero rates in an editable table; any cell they
# overwrite flips the curve provenance to "manual".  If the fetch fails
# (network down, snap_date in the future, etc.) we fall back to a manual-
# entry mode pre-populated with a flat 4.5 % curve.
_yields_default, _yield_fetch_err = _cached_treasury_yields(snap_date.isoformat())

if _yields_default is not None:
    try:
        _curve_default = build_curve_from_yields(_yields_default, snap_date=snap_date)
        _default_table = _curve_default.as_table()
    except Exception as exc:
        _yield_fetch_err = f"Bootstrap failed: {exc}"
        _curve_default = None
else:
    _curve_default = None

if _curve_default is None:
    _default_table = pd.DataFrame({
        "Tenor": list(TENOR_LABELS),
        "Years": [TENOR_YEARS[lbl] for lbl in TENOR_LABELS],
        "Zero rate (%)": [4.500] * len(TENOR_LABELS),
    })

with rate_curve_placeholder.container():
    st.divider()
    st.subheader("Rate curve")

    st.caption(
        "💡 Bloomberg builds option forwards from a SOFR/futures-implied "
        "rate curve, so Treasury-bootstrapped rates leave a SOFR-Treasury "
        "basis (~30-50 bp at long tenors) baked into the implied dividend.  "
        "If you want Treasury rates from a different fitting methodology, "
        "the Federal Reserve Board's **Kim-Wright fitted zero-coupon "
        "yields** are published daily and continuously-compounded at "
        "[FRED](https://fred.stlouisfed.org/release/tables?rid=354&eid=212994) "
        "— pick the date in the page's date selector and paste the values "
        "directly into the table below.  Note: this is still **Treasury**, "
        "not SOFR, so it won't materially close the basis to Bloomberg.  "
        "A free, no-key SOFR-OIS curve source isn't currently available."
    )

    if _yield_fetch_err:
        st.warning(
            f"⚠️ Treasury fetch unavailable for **{snap_date.isoformat()}** — "
            f"please enter zero rates manually below.\n\n"
            f"_Reason: {_yield_fetch_err}_"
        )
    else:
        st.caption(
            f"📊 Bootstrapped from US Treasury CMT par yields on "
            f"**{snap_date.isoformat()}** (or the closest preceding business day)."
        )

    _edited_table = st.data_editor(
        _default_table,
        column_config={
            "Tenor": st.column_config.TextColumn(
                "Tenor",
                disabled=True,
                help="Treasury constant-maturity tenor",
            ),
            "Years": st.column_config.NumberColumn(
                "Years",
                disabled=True,
                format="%.4f",
                help="Tenor expressed in calendar years",
            ),
            "Zero rate (%)": st.column_config.NumberColumn(
                "Zero rate (%)",
                format="%.3f",
                min_value=-5.0, max_value=20.0, step=0.001,
                help=(
                    "Continuously-compounded zero rate.  Pre-filled from the "
                    "Treasury bootstrap; edit any cell to override.  Rates not "
                    "shown are interpolated log-linearly in discount-factor "
                    "space (piecewise-flat forwards) at pricing time."
                ),
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="rate_curve_editor",
    )

    # Rebuild the curve from the (possibly edited) table.
    _user_modified = bool(
        (_edited_table["Zero rate (%)"].values
         != _default_table["Zero rate (%)"].values).any()
    )
    # Provenance: the user edit takes priority; otherwise treasury (fetch ok)
    # or flat fallback (fetch failed and user accepted defaults as-is).
    if _user_modified:
        _src = "manual"
    elif _yield_fetch_err:
        _src = "flat"
    else:
        _src = "treasury"

    _edit_dict = {
        str(row["Tenor"]): float(row["Zero rate (%)"])
        for _, row in _edited_table.iterrows()
        if pd.notna(row["Zero rate (%)"])
    }
    try:
        rate_curve = build_curve_from_zero_rates(
            _edit_dict, snap_date=snap_date, source=_src,
        )
    except ValueError:
        # Empty table edge case — fall back to flat 4.5 %.
        rate_curve = flat_curve(0.045, snap_date=snap_date)
        _src = "flat"

    # Status badge
    _src_label = {
        "treasury": "✅ US Treasury bootstrap",
        "manual":   "✏️ Manual override",
        "flat":     "↔️ Flat fallback (4.50 %)",
    }.get(_src, _src)
    st.caption(_src_label)


# ---------------------------------------------------------------------------
# Dividend curve — implied from Bloomberg forwards + (possibly edited) rates
# ---------------------------------------------------------------------------
# `q(T) = r(T) − ln(F(T)/S) / T`.  Re-extracted on every rerun against the
# *current* rate curve, so editing a rate immediately re-flows the implied
# div column.  The user can override any cell to switch provenance to manual,
# but those overrides only persist as long as the rate curve doesn't change:
# the data-editor's `key` includes a hash of the zero rates, so any rate
# edit remounts the editor with freshly-recomputed defaults (q is *implied
# from* r — if r moves, q must too).
_spot_for_div = float(meta.get("spot_price", 0.0))
try:
    _div_default_table = extract_div_canonical(
        vol_df, _spot_for_div, rate_curve, snap_date=snap_date, max_T=5.0,
    ).as_table()
    _div_extract_err: str | None = None
except Exception as exc:
    _div_default_table = pd.DataFrame({
        "Tenor": ["fallback"], "Years": [1.0], "q (%)": [0.0],
    })
    _div_extract_err = str(exc)

with div_curve_placeholder.container():
    st.divider()
    st.subheader("Dividend curve")

    if _div_extract_err:
        st.warning(
            f"⚠️ Could not extract implied dividends: {_div_extract_err}"
        )
    else:
        st.caption(
            "📊 Implied from Bloomberg forwards via "
            "`q(T) = r(T) − ln(F(T)/S) / T`, sampled at canonical tenors.  "
            "The full per-expiry curve is used for interpolation at pricing "
            "time; canonical sampling here hides front-end parity noise "
            "from discrete dividend ex-dates."
        )

    _div_edited_table = st.data_editor(
        _div_default_table,
        column_config={
            "Tenor": st.column_config.TextColumn(
                "Tenor", disabled=True,
                help="Canonical maturity (matches the rate-curve grid)",
            ),
            "Years": st.column_config.NumberColumn(
                "Years", disabled=True, format="%.4f",
                help="Time to expiry in calendar years",
            ),
            "q (%)": st.column_config.NumberColumn(
                "q (%)", format="%.3f",
                min_value=-5.0, max_value=20.0, step=0.001,
                help=(
                    "Continuously-compounded implied dividend yield.  "
                    "Pre-filled from the Bloomberg forward and the rate "
                    "curve above; edit any cell to override.  Yields "
                    "between knots are interpolated linearly in q·T "
                    "(piecewise-flat instantaneous div); flat extrapolation "
                    "beyond the curve range."
                ),
            ),
        },
        hide_index=True, use_container_width=True,
        # Bind the widget key to a hash of the current rate curve so that
        # any change to the rate table forces this editor to remount with
        # the freshly-recomputed implied q values.  Without this, Streamlit
        # would replay the user's stale q edits on top of new defaults and
        # the dividend column would drift out of sync with the rates the
        # user just typed.  Manual q overrides therefore live only as long
        # as the rate curve they were entered against — which is the right
        # invariant: q is *implied from* r, so if r moves, q must too.
        key=f"div_curve_editor_{hash(tuple(round(r, 12) for r in rate_curve.zero_rates))}",
    )

    _div_user_modified = bool(
        (_div_edited_table["q (%)"].values
         != _div_default_table["q (%)"].values).any()
    )
    # User edits win over the fallback label — even if extract failed and
    # the user is editing the 1-row fallback table, those edits are real.
    if _div_user_modified:
        _div_src = "manual"
    elif _div_extract_err:
        _div_src = "flat"
    else:
        _div_src = "implied"

    try:
        div_curve = build_div_from_table(
            _div_edited_table, snap_date=snap_date, source=_div_src,
        )
    except ValueError:
        div_curve = flat_div_curve(0.0, snap_date=snap_date)
        _div_src = "flat"

    _div_src_label = {
        "implied": "✅ Implied from Bloomberg forwards",
        "manual":  "✏️ Manual override",
        "flat":    "↔️ Flat fallback (0.00 %)",
    }.get(_div_src, _div_src)
    st.caption(_div_src_label)


# ---------------------------------------------------------------------------
# Sidebar filters (now that data is loaded)
# ---------------------------------------------------------------------------
expiry_labels  = (
    vol_df.drop_duplicates("expiry_label")
    .sort_values("time_to_expiry")["expiry_label"]
    .tolist()
)
moneyness_vals = sorted(vol_df["moneyness_pct"].unique().tolist())

# Bucket expiries by tenor for the quick-select buttons
_tte_map = (
    vol_df.drop_duplicates("expiry_label")
    .set_index("expiry_label")["time_to_expiry"]
    .to_dict()
)
_short  = [l for l in expiry_labels if _tte_map[l] <= 1/12]        # ≤ 1 month
_medium = [l for l in expiry_labels if 1/12 < _tte_map[l] <= 1.0]  # 1 m – 1 yr
_long   = [l for l in expiry_labels if _tte_map[l] > 1.0]          # > 1 yr

with range_slider_placeholder.container():
    st.subheader("Moneyness Range")
    strike_range = st.slider(
        "Moneyness range (%)",
        min_value=float(moneyness_vals[0]),
        max_value=float(moneyness_vals[-1]),
        value=(float(moneyness_vals[0]), float(moneyness_vals[-1])),
        step=0.5,
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Header — title bar + status pills (snap date, ticker, source)
# ---------------------------------------------------------------------------
_ticker     = meta.get("ticker",    "—")
_snap_iso   = meta.get("snap_date", "—")
_currency   = meta.get("currency",  "—")
_source     = source_label.replace("data_vol_surface/", "")  # short form

# Header HTML — three stacked rows (title, subtitle, status pills),
# each on a single line, so the subtitle is never wrapped onto two lines
# and the pills never collide with the subtitle.  Flush-left to dodge
# Streamlit's "4-space indent = code block" markdown rule.
_HEADER_HTML = (
    '<div class="app-header">'
    '<div class="title">📈 Volatility Surface Explorer</div>'
    '<div class="subtitle">Implied vol surface · Dupire local vol · Heston stochastic vol</div>'
    '<div class="status-row">'
    f'<span class="pill pill-accent"><span class="pill-label">Underlying</span>&nbsp;{_ticker}</span>'
    f'<span class="pill"><span class="pill-label">CCY</span>&nbsp;{_currency}</span>'
    f'<span class="pill"><span class="pill-label">Snap</span>&nbsp;{_snap_iso}</span>'
    f'<span class="pill"><span class="pill-label">Source</span>&nbsp;{_source}</span>'
    '</div>'
    '</div>'
)
st.markdown(_HEADER_HTML, unsafe_allow_html=True)

# Status row: spot + dataset shape.  Replaces the previous 5-card grid which
# mixed identifiers (Ticker, Source) with payload (Spot, Expiries, Strikes).
# Identifiers are now in the header pills above; this row is payload only.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot",         f"{meta.get('spot_price', 0):,.2f}")
c2.metric("Expiries",     f"{meta.get('n_expiries', 0)}")
c3.metric("Strikes",      f"{meta.get('n_strikes',  0)}")
c4.metric("Interpolation", "Bicubic Spline")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_iv, tab_lv, tab_mc, tab_heston, tab_arb = st.tabs(
    ["IV Surface", "LV Surface", "LV MC Pricer", "Heston Pricer", "⚠️ Arbitrage Checks"]
)

# ── IV Surface (3-D / Smiles / Term Structure) ──────────────────────────────
with tab_iv:
    st.subheader("Implied Volatility Surface")

    # ── Desk-style summary: ATM level at two tenors + skew / convexity ─────
    # Skew and butterfly are computed on the 90 / 100 / 110 moneyness points,
    # which is the Bloomberg OVME equivalent of the 25-delta convention.
    def _nearest_slice(target_T: float) -> tuple[float, pd.DataFrame]:
        expiries = vol_df[["time_to_expiry"]].drop_duplicates()
        idx = (expiries["time_to_expiry"] - target_T).abs().idxmin()
        t_act = float(expiries.loc[idx, "time_to_expiry"])
        return t_act, vol_df[vol_df["time_to_expiry"] == t_act]

    def _iv_at(slice_df: pd.DataFrame, m_pct: float) -> float | None:
        match = slice_df[slice_df["moneyness_pct"] == m_pct]
        return float(match["implied_vol"].iloc[0]) if len(match) else None

    _t_1m, _slice_1m = _nearest_slice(1.0 / 12.0)
    _t_1y, _slice_1y = _nearest_slice(1.0)

    _atm_1m = _iv_at(_slice_1m, 100.0)
    _atm_1y = _iv_at(_slice_1y, 100.0)
    _iv_90  = _iv_at(_slice_1m,  90.0)
    _iv_110 = _iv_at(_slice_1m, 110.0)

    _skew_1m = (_iv_90 - _iv_110) if (_iv_90 is not None and _iv_110 is not None) else None
    _bf_1m   = (0.5 * (_iv_90 + _iv_110) - _atm_1m) if (
        _iv_90 is not None and _iv_110 is not None and _atm_1m is not None
    ) else None

    def _fmt_pct(v):    return f"{v*100:.2f}%"    if v is not None else "—"
    def _fmt_diff(v):   return f"{v*100:+.2f} pp" if v is not None else "—"

    def _tenor_label(t: float) -> str:
        if t < 1.0 / 12.0 * 1.5:
            return f"~{t*365:.0f}D"
        if t < 1.0:
            return f"~{t*12:.0f}M"
        return f"~{t:.1f}Y"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"ATM IV ({_tenor_label(_t_1m)})",
        _fmt_pct(_atm_1m),
        help=f"Implied vol at 100% moneyness on the expiry closest to 1 month (T = {_t_1m:.3f} yr).",
    )
    m2.metric(
        f"ATM IV ({_tenor_label(_t_1y)})",
        _fmt_pct(_atm_1y),
        help=f"Implied vol at 100% moneyness on the expiry closest to 1 year (T = {_t_1y:.3f} yr).",
    )
    m3.metric(
        "Skew (90–110, 1M)",
        _fmt_diff(_skew_1m),
        help=(
            "IV(90%) − IV(110%) on the ~1M slice.  "
            "Proxy for the 25Δ risk reversal.  "
            "Positive = downside premium (equity put skew, the typical regime)."
        ),
    )
    m4.metric(
        "Butterfly (1M)",
        _fmt_diff(_bf_1m),
        help=(
            "½·(IV(90%) + IV(110%)) − IV(100%) on the ~1M slice.  "
            "Proxy for the 25Δ butterfly / smile convexity.  "
            "Positive = wings sit above ATM (the typical regime)."
        ),
    )

    st.divider()

    iv_sub3d, iv_sub_smile, iv_sub_ts, iv_sub_heat = st.tabs(
        ["3-D Surface", "Vol Smiles", "Term Structure", "Heatmap"]
    )

with iv_sub3d:
    show_raw = st.toggle("Show raw data points", value=True,
                         help="Overlay the actual Bloomberg quotes as white dots")
    st.plotly_chart(
        plot_surface_3d(
            grid,
            df=vol_df if show_raw else None,
            title=meta.get("title", "Implied Volatility Surface"),
        ),
        use_container_width=True,
    )
    st.caption(
        "**Bicubic spline interpolated** surface on a 60×60 grid.  "
        "Colour scale: green = low IV, red = high IV.  "
        "Axes: Moneyness K/F × 100 (%) · Time to Expiry (yr) · Implied Vol (%)."
    )

# ── Smile Slices ─────────────────────────────────────────────────────────────
with iv_sub_smile:
    # ── Expiry selector ──────────────────────────────────────────────────────
    st.markdown("**Select expiries to display**")

    # Initialise session state on very first run
    if "smile_sel" not in st.session_state:
        st.session_state["smile_sel"] = expiry_labels[:min(6, len(expiry_labels))]

    # Quick-select buttons — set the multiselect key directly then rerun
    btn_cols = st.columns([1, 1, 1, 1, 1])
    if btn_cols[0].button("All",               key="sel_all",   use_container_width=True):
        st.session_state["smile_sel"] = expiry_labels
        st.rerun()
    if btn_cols[1].button("None",              key="sel_none",  use_container_width=True):
        st.session_state["smile_sel"] = []
        st.rerun()
    if btn_cols[2].button(f"≤1M  ({len(_short)})",  key="sel_short", use_container_width=True):
        st.session_state["smile_sel"] = _short
        st.rerun()
    if btn_cols[3].button(f"1M–1Y  ({len(_medium)})", key="sel_med",  use_container_width=True):
        st.session_state["smile_sel"] = _medium
        st.rerun()
    if btn_cols[4].button(f">1Y  ({len(_long)})",  key="sel_long",  use_container_width=True):
        st.session_state["smile_sel"] = _long
        st.rerun()

    selected_expiries = st.multiselect(
        "Expiries",
        options=expiry_labels,
        default=st.session_state["smile_sel"],
        key="smile_sel",
        label_visibility="collapsed",
        help="Add / remove individual expiries, or use the quick-select buttons above",
    )

    st.divider()

    if not selected_expiries:
        st.warning("Select at least one expiry using the controls above.")
    else:
        show_svi = st.toggle(
            "Overlay SVI fit",
            value=False,
            help=(
                "Fits a Gatheral SVI (Stochastic Volatility Inspired) parametric smile "
                "w(x) = a + b·(ρ·(x−m) + √((x−m)²+σ²)) to each expiry slice and overlays "
                "the result as a dashed curve.  The 5 parameters are shown in the table below."
            ),
        )

        # Fit SVI for each selected expiry (cached via st.cache_data on file+params)
        svi_fits: dict | None = None
        if show_svi:
            svi_fits = {}
            for label in selected_expiries:
                s = vol_df[vol_df["expiry_label"] == label].sort_values("moneyness_pct")
                if len(s) < 5:
                    continue
                F = float(s["forward_price"].iloc[0])
                T = float(s["time_to_expiry"].iloc[0])
                strikes = (s["moneyness_pct"].values / 100.0) * F
                vols    = s["implied_vol"].values
                result  = svi_fit(strikes, vols, F, T)
                svi_fits[label] = {**result, "T": T, "F": F}

        st.plotly_chart(
            plot_smile_slices(
                vol_df, selected_expiries,
                k_min=strike_range[0], k_max=strike_range[1],
                svi_fits=svi_fits,
            ),
            use_container_width=True,
        )

        # SVI parameter table
        if show_svi and svi_fits:
            with st.expander("SVI parameters", expanded=True):
                st.caption(
                    "Model: **w(x) = a + b·(ρ·(x−m) + √((x−m)²+σ²))**  where "
                    "x = ln(K/F) and w = σ²_IV · T is total variance.\n\n"
                    "| Param | Interpretation |\n"
                    "|---|---|\n"
                    "| **a** | ATM total variance level |\n"
                    "| **b** | Slope of the wings (vol-of-vol proxy) |\n"
                    "| **ρ** | Skewness — negative for equity (put skew) |\n"
                    "| **m** | Horizontal shift of the smile |\n"
                    "| **σ** | Smile curvature / width |\n"
                    "| **residual** | Sum of squared errors vs raw quotes |"
                )
                rows = []
                for label, p in svi_fits.items():
                    rows.append({
                        "Expiry":    label,
                        "TTE (yr)":  round(p["T"], 4),
                        "a":         round(p["a"],     5),
                        "b":         round(p["b"],     5),
                        "ρ (skew)":  round(p["rho"],   4),
                        "m":         round(p["m"],     5),
                        "σ (curv.)": round(p["sigma"], 5),
                        "Residual":  f'{p["residual"]:.2e}',
                        "Converged": "✅" if p["success"] else "⚠️",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Per-slice SVI fit (Gatheral form) — same fitter used by the "
                    "local-vol pipeline as the source of analytical x-derivatives.  "
                    "Calendar-spread no-arbitrage is **not** enforced between slices: "
                    "any crossing of `w(·, T)` between adjacent expiries is detected "
                    "downstream by Dupire's denominator (`g ≤ 0`) and shown as a hole "
                    "in the LV surface."
                )

        with st.expander(f"Raw data — {len(selected_expiries)} expiry slice(s)"):
            show = (
                vol_df[vol_df["expiry_label"].isin(selected_expiries)]
                [["expiry_label", "moneyness_pct", "implied_vol", "forward_price", "time_to_expiry"]]
                .assign(implied_vol_pct=lambda d: (d["implied_vol"] * 100).round(2))
                .drop(columns="implied_vol")
                .rename(columns={
                    "expiry_label":   "Expiry",
                    "moneyness_pct":  "Moneyness (%)",
                    "implied_vol_pct":"IV (%)",
                    "forward_price":  "Forward",
                    "time_to_expiry": "TTE (yr)",
                })
            )
            st.dataframe(show, use_container_width=True)

# ── Term Structure ────────────────────────────────────────────────────────────
with iv_sub_ts:
    st.plotly_chart(plot_term_structure(vol_df), use_container_width=True)

    atm = (
        vol_df[vol_df["moneyness_pct"] == 100.0]
        .sort_values("time_to_expiry")
        .reset_index(drop=True)
    )
    

# ── Heatmap ───────────────────────────────────────────────────────────────────
with iv_sub_heat:
    st.plotly_chart(plot_heatmap(vol_df), use_container_width=True)
    st.caption("Colour = implied vol (%).  Rows sorted by ascending time-to-expiry.")

# ── Local Vol (Dupire) ────────────────────────────────────────────────────────
with tab_lv:
    st.subheader("Dupire Local Volatility Surface")

    # Build controls in a compact row
    lv_col1, lv_col2, lv_col3 = st.columns([1, 1, 2])
    n_lv_grid = lv_col1.select_slider(
        "Grid resolution",
        options=[50, 80, 100, 150],
        value=100,
        help="Number of points on each axis of the dense output grid",
    )
    lv_col2.write("")  # spacing

    lv_grid = None
    with st.spinner("Computing Dupire local volatility…"):
        try:
            lv_grid = _build_lv(file_bytes, snap_date.isoformat(), n_lv_grid)
        except Exception as exc:
            st.error(f"Local vol computation failed: {exc}")

    if lv_grid is None:
        st.info("Local vol surface could not be computed. Check the error above.")
    else:
        # Coverage stats
        n_total  = lv_grid.LV_grid.size
        n_nan    = int(np.isnan(lv_grid.LV_grid).sum())
        n_arb    = int(lv_grid.arb_mask.sum())
        lv_valid = lv_grid.LV_grid[~np.isnan(lv_grid.LV_grid)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min LV",          f"{lv_valid.min()*100:.1f}%" if len(lv_valid) else "—")
        m2.metric("Max LV",          f"{lv_valid.max()*100:.1f}%" if len(lv_valid) else "—")
        m3.metric("Arbitrage holes", f"{n_arb} / {n_total} pts",
                  help="Points where g≤0 or ∂w/∂T≤0 — shown as gaps in the surface")
        m4.metric("Valid coverage",  f"{(n_total-n_nan)/n_total*100:.0f}%")

        st.divider()

        # Sub-tabs inside the local vol tab
        lv_sub3d, lv_sub_slice, lv_sub_atm = st.tabs(
            ["3-D Local Vol", "LV vs IV Slice", "ATM Term Structure"]
        )

        with lv_sub3d:
            st.plotly_chart(
                plot_local_vol_3d(lv_grid, title="Dupire Local Volatility — SPX"),
                use_container_width=True,
            )
            st.caption(
                "White gaps = arbitrage violations (butterfly or calendar) where the "
                "Dupire formula is ill-defined.  These correspond to the violations "
                "flagged in the Arbitrage Checks tab."
            )

        with lv_sub_slice:
            dense_T = lv_grid.expiries
            raw_tte = vol_df.drop_duplicates("time_to_expiry").set_index("time_to_expiry")["expiry_label"]

            slice_options = {}
            for i, t in enumerate(dense_T[::max(1, len(dense_T)//20)]):
                nearest_label = raw_tte.iloc[np.abs(raw_tte.index.to_numpy() - t).argmin()]
                slice_options[f"T={t:.3f}yr  (~{nearest_label})"] = i * max(1, len(dense_T)//20)

            chosen_key = st.selectbox(
                "Select expiry slice",
                options=list(slice_options.keys()),
                index=min(5, len(slice_options)-1),
            )
            chosen_idx   = slice_options[chosen_key]
            chosen_label = chosen_key.split("(~")[1].rstrip(")")

            st.plotly_chart(
                plot_lv_vs_iv_slice(lv_grid, chosen_idx, expiry_label=chosen_label),
                use_container_width=True,
            )
            st.caption(
                "**Red solid** = Dupire local vol.  "
                "**Blue dashed** = implied vol from the interpolated surface.  "
                "**Orange dotted** = Derman–Kani approximation: IV_ATM + 2×(IV − IV_ATM).  "
                "The D–K result shows local vol skew is approximately **twice** the implied vol skew."
            )

        with lv_sub_atm:
            st.plotly_chart(
                plot_atm_term_structure(lv_grid, df_raw=vol_df),
                use_container_width=True,
            )
            st.caption(
                "**Crimson** = ATM local vol.  "
                "**Blue dashed** = ATM implied vol (interpolated).  "
                "**Open circles** = raw Bloomberg ATM quotes.  \n"
                "Where the IV term structure is **upward-sloping**, local vol > implied vol.  "
                "Where it is **flat or inverted**, local vol ≈ implied vol or falls below it."
            )

# ── MC Pricer ─────────────────────────────────────────────────────────────────
with tab_mc:
    st.subheader("Monte Carlo Option Pricer — Dupire Local Vol")

    # Build the local vol grid (cached; shares result with Local Vol tab if
    # the same resolution was already requested)
    _mc_lv_grid = None
    with st.spinner("Loading local vol surface…"):
        try:
            _mc_lv_grid = _build_lv(file_bytes, snap_date.isoformat(), 100)
        except Exception as _exc:
            st.error(f"Could not build local vol surface: {_exc}")

    if _mc_lv_grid is None:
        st.info("Local vol surface unavailable. Check the error above.")
    else:
        # ── Inputs ────────────────────────────────────────────────────────
        # Form rendering, barrier/OT sanity warnings and the curve / manual
        # `r` and `q` resolution all live in the shared helper near the top
        # of this file (see `_render_mc_pricer_form`).  This tab only adds
        # the LV-specific surface-coverage warnings and the Compute / display
        # block below.
        spot_default = float(meta.get("spot_price", 5000))
        _form = _render_mc_pricer_form(
            prefix="mc",
            spot_default=spot_default,
            T_max=float(_mc_lv_grid.expiries.max()),
            T_help_suffix=f"surface covers up to {_mc_lv_grid.expiries.max():.2f} yr",
            rate_curve=rate_curve,
            div_curve=div_curve,
            eps_help=(
                "Stop when SE(n)/|P(n)| < ε — a true 1-σ relative half-width on "
                "the MC estimator (running standard error from the antithetic "
                "pair-average variance).  Hard cap 200 000 paths; if the cap is "
                "reached the price is still the best unbiased estimate but the "
                "precision target was not met (typical for digital / one-touch "
                "payoffs at very tight ε)."
            ),
        )
        # Pull out the fields this tab actually consumes — keeps the pricing
        # call below readable without spreading `_form.x` everywhere.
        mc_S0                  = _form.S0
        mc_K                   = _form.K
        mc_T                   = _form.T
        mc_r                   = _form.r
        mc_q                   = _form.q
        mc_option_type         = _form.option_type
        mc_is_digital          = _form.is_digital
        mc_is_one_touch        = _form.is_one_touch
        mc_one_touch_direction = _form.one_touch_direction
        mc_digital_payout      = _form.digital_payout
        mc_barrier_type        = _form.barrier_type
        mc_barrier_level       = _form.barrier_level
        mc_barrier_style       = _form.barrier_style
        mc_steps               = _form.steps_per_year
        mc_eps                 = _form.eps

        # ── LV-specific surface-coverage warnings (after the shared form) ───
        # The helper handles barrier/OT placement vs spot; what's left is
        # the strike-moneyness and maturity-vs-surface-domain warning that
        # only the LV-MC tab cares about (the Heston engine doesn't grid).
        # Surface coverage warning (only meaningful when there is a strike)
        m_lo = float(_mc_lv_grid.moneyness.min())
        m_hi = float(_mc_lv_grid.moneyness.max())
        if not mc_is_one_touch:
            atm_moneyness = (mc_K / mc_S0) * 100.0
            if atm_moneyness < m_lo or atm_moneyness > m_hi:
                st.warning(
                    f"Strike moneyness {atm_moneyness:.1f}% is outside the surface range "
                    f"[{m_lo:.0f}%, {m_hi:.0f}%].  The boundary vol will be used at every step — "
                    "pricing accuracy will be reduced."
                )
        if mc_T > float(_mc_lv_grid.expiries.max()):
            st.warning(
                f"Maturity {mc_T:.2f} yr exceeds the surface max "
                f"{_mc_lv_grid.expiries.max():.2f} yr.  Time will be clamped at the boundary."
            )

        # (Barrier / one-touch placement-vs-spot warnings are emitted from
        # within `_render_mc_pricer_form`.)

        st.divider()

        # ── Run button ────────────────────────────────────────────────────
        if st.button("Compute Price", type="primary", use_container_width=False):
            with st.spinner("Running Monte Carlo simulation…"):
                _mc_result = price_european_option(
                    lv_grid        = _mc_lv_grid,
                    S0             = mc_S0,
                    K              = mc_K,
                    T              = mc_T,
                    r              = mc_r,
                    q              = mc_q,
                    option_type    = mc_option_type,
                    is_digital     = mc_is_digital,
                    digital_payout = mc_digital_payout,
                    barrier_type   = mc_barrier_type,
                    barrier_level  = mc_barrier_level,
                    barrier_style  = mc_barrier_style,
                    is_one_touch   = mc_is_one_touch,
                    one_touch_direction = mc_one_touch_direction,
                    steps_per_year = mc_steps,
                    eps            = mc_eps,
                )
            st.session_state["mc_result"] = _mc_result
            st.session_state["mc_params"] = {
                "option_type":    mc_option_type,
                "is_digital":     mc_is_digital,
                "is_one_touch":   mc_is_one_touch,
                "one_touch_direction": mc_one_touch_direction,
                "digital_payout": mc_digital_payout,
                "barrier_type":   mc_barrier_type,
                "barrier_level":  mc_barrier_level,
                "barrier_style":  mc_barrier_style,
                "S0": mc_S0, "K": mc_K, "T": mc_T,
                "r": mc_r, "q": mc_q,
                "steps": mc_steps, "eps": mc_eps,
            }

        # ── Results ───────────────────────────────────────────────────────
        if "mc_result" in st.session_state:
            res    = st.session_state["mc_result"]
            params = st.session_state["mc_params"]

            st.divider()
            st.markdown("#### Results")

            r1, r2, r3, r4 = st.columns(4)
            if params.get("is_one_touch"):
                _ot_dir = (params.get("one_touch_direction") or "up").capitalize()
                _price_label = f"{_ot_dir} One-Touch Price"
            else:
                _bt = params.get("barrier_type")
                _bt_label = {
                    None:        "",
                    "up_out":    "Up-and-Out ",
                    "up_in":     "Up-and-In ",
                    "down_out":  "Down-and-Out ",
                    "down_in":   "Down-and-In ",
                }.get(_bt, "")
                # Barrier monitoring style appears as a *suffix*, never as a
                # leading "(American) Call …" — that wording wrongly suggests
                # an American-exercise option.  This MC has no early-exercise
                # logic; "American" / "European" refer only to whether the
                # barrier is checked every Euler step or only at expiry.
                _bs_suffix = ""
                if _bt is not None:
                    _bs = params.get("barrier_style", "american").capitalize()
                    _bs_suffix = f" ({_bs}-monitored)"
                _digi = "Digital " if params.get("is_digital") else ""
                _price_label = f"{_bt_label}{_digi}{params.get('option_type', 'call').capitalize()} Price{_bs_suffix}"
            r1.metric(_price_label,     f"{res.price:.4f}")
            r2.metric("Std Error",      f"± {res.std_error:.4f}")
            r3.metric("Paths simulated", f"{res.n_paths:,}")
            r4.metric(
                "95% CI",
                f"[{res.price - 1.96*res.std_error:.4f},  {res.price + 1.96*res.std_error:.4f}]",
            )

            conv_badge = (
                '<span style="background:#d4edda;color:#155724;padding:2px 10px;'
                'border-radius:4px;font-weight:600;">Converged</span>'
                if res.converged else
                '<span style="background:#fff3cd;color:#856404;padding:2px 10px;'
                f'border-radius:4px;font-weight:600;">Cap reached ({res.n_paths:,} paths)</span>'
            )
            st.markdown(conv_badge, unsafe_allow_html=True)
            if not res.converged and res.price != 0.0:
                _ach_eps = res.std_error / abs(res.price)
                st.caption(
                    f"Achieved precision SE/|P| = {_ach_eps:.2%} vs requested ε = "
                    f"{params.get('eps', 0.001):.2%}.  For digital / one-touch payoffs "
                    "the variance is bounded below by the indicator's, so very tight ε "
                    "may need a control variate or many more paths."
                )

            if res.clamp_pct > 2.0:
                st.warning(
                    f"{res.clamp_pct:.1f}% of simulation steps were clamped to the surface "
                    "boundary.  Consider a strike or maturity closer to the surface range."
                )
            else:
                st.caption(f"Surface clamping: {res.clamp_pct:.2f}% of steps")

            with st.expander("Pricing parameters used"):
                st.json(params)


# ── Heston Pricer ────────────────────────────────────────────────────────────
with tab_heston:
    st.subheader("Heston Stochastic Volatility Model")
    st.caption(
        "dS = (r−q)S dt + √v S dW^S  ·  "
        "dv = κ(θ−v) dt + ξ√v dW^v  ·  "
        "Corr(dW^S, dW^v) = ρ"
    )

    # ── Calibration controls ─────────────────────────────────────────────
    # No `q` input here: under moneyness-quoted IVs, Heston/B-76 prices and
    # vegas are positively homogeneous of degree 1 in (F, K), so calibrated
    # parameters are independent of `q` (and of the level of `r`).  `q`
    # only matters where strikes are absolute — see the MC section below.
    _hcol1, _hcol2, _hcol3 = st.columns([1, 1, 1])
    with _hcol1:
        heston_max_slices = st.select_slider(
            "Calibration slices",
            options=[8, 10, 12, 15, 20],
            value=12,
            help="Number of expiry slices used in calibration (more = slower but better fit)",
        )
    with _hcol2:
        heston_min_T = st.number_input(
            "Min maturity (days)",
            min_value=1, max_value=90, value=18, step=1,
            help="Exclude options shorter than this — Heston struggles with ultra-short maturities",
            key="heston_min_T",
        ) / 365.25
    with _hcol3:
        heston_m_range = st.slider(
            "Moneyness range (%)",
            min_value=60, max_value=140, value=(70, 130), step=5,
            help="Calibration moneyness window — extreme wings add noise",
            key="heston_m_range",
        )

    spot_heston = float(meta.get("spot_price", 5000))

    if st.button("Calibrate Heston Model", type="primary", key="heston_calibrate_btn"):
        with st.spinner("Calibrating Heston model…"):
            _hcalib = heston_calibrate(
                vol_df, spot_heston, rate_curve, 0.0,
                max_slices=heston_max_slices,
                min_T=heston_min_T,
                m_range=(float(heston_m_range[0]), float(heston_m_range[1])),
            )
        st.session_state["heston_calib"]  = _hcalib
        st.session_state["heston_spot"]   = spot_heston
        st.session_state["heston_curve"]  = rate_curve

    # ── Show calibration results ─────────────────────────────────────────
    if "heston_calib" not in st.session_state:
        st.info(
            "Click **Calibrate Heston Model** to fit the five parameters "
            "(v₀, κ, θ, ξ, ρ) to the market implied-vol surface."
        )
    else:
        _hcalib = st.session_state["heston_calib"]
        _hp     = _hcalib.params

        # ── Calibrated parameters — custom HTML grid ────────────────────
        # Custom HTML rather than st.metric so the Greek symbols render
        # in their actual lowercase form (st.metric labels are forced to
        # uppercase by the bank-app CSS, which mangles κ ξ ρ θ).
        feller = 2 * _hp.kappa * _hp.theta > _hp.xi ** 2

        def _param(symbol: str, value: str, meta: str = "") -> str:
            meta_html = f'<div class="heston-param-meta">{meta}</div>' if meta else ""
            return (
                f'<div class="heston-param">'
                f'<div class="heston-param-symbol">{symbol}</div>'
                f'<div class="heston-param-value">{value}</div>'
                f'{meta_html}'
                f'</div>'
            )

        _half_life = np.log(2) / _hp.kappa
        _sig_0     = np.sqrt(_hp.v0) * 100
        _sig_inf   = np.sqrt(_hp.theta) * 100

        _feller_badge = (
            '<span class="heston-badge ok">Feller ✓</span>' if feller
            else '<span class="heston-badge warn">Feller ✗</span>'
        )
        _conv_badge = (
            '<span class="heston-badge ok">Converged</span>' if _hcalib.success
            else '<span class="heston-badge warn">Not converged</span>'
        )

        _result_html = (
            '<div class="heston-result">'
            '<div class="heston-param-grid">'
            + _param("v₀",  f"{_hp.v0:.4f}",     f"σ₀ = {_sig_0:.1f}%")
            + _param("κ",   f"{_hp.kappa:.2f}",   f"t½ = {_half_life:.2f} yr")
            + _param("θ",   f"{_hp.theta:.4f}",   f"σ∞ = {_sig_inf:.1f}%")
            + _param("ξ",   f"{_hp.xi:.3f}",      "vol of vol")
            + _param("ρ",   f"{_hp.rho:+.3f}",    "spot–var corr")
            + '</div>'
            '<div class="heston-quality-row">'
            f'<div class="heston-quality-stat" title="Root-mean-square IV residual on the calibrated region (filtered by min T and moneyness range).  Rule of thumb: <1 pp excellent · 1–3 pp typical · >3 pp poor.">'
            f'<div class="heston-quality-label">RMSE (IV)<span class="help-mark">?</span></div>'
            f'<div class="heston-quality-value">{_hcalib.rmse_iv*100:.2f} pp</div>'
            f'</div>'
            f'<div class="heston-quality-stat" title="Worst single-point absolute IV gap between model and market on the calibrated region.  Flags one-off mismatches that the RMSE can wash out.">'
            f'<div class="heston-quality-label">Max error<span class="help-mark">?</span></div>'
            f'<div class="heston-quality-value">{_hcalib.max_err_iv*100:.2f} pp</div>'
            f'</div>'
            f'<div class="heston-quality-stat" title="Number of (expiry × moneyness) market quotes the calibration evaluated, after applying the min-T and moneyness filters.">'
            f'<div class="heston-quality-label">Points<span class="help-mark">?</span></div>'
            f'<div class="heston-quality-value">{_hcalib.n_points}</div>'
            f'</div>'
            f'<div class="heston-badges">{_feller_badge}{_conv_badge}</div>'
            '</div>'
            '</div>'
        )
        st.markdown(_result_html, unsafe_allow_html=True)

        if not feller:
            st.caption(
                "⚠️ Feller condition violated (2κθ < ξ²): variance can touch zero.  "
                "MC uses full-truncation Euler which handles this correctly, but "
                "the variance paths will spend more time near zero."
            )

        st.divider()

        # ── Sub-tabs ─────────────────────────────────────────────────────
        # MC Pricer is the primary use of this tab once the user has run a
        # calibration; Smile Fits is the diagnostic view.  Listing MC first
        # makes it the default when the page loads.
        heston_sub_mc, heston_sub_calib = st.tabs(
            ["MC Pricer", "Smile Fits"]
        )

        # ── Smile Fits sub-tab ───────────────────────────────────────────
        with heston_sub_calib:
            _det = _hcalib.detail_df
            _det_expiries = sorted(
                _det["expiry_label"].unique(),
                key=lambda l: float(_det[_det["expiry_label"] == l]["T"].iloc[0]),
            )
            _n_show = min(6, len(_det_expiries))
            _show_idx = np.round(
                np.linspace(0, len(_det_expiries) - 1, _n_show)
            ).astype(int)
            _default_show = [_det_expiries[i] for i in _show_idx]

            _sel_exp = st.multiselect(
                "Select expiries to plot",
                options=_det_expiries,
                default=_default_show,
                key="heston_smile_sel",
            )

            if _sel_exp:
                _h_spot = st.session_state.get("heston_spot", spot_heston)
                _h_curve = st.session_state.get("heston_curve", rate_curve)
                m_dense = np.linspace(
                    float(vol_df["moneyness_pct"].min()),
                    float(vol_df["moneyness_pct"].max()),
                    50,
                )
                _colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                    "#bcbd22", "#17becf",
                ]
                fig_smile = go.Figure()
                for idx, label in enumerate(_sel_exp):
                    color = _colors[idx % len(_colors)]
                    sl = _det[_det["expiry_label"] == label]
                    T_val = float(sl["T"].iloc[0])

                    # Market dots
                    fig_smile.add_trace(go.Scatter(
                        x=sl["moneyness_pct"], y=sl["iv_market"] * 100,
                        mode="markers", marker=dict(size=8, color=color),
                        name=f"{label} (market)", legendgroup=label,
                    ))
                    # Heston smooth curve in moneyness — q is irrelevant here
                    # (moneyness-IV is invariant to F-scaling under the same
                    # homogeneity that makes calibration independent of q).
                    iv_heston = heston_smile(
                        _h_spot, T_val, _h_curve.zero_rate(T_val), 0.0, _hp, m_dense,
                    )
                    fig_smile.add_trace(go.Scatter(
                        x=m_dense, y=iv_heston * 100,
                        mode="lines", line=dict(color=color, dash="dash"),
                        name=f"{label} (Heston)", legendgroup=label,
                    ))

                fig_smile.update_layout(
                    title="Market vs Heston Implied-Vol Smiles",
                    xaxis_title="Moneyness (%)",
                    yaxis_title="Implied Vol (%)",
                    template="plotly_white",
                    height=500,
                    legend=dict(font_size=10),
                )
                st.plotly_chart(fig_smile, use_container_width=True)

            # Per-expiry summary
            with st.expander("Per-expiry fit quality"):
                _summary = (
                    _det.groupby("expiry_label")
                    .agg(
                        T=("T", "first"),
                        mean_abs_err=("error_pp", lambda x: np.mean(np.abs(x))),
                        max_abs_err=("error_pp", lambda x: np.max(np.abs(x))),
                        n=("error_pp", "count"),
                    )
                    .sort_values("T")
                    .reset_index()
                    .rename(columns={
                        "expiry_label": "Expiry",
                        "T": "TTE (yr)",
                        "mean_abs_err": "Mean |Error| (pp)",
                        "max_abs_err": "Max |Error| (pp)",
                        "n": "Points",
                    })
                )
                for _c in ["Mean |Error| (pp)", "Max |Error| (pp)", "TTE (yr)"]:
                    _summary[_c] = _summary[_c].round(2)
                st.dataframe(_summary, use_container_width=True, hide_index=True)

            with st.expander("Full detail table"):
                _disp = _det.copy()
                _disp["iv_market_%"] = (_disp["iv_market"] * 100).round(2)
                _disp["iv_model_%"]  = (_disp["iv_model"]  * 100).round(2)
                _disp["error_pp"]    = _disp["error_pp"].round(2)
                st.dataframe(
                    _disp[["expiry_label", "moneyness_pct", "T",
                           "iv_market_%", "iv_model_%", "error_pp"]],
                    use_container_width=True, hide_index=True,
                )

        # ── MC Pricer sub-tab ────────────────────────────────────────────
        with heston_sub_mc:
            # The form (payoff selector, option type / OT direction, barrier
            # controls, S₀ / K / T / r / q, steps, ε) plus barrier and
            # one-touch placement-vs-spot sanity warnings live in the shared
            # `_render_mc_pricer_form` helper near the top of this file.
            _h_form = _render_mc_pricer_form(
                prefix="heston",
                spot_default=spot_heston,
                T_max=5.0,
                T_help_suffix="capped at 5 yr to match the calibration domain",
                rate_curve=rate_curve,
                div_curve=div_curve,
                eps_help=(
                    "Stop when SE(n)/|P(n)| < ε — a true 1-σ relative half-width on "
                    "the MC estimator (running standard error from the antithetic "
                    "pair-average variance).  Hard cap 400 000 paths; if the cap is "
                    "reached the price is still the best unbiased estimate but the "
                    "precision target was not met (typical for digital / one-touch "
                    "payoffs at very tight ε)."
                ),
            )
            h_S0                  = _h_form.S0
            h_K                   = _h_form.K
            h_T                   = _h_form.T
            h_mc_r                = _h_form.r
            h_mc_q                = _h_form.q
            h_option_type         = _h_form.option_type
            h_is_digital          = _h_form.is_digital
            h_is_one_touch        = _h_form.is_one_touch
            h_one_touch_direction = _h_form.one_touch_direction
            h_digital_payout      = _h_form.digital_payout
            h_barrier_type        = _h_form.barrier_type
            h_barrier_level       = _h_form.barrier_level
            h_barrier_style       = _h_form.barrier_style
            h_steps               = _h_form.steps_per_year
            h_eps                 = _h_form.eps

            st.divider()

            if st.button("Compute Price", type="primary", key="heston_price_btn"):
                _h_r_T = h_mc_r
                with st.spinner("Running Heston Monte Carlo…"):
                    _h_mc_result = mc_heston(
                        params=_hp,
                        S0=h_S0, K=h_K, T=h_T,
                        r=_h_r_T, q=h_mc_q,
                        option_type=h_option_type,
                        is_digital=h_is_digital,
                        digital_payout=h_digital_payout,
                        barrier_type=h_barrier_type,
                        barrier_level=h_barrier_level,
                        barrier_style=h_barrier_style,
                        is_one_touch=h_is_one_touch,
                        one_touch_direction=h_one_touch_direction,
                        steps_per_year=h_steps,
                        eps=h_eps,
                    )
                st.session_state["heston_mc_result"] = _h_mc_result
                st.session_state["heston_mc_params"] = {
                    "option_type": h_option_type,
                    "is_digital": h_is_digital,
                    "is_one_touch": h_is_one_touch,
                    "one_touch_direction": h_one_touch_direction,
                    "digital_payout": h_digital_payout,
                    "barrier_type": h_barrier_type,
                    "barrier_level": h_barrier_level,
                    "barrier_style": h_barrier_style,
                    "S0": h_S0, "K": h_K, "T": h_T,
                    "r": _h_r_T, "q": h_mc_q,
                    "steps": h_steps, "eps": h_eps,
                }
                # Analytic reference only for vanilla Europeans (no closed
                # form for OT under Heston in this codebase)
                if not h_is_digital and not h_is_one_touch and h_barrier_type is None:
                    _h_analytic = heston_call(
                        h_S0, h_K, h_T, _h_r_T, h_mc_q, *_hp,
                    )
                    if h_option_type == "put":
                        _h_analytic = (
                            _h_analytic
                            - h_S0 * np.exp(-h_mc_q * h_T)
                            + h_K * np.exp(-_h_r_T * h_T)
                        )
                    st.session_state["heston_analytic"] = float(_h_analytic)
                else:
                    st.session_state["heston_analytic"] = None

            # ── Results ──────────────────────────────────────────────
            if "heston_mc_result" in st.session_state:
                _hres    = st.session_state["heston_mc_result"]
                _hparams = st.session_state["heston_mc_params"]

                st.divider()
                st.markdown("#### Results")

                if _hparams.get("is_one_touch"):
                    _h_ot_dir = (_hparams.get("one_touch_direction") or "up").capitalize()
                    _price_label = f"{_h_ot_dir} One-Touch Price"
                else:
                    _bt = _hparams.get("barrier_type")
                    _bt_label = {
                        "up_out": "Up-and-Out ", "up_in": "Up-and-In ",
                        "down_out": "Down-and-Out ", "down_in": "Down-and-In ",
                    }.get(_bt, "")
                    # Barrier monitoring as a suffix, not a leading "(American)
                    # Call …" — that would imply an American-exercise option,
                    # which this MC does not implement (no early exercise logic).
                    _bs_suffix = ""
                    if _bt:
                        _bs = _hparams.get("barrier_style", "american").capitalize()
                        _bs_suffix = f" ({_bs}-monitored)"
                    _digi = "Digital " if _hparams.get("is_digital") else ""
                    _price_label = (
                        f"{_bt_label}{_digi}"
                        f"{_hparams.get('option_type','call').capitalize()} Price"
                        f"{_bs_suffix}"
                    )

                r1, r2, r3, r4 = st.columns(4)
                r1.metric(f"MC {_price_label}", f"{_hres.price:.4f}")
                r2.metric("Std Error", f"± {_hres.std_error:.4f}")
                r3.metric("Paths simulated", f"{_hres.n_paths:,}")
                r4.metric(
                    "95% CI",
                    f"[{_hres.price - 1.96*_hres.std_error:.4f},  "
                    f"{_hres.price + 1.96*_hres.std_error:.4f}]",
                )

                # Semi-analytic reference
                _h_analytic = st.session_state.get("heston_analytic")
                if _h_analytic is not None:
                    _an_col, _diff_col, _ = st.columns([1, 1, 2])
                    _an_col.metric(
                        "Semi-analytic (Fourier)",
                        f"{_h_analytic:.4f}",
                        help="Heston closed-form via characteristic-function inversion — no MC noise",
                    )
                    _diff_col.metric(
                        "MC − Analytic",
                        f"{_hres.price - _h_analytic:+.4f}",
                        help="Difference due to MC discretisation error and sampling noise",
                    )

                _hconv = (
                    '<span style="background:#d4edda;color:#155724;padding:2px 10px;'
                    'border-radius:4px;font-weight:600;">Converged</span>'
                    if _hres.converged else
                    '<span style="background:#fff3cd;color:#856404;padding:2px 10px;'
                    f'border-radius:4px;font-weight:600;">Cap reached ({_hres.n_paths:,} paths)</span>'
                )
                st.markdown(_hconv, unsafe_allow_html=True)
                if not _hres.converged and _hres.price != 0.0:
                    _h_ach_eps = _hres.std_error / abs(_hres.price)
                    st.caption(
                        f"Achieved precision SE/|P| = {_h_ach_eps:.2%} vs requested ε = "
                        f"{_hparams.get('eps', 0.001):.2%}.  For digital / one-touch "
                        "payoffs the variance is bounded below by the indicator's, so "
                        "very tight ε may need a control variate or many more paths."
                    )

                with st.expander("Pricing parameters used"):
                    st.json(_hparams)


# ── Arbitrage Checks ──────────────────────────────────────────────────────────
with tab_arb:
    st.subheader("Arbitrage-Free Checks")
    # Three independent no-arbitrage checks run each time the risk-free rate
    # number input is changed.

    with st.expander("ℹ️  Why does Bloomberg data show violations?", expanded=False):
        st.markdown(
            """
Bloomberg OVME implied vol quotes are **raw market mid-prices**, not an
arbitrage-free model fit.  Violations in real data have well-understood causes:

| Check | Typical cause in SPX data |
|---|---|
| **Calendar spread** | Very short-dated options near a macro event (FOMC, CPI) can carry an event premium that compresses as the event passes, causing total variance to decrease across consecutive weekly expiries. |
| **Butterfly spread** | For 1–3 day expiries, market-makers widen quotes aggressively around specific strikes due to event risk.  The resulting non-smooth smile breaks call-price convexity (`d²C/dK² ≥ 0`, Breeden–Litzenberger). |
| **Vertical spread** | Rare in liquid markets; if flagged it usually signals a data extraction artefact or a crossed market. |

These violations confirm the checks are **working correctly** — they are
detecting genuine anomalies in the raw surface, not code errors.  The
local-vol pipeline fits Gatheral SVI per expiry (matching Bloomberg
OVME), so per-slice noise is smoothed inside each smile but cross-slice
arbitrages are not actively repaired — they show up as `g ≤ 0` holes in
the LV surface, which the MC pricer fills by linear interpolation in T.
            """
        )

    with st.spinner("Running checks…"):
        checks = _run_checks(
            file_bytes,
            snap_date.isoformat(),
            tuple(rate_curve.tenors_yr),
            tuple(rate_curve.zero_rates),
        )
    all_violations = {
        "calendar":  checks[0].violations,
        "butterfly": checks[1].violations,
        "vertical":  checks[2].violations,
    }
    n_total = sum(len(c.violations) for c in checks)

    # Summary row
    cols = st.columns(3)
    for col, check in zip(cols, checks):
        with col:
            st.markdown(f"### {check.name}", unsafe_allow_html=True)
            st.markdown(_badge(check.passed), unsafe_allow_html=True)
            st.caption(check.details)

    st.divider()

    # Per-check detail expanders
    for check in checks:
        n = len(check.violations)
        status = "PASS" if check.passed else "FAIL"
        with st.expander(
            f"{check.name} — {status} — {n} violation(s)",
            expanded=(not check.passed),
        ):
            if check.violations:
                vdf = pd.DataFrame(check.violations).copy()
                for col in vdf.select_dtypes("float").columns:
                    vdf[col] = vdf[col].round(4)
                styled = vdf.style.set_properties(**{
                    "background-color": "#f8d7da",
                    "color": "#721c24",
                })
                st.dataframe(styled, use_container_width=True)
            else:
                st.success("No violations detected.")

    st.divider()

    # Smile with overlaid flags
    st.subheader("Violation Overlay — Smile Chart")
    flag_expiry = st.selectbox(
        "Select expiry to inspect",
        options=expiry_labels,
        index=0,
        key="flag_expiry",
    )
    st.plotly_chart(
        plot_arbitrage_flags(
            vol_df, all_violations, flag_expiry,
            k_min=strike_range[0], k_max=strike_range[1],
        ),
        use_container_width=True,
    )
    st.caption(
        "**×** = calendar spread violation  |  "
        "**◆** = butterfly violation  |  "
        "**dashed line** = vertical spread violation (call prices inverted)"
    )

    st.divider()

    # Download
    if n_total > 0:
        xlsx_bytes = _export_violations(checks)
        st.download_button(
            label="Download Violation Report (.xlsx)",
            data=xlsx_bytes,
            file_name="arbitrage_violations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="One sheet per check type, listing every violating (K, T) pair",
        )
    else:
        st.success("Surface is arbitrage-free across all three checks with the current parameters.")
