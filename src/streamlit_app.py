"""
streamlit_app.py — UI entry-point for the Volatility Surface Explorer.

Run with:
    uv run streamlit run src/streamlit_app.py

UI layout
---------
Sidebar  : daily file selector (data_vol_surface/) · file uploader · snap-date
           · risk-free rate · moneyness range slider
Main     : metric cards · 4 tabs (IV Surface [3-D / Smiles / Term Structure /
           Heatmap] / Local Volatility / MC Pricer / Arbitrage Checks)
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
import streamlit as st

from arbitrage_checks import run_all_checks, CheckResult
from data_loader import load_workbook, VolDataset
from local_vol import build_local_vol, LocalVolGrid
from montecarlo import price_european_option, MCResult
from iv_surface_builder import svi_fit, build_surface, SurfaceGrid
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
    page_title="Vol Surface Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Tab list — children grow equally to fill the full horizontal width */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid #e0e0e0;
    }
    /* Each tab button — flex:1 forces equal width */
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0;
        justify-content: center;
        font-size: 1.1rem;
        font-weight: 500;
        padding: 16px 12px;
        border-right: 1px solid #e0e0e0;
        color: #555;
        letter-spacing: 0.02em;
    }
    .stTabs [data-baseweb="tab"]:first-child {
        border-left: 1px solid #e0e0e0;
    }
    /* Nested (sub) tabs — revert to natural size / left-aligned layout */
    .stTabs .stTabs [data-baseweb="tab"] {
        flex: 0 0 auto;
        justify-content: flex-start;
        font-size: 0.95rem;
        padding: 10px 28px;
    }
    /* Active tab */
    .stTabs [aria-selected="true"] {
        font-weight: 700;
        color: #0f52ba;
        background-color: #f0f4ff;
        border-bottom: 3px solid #0f52ba;
    }
    /* Hover */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f5f5f5;
        color: #0f52ba;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
def _run_checks(file_bytes: bytes, snap_date_iso: str, r: float) -> list[CheckResult]:
    """Cache keyed on the raw file + snap date + rate — avoids JSON round-trip."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        from datetime import date as _date
        ds = load_workbook(tmp_path, snap_date=_date.fromisoformat(snap_date_iso))
        return run_all_checks(ds.vol_df, r=r)
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

    interp_method = "spline"

    st.divider()
    st.subheader("Pricing / Arbitrage")
    risk_free_rate = st.number_input(
        "Risk-free rate r (%)",
        min_value=0.0, max_value=20.0, value=4.500, step=0.001, format="%.3f",
        help="Used in Black-76 call prices for the vertical spread check",
    ) / 100.0

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
# Header metrics
# ---------------------------------------------------------------------------
st.title("📈 Volatility Surface Explorer")
st.caption(
    f"{meta.get('title', '')}  |  Source: {source_label}  |  "
    f"Snap date: {meta.get('snap_date', '—')}"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ticker",      meta.get("ticker",    "—"))
c2.metric("Spot",  f"{meta.get('spot_price', 0):,.0f}")
c3.metric("Expiries",    str(meta.get("n_expiries", 0)))
c4.metric("Strikes",     str(meta.get("n_strikes",  0)))
c5.metric("Interpolation", "Bicubic Spline")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_iv, tab_lv, tab_mc, tab_arb = st.tabs(
    ["IV Surface", "Local Volatility", "MC Pricer", "⚠️ Arbitrage Checks"]
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
                    "⚠️ Note: this implementation uses simple parameter bounds and does **not** "
                    "enforce the full Gatheral no-static-arbitrage conditions (Lee moment formula, "
                    "durrleman condition).  For a fully arbitrage-free surface, consider SSVI or "
                    "a constrained SVI calibration."
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
        spot_default   = float(meta.get("spot_price", 5000))

        # ── Payoff selector ───────────────────────────────────────────────
        _payoff_col, _digi_col = st.columns([2, 1])
        with _payoff_col:
            _type_choice = st.segmented_control(
                "Option type",
                options=["Call", "Put"],
                default="Call",
                selection_mode="single",
                key="mc_type_seg",
            )
            mc_option_type = (_type_choice or "Call").lower()
        with _digi_col:
            mc_is_digital = st.toggle(
                "Digital (cash-or-nothing)",
                value=False,
                key="mc_digital_toggle",
                help=(
                    "Off → vanilla payoff max(S_T−K, 0) / max(K−S_T, 0).  "
                    "On  → pays a fixed cash amount if ITM at expiry, else 0."
                ),
            )

        mc_digital_payout = 1.0
        if mc_is_digital:
            mc_digital_payout = st.number_input(
                "Cash payout",
                min_value=0.0001, value=1.0, step=0.5, format="%.4f",
                help="Cash amount paid if S_T is ITM at expiry.  Common conventions: 1.0 (unit), or the option notional.",
            )

        # ── Barrier selector (mutually exclusive with Digital) ────────────
        # A digital pays a fixed cash amount if ITM at expiry — the barrier
        # concept does not apply, so we hide the barrier controls in that case.
        mc_barrier_type  = None
        mc_barrier_level = None
        mc_barrier_style = "american"

        if mc_is_digital:
            st.caption(
                "Barrier options are disabled for digitals."
            )
        else:
            _BARRIER_LABELS = {
                "None":          None,
                "Up-and-Out":    "up_out",
                "Up-and-In":     "up_in",
                "Down-and-Out":  "down_out",
                "Down-and-In":   "down_in",
            }
            _barrier_choice = st.segmented_control(
                "Barrier",
                options=list(_BARRIER_LABELS.keys()),
                default="None",
                selection_mode="single",
                key="mc_barrier_seg",
                help=(
                    "Up = barrier above spot, Down = below.  "
                    "Out = option dies if barrier is touched.  "
                    "In = option only activates once barrier is touched."
                ),
            )
            mc_barrier_type = _BARRIER_LABELS[_barrier_choice or "None"]

            if mc_barrier_type is not None:
                _bcol1, _bcol2 = st.columns([1, 1])
                with _bcol1:
                    _is_up = mc_barrier_type.startswith("up")
                    _default_B = spot_default * (1.10 if _is_up else 0.90)
                    mc_barrier_level = st.number_input(
                        "Barrier level B",
                        min_value=0.01, value=float(_default_B), step=10.0,
                        help=(
                            "Up barriers require B > S₀; down barriers require B < S₀.  "
                            "An option already past its barrier at inception is degenerate "
                            "(knock-out → 0, knock-in → vanilla)."
                        ),
                    )
                with _bcol2:
                    _monitor_choice = st.segmented_control(
                        "Barrier monitoring",
                        options=["American", "European"],
                        default="American",
                        selection_mode="single",
                        key="mc_monitor_seg",
                        help=(
                            "American: barrier monitored at every Euler step throughout the life.  "
                            "European: barrier checked only at expiry — S_T alone decides hit / no-hit."
                        ),
                    )
                    mc_barrier_style = (_monitor_choice or "American").lower()

        mc_col1, mc_col2, mc_col3 = st.columns(3)

        with mc_col1:
            mc_S0 = st.number_input(
                "Spot S₀",
                min_value=1.0, value=spot_default, step=10.0,
                help="Current spot price (pre-filled from the loaded dataset)",
            )
            mc_q = st.number_input(
                "Dividend yield q (%)",
                min_value=0.000, max_value=20.0, value=0.000, step=0.001, format="%.3f",
                help="Continuous dividend yield in %",
            ) / 100.0

        with mc_col2:
            mc_K = st.number_input(
                "Strike K",
                min_value=1.0, value=spot_default, step=10.0,
                help="Option strike (default = ATM)",
            )
            mc_steps = st.selectbox(
                "Steps per year",
                options=[252, 52],
                index=0,
                format_func=lambda x: f"{x}  ({'daily' if x == 252 else 'weekly'})",
                help="Number of Euler time steps per year",
            )

        with mc_col3:
            mc_T = st.number_input(
                "Maturity T (years)",
                min_value=0.001, max_value=float(_mc_lv_grid.expiries.max()),
                value=1.000, step=0.0001, format="%.4f",
                help=(
                    f"Option maturity in years (surface covers up to {_mc_lv_grid.expiries.max():.2f} yr).  "
                    "Any 4-decimal value is accepted — e.g. 0.2500 (3 m), 0.1781 (65 d)."
                ),
            )
            mc_eps = st.number_input(
                "Convergence ε",
                min_value=1e-6, max_value=0.05,
                value=0.001, step=0.0005, format="%.4f",
                help="Stop when |P(n) − P(n−500)| / P(n) < ε  (relative threshold)",
            )

        # Surface coverage warning
        atm_moneyness = (mc_K / mc_S0) * 100.0
        m_lo = float(_mc_lv_grid.moneyness.min())
        m_hi = float(_mc_lv_grid.moneyness.max())
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

        # Barrier sanity check
        if mc_barrier_type is not None:
            if mc_barrier_type.startswith("up") and mc_barrier_level <= mc_S0:
                st.warning(
                    f"Up barrier B={mc_barrier_level:g} is not above spot S₀={mc_S0:g}. "
                    "The option is knocked out / activated at inception."
                )
            elif mc_barrier_type.startswith("down") and mc_barrier_level >= mc_S0:
                st.warning(
                    f"Down barrier B={mc_barrier_level:g} is not below spot S₀={mc_S0:g}. "
                    "The option is knocked out / activated at inception."
                )

        st.divider()

        # ── Run button ────────────────────────────────────────────────────
        if st.button("Compute Price", type="primary", use_container_width=False):
            with st.spinner("Running Monte Carlo simulation…"):
                _mc_result = price_european_option(
                    lv_grid        = _mc_lv_grid,
                    S0             = mc_S0,
                    K              = mc_K,
                    T              = mc_T,
                    r              = risk_free_rate,
                    q              = mc_q,
                    option_type    = mc_option_type,
                    is_digital     = mc_is_digital,
                    digital_payout = mc_digital_payout,
                    barrier_type   = mc_barrier_type,
                    barrier_level  = mc_barrier_level,
                    barrier_style  = mc_barrier_style,
                    steps_per_year = mc_steps,
                    eps            = mc_eps,
                )
            st.session_state["mc_result"] = _mc_result
            st.session_state["mc_params"] = {
                "option_type":    mc_option_type,
                "is_digital":     mc_is_digital,
                "digital_payout": mc_digital_payout,
                "barrier_type":   mc_barrier_type,
                "barrier_level":  mc_barrier_level,
                "barrier_style":  mc_barrier_style,
                "S0": mc_S0, "K": mc_K, "T": mc_T,
                "r": risk_free_rate, "q": mc_q,
                "steps": mc_steps, "eps": mc_eps,
            }

        # ── Results ───────────────────────────────────────────────────────
        if "mc_result" in st.session_state:
            res    = st.session_state["mc_result"]
            params = st.session_state["mc_params"]

            st.divider()
            st.markdown("#### Results")

            r1, r2, r3, r4 = st.columns(4)
            _bt = params.get("barrier_type")
            _bt_label = {
                None:        "",
                "up_out":    "Up-and-Out ",
                "up_in":     "Up-and-In ",
                "down_out":  "Down-and-Out ",
                "down_in":   "Down-and-In ",
            }.get(_bt, "")
            _bs_label = ""
            if _bt is not None:
                _bs_label = f"({params.get('barrier_style', 'american').capitalize()}) "
            _digi = "Digital " if params.get("is_digital") else ""
            _price_label = f"{_bs_label}{_bt_label}{_digi}{params.get('option_type', 'call').capitalize()} Price"
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
                'border-radius:4px;font-weight:600;">Cap reached (50 000 paths)</span>'
            )
            st.markdown(conv_badge, unsafe_allow_html=True)

            if res.clamp_pct > 2.0:
                st.warning(
                    f"{res.clamp_pct:.1f}% of simulation steps were clamped to the surface "
                    "boundary.  Consider a strike or maturity closer to the surface range."
                )
            else:
                st.caption(f"Surface clamping: {res.clamp_pct:.2f}% of steps")

            with st.expander("Pricing parameters used"):
                st.json(params)


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
| **Butterfly spread** | For 1–3 day expiries, market-makers widen quotes aggressively around specific strikes due to event risk.  The resulting non-smooth smile can breach local convexity. |
| **Vertical spread** | Rare in liquid markets; if flagged it usually signals a data extraction artefact or a crossed market. |

These violations confirm the checks are **working correctly** — they are
detecting genuine anomalies in the raw surface, not code errors.  An
arbitrage-free surface model (e.g. SVI, SABR) would fit a smooth surface
through the data that satisfies all three conditions by construction.
            """
        )

    with st.spinner("Running checks…"):
        checks = _run_checks(file_bytes, snap_date.isoformat(), risk_free_rate)
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
