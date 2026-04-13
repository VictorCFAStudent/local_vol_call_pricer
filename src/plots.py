"""
plots.py — All chart generation functions.

Every function returns a Plotly Figure so Streamlit can render it
interactively.  Vols are converted back to percentage points for display.

Charts produced:
  · plot_surface_3d       — 3-D colour-map surface of IV(K, T)
  · plot_smile_slices     — IV vs moneyness for selected expiries
  · plot_term_structure   — ATM IV vs time-to-expiry
  · plot_heatmap          — IV as annotated colour-coded grid
  · plot_arbitrage_flags  — Smile with violation markers overlaid
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from iv_surface_builder import SurfaceGrid
from local_vol import LocalVolGrid

_PALETTE = px.colors.qualitative.Plotly


# ---------------------------------------------------------------------------
def plot_surface_3d(
    grid: SurfaceGrid,
    df: pd.DataFrame | None = None,
    title: str = "Implied Volatility Surface",
) -> go.Figure:
    """Colour-map 3-D surface of IV(moneyness, T).

    Parameters
    ----------
    grid : interpolated surface grid
    df   : original long-format DataFrame — when supplied, raw data points
           are overlaid as scatter markers so the user can see actual quotes
    """
    IV_pct = grid.IV_grid * 100

    # Use actual expiry dates on the time axis by mapping T values to labels.
    # We build a custom tick system based on the raw expiry dates.
    tickvals, ticktext = [], []
    if df is not None:
        raw_exp = (
            df.drop_duplicates("time_to_expiry")
            .sort_values("time_to_expiry")[["time_to_expiry", "expiry_label"]]
            .reset_index(drop=True)
        )
        # Pick ~7 ticks evenly spaced in log(T) so we get good coverage of
        # both the crowded front end and the long end without label overlap.
        log_t = np.log(raw_exp["time_to_expiry"].values)
        targets = np.linspace(log_t.min(), log_t.max(), 7)
        chosen_idx: set[int] = set()
        for t in targets:
            chosen_idx.add(int(np.abs(log_t - t).argmin()))
        for idx in sorted(chosen_idx):
            tickvals.append(raw_exp.loc[idx, "time_to_expiry"])
            ticktext.append(raw_exp.loc[idx, "expiry_label"])

    fig = go.Figure()

    # Main interpolated surface
    fig.add_trace(
        go.Surface(
            x=grid.K_grid,
            y=grid.T_grid,
            z=IV_pct,
            colorscale="RdYlGn_r",
            colorbar=dict(title="IV (%)", thickness=14, len=0.7),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white",
                       project=dict(z=True)),
            ),
            hovertemplate=(
                "Moneyness: %{x:.1f}%<br>"
                "TTE: %{y:.3f} yr<br>"
                "IV: %{z:.2f}%<extra></extra>"
            ),
            name="Surface",
        )
    )

    # Raw data points — coloured by IV value to match the surface colorscale
    if df is not None:
        iv_vals = df["implied_vol"] * 100
        fig.add_trace(
            go.Scatter3d(
                x=df["moneyness_pct"],
                y=df["time_to_expiry"],
                z=iv_vals,
                mode="markers",
                marker=dict(
                    size=5,
                    color=iv_vals,
                    colorscale="RdYlGn_r",
                    cmin=float(IV_pct.min()),
                    cmax=float(IV_pct.max()),
                    opacity=1.0,
                    line=dict(width=0.5, color="black"),
                    showscale=False,   # reuse the surface colorbar
                ),
                customdata=df[["expiry_label", "moneyness_pct"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Moneyness: %{customdata[1]:.1f}%<br>"
                    "IV: %{z:.2f}%<extra>Bloomberg quote</extra>"
                ),
                name="Bloomberg quotes",
                showlegend=True,
            )
        )

    yaxis_cfg = dict(title="Time to Expiry (yr)")
    if tickvals:
        yaxis_cfg.update(tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(title="Moneyness (%)"),
            yaxis=yaxis_cfg,
            zaxis=dict(title="Implied Vol (%)"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.7)),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.8),
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
def plot_smile_slices(
    df: pd.DataFrame,
    selected_labels: list[str],
    k_min: float = 80.0,
    k_max: float = 120.0,
    svi_fits: dict[str, dict] | None = None,
) -> go.Figure:
    """IV vs moneyness for a set of selected expiries.

    Parameters
    ----------
    svi_fits : optional dict  {expiry_label -> svi_fit() result dict}
               When supplied, a dashed SVI curve is overlaid for each expiry.
    """
    mask = (
        df["expiry_label"].isin(selected_labels)
        & df["moneyness_pct"].between(k_min, k_max)
    )
    sub = df[mask].copy()

    fig = go.Figure()
    m_dense = np.linspace(k_min, k_max, 200)
    x_dense = np.log(m_dense / 100.0)   # log-moneyness for SVI evaluation

    for idx, label in enumerate(selected_labels):
        s = sub[sub["expiry_label"] == label].sort_values("moneyness_pct")
        if s.empty:
            continue
        color = _PALETTE[idx % len(_PALETTE)]
        T = float(s["time_to_expiry"].iloc[0])

        # Raw Bloomberg quotes
        fig.add_trace(
            go.Scatter(
                x=s["moneyness_pct"],
                y=s["implied_vol"] * 100,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                line=dict(color=color, width=2),
                marker=dict(size=7),
                hovertemplate=(
                    "Moneyness: %{x:.1f}%<br>IV: %{y:.2f}%"
                    f"<extra>{label}</extra>"
                ),
            )
        )

        # SVI fitted curve (dashed, same colour)
        if svi_fits and label in svi_fits:
            p = svi_fits[label]
            if p.get("success"):
                a, b, rho, m, sigma = p["a"], p["b"], p["rho"], p["m"], p["sigma"]
                w_fit = a + b * (rho * (x_dense - m) + np.sqrt((x_dense - m) ** 2 + sigma ** 2))
                w_fit = np.clip(w_fit, 1e-8, None)
                iv_fit = np.sqrt(w_fit / T) * 100   # total variance → IV %
                fig.add_trace(
                    go.Scatter(
                        x=m_dense,
                        y=iv_fit,
                        mode="lines",
                        name=f"{label} SVI",
                        legendgroup=label,
                        showlegend=True,
                        line=dict(color=color, width=1.5, dash="dash"),
                        hovertemplate=(
                            "SVI fit<br>Moneyness: %{x:.1f}%<br>IV: %{y:.2f}%"
                            f"<extra>{label} SVI</extra>"
                        ),
                    )
                )

    fig.add_vline(x=100.0, line_dash="dash", line_color="grey",
                  annotation_text="ATM", annotation_position="top right")
    fig.update_layout(
        title="Volatility Smile Slices" + (" + SVI fits" if svi_fits else ""),
        xaxis_title="Moneyness — K/F × 100 (%)",
        yaxis_title="Implied Vol (%)",
        legend_title="Expiry",
        hovermode="x unified",
        height=520,
    )
    return fig


# ---------------------------------------------------------------------------
def plot_term_structure(df: pd.DataFrame) -> go.Figure:
    """ATM implied vol vs time-to-expiry (term structure)."""
    atm = (
        df[df["moneyness_pct"] == 100.0]
        .sort_values("time_to_expiry")
        .drop_duplicates("time_to_expiry")
    )
    if atm.empty:
        # Fallback: pick closest available moneyness to 100 per expiry
        df2 = df.copy()
        df2["_dist"] = (df2["moneyness_pct"] - 100.0).abs()
        atm = (
            df2.sort_values(["time_to_expiry", "_dist"])
            .groupby("time_to_expiry", as_index=False)
            .first()
            .sort_values("time_to_expiry")
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=atm["time_to_expiry"],
            y=atm["implied_vol"] * 100,
            mode="lines+markers",
            name="ATM IV",
            line=dict(color="royalblue", width=2.5),
            marker=dict(size=8, symbol="circle"),
            customdata=atm[["expiry_label"]].values,
            hovertemplate=(
                "%{customdata[0]}<br>"
                "TTE: %{x:.3f} yr<br>"
                "ATM IV: %{y:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="ATM Volatility Term Structure",
        xaxis=dict(title="Time to Expiry (years)", type="log"),
        yaxis_title="ATM Implied Vol (%)",
        height=460,
    )
    return fig


# ---------------------------------------------------------------------------
def plot_heatmap(df: pd.DataFrame) -> go.Figure:
    """IV matrix as a colour-coded grid (expiry × moneyness)."""
    pivot = (
        df.pivot_table(
            index="expiry_label",
            columns="moneyness_pct",
            values="implied_vol",
            aggfunc="mean",
        )
        * 100
    )

    # Sort rows by ascending time-to-expiry
    tte_order = (
        df.drop_duplicates("expiry_label")
        .sort_values("time_to_expiry")["expiry_label"]
        .tolist()
    )
    pivot = pivot.reindex(tte_order)

    text = np.round(pivot.values, 1).astype(str)
    text[text == "nan"] = ""

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{c:.1f}%" for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            colorbar=dict(title="IV (%)"),
            text=text,
            texttemplate="%{text}",
            hoverongaps=False,
            hovertemplate=(
                "Moneyness: %{x}<br>Expiry: %{y}<br>IV: %{z:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Implied Volatility Heatmap",
        xaxis_title="Moneyness (%)",
        yaxis_title="Expiry",
        height=max(500, 20 * len(pivot)),
        margin=dict(l=120),
    )
    return fig


# ---------------------------------------------------------------------------
def plot_arbitrage_flags(
    df: pd.DataFrame,
    violations: dict[str, list[dict]],
    selected_label: str,
    k_min: float = 80.0,
    k_max: float = 120.0,
) -> go.Figure:
    """Smile chart for one expiry with arbitrage violation markers overlaid.

    Parameters
    ----------
    df             : full long-format DataFrame
    violations     : dict with keys 'calendar', 'butterfly', 'vertical'
                     each containing a list of violation dicts
    selected_label : the expiry_label to display
    k_min, k_max   : moneyness filter
    """
    s = (
        df[df["expiry_label"] == selected_label]
        .sort_values("moneyness_pct")
    )
    s = s[s["moneyness_pct"].between(k_min, k_max)]

    fig = go.Figure()

    # Base smile
    fig.add_trace(
        go.Scatter(
            x=s["moneyness_pct"],
            y=s["implied_vol"] * 100,
            mode="lines+markers",
            name="IV Smile",
            line=dict(color="royalblue", width=2),
            marker=dict(size=7),
        )
    )

    # Helper: look up IV for a moneyness value in this expiry slice
    iv_lookup = dict(zip(s["moneyness_pct"], s["implied_vol"] * 100))

    # Calendar violations (red ×)
    cal_v = [v for v in violations.get("calendar", []) if v.get("expiry_label") == selected_label]
    if cal_v:
        ks  = [v["moneyness_pct"] for v in cal_v]
        ivs = [iv_lookup.get(k, np.nan) for k in ks]
        fig.add_trace(
            go.Scatter(
                x=ks, y=ivs, mode="markers", name="Calendar Viol.",
                marker=dict(color="red", size=14, symbol="x", line=dict(width=2)),
                hovertemplate="Calendar violation<br>K: %{x:.1f}%<extra></extra>",
            )
        )

    # Butterfly violations (orange diamond)
    bfly_v = [v for v in violations.get("butterfly", []) if v.get("expiry_label") == selected_label]
    if bfly_v:
        ks  = [v["moneyness_pct"] for v in bfly_v]
        ivs = [iv_lookup.get(k, np.nan) for k in ks]
        fig.add_trace(
            go.Scatter(
                x=ks, y=ivs, mode="markers", name="Butterfly Viol.",
                marker=dict(color="darkorange", size=13, symbol="diamond",
                            line=dict(width=1, color="black")),
                hovertemplate="Butterfly violation<br>K: %{x:.1f}%<br>d²IV/dK²: %{customdata:.4f}<extra></extra>",
                customdata=[v["d2iv"] for v in bfly_v],
            )
        )

    # Vertical spread violations (red dashed segment between K1 and K2)
    vert_v = [v for v in violations.get("vertical", []) if v.get("expiry_label") == selected_label]
    for i, v in enumerate(vert_v):
        k1, k2 = v["k1"], v["k2"]
        y1 = iv_lookup.get(k1, np.nan)
        y2 = iv_lookup.get(k2, np.nan)
        fig.add_trace(
            go.Scatter(
                x=[k1, k2], y=[y1, y2],
                mode="lines+markers",
                name="Vertical Viol." if i == 0 else None,
                showlegend=(i == 0),
                line=dict(color="red", dash="dash", width=2.5),
                marker=dict(color="red", size=10, symbol="circle-open"),
                hovertemplate="Vertical violation<br>C(K₁)=%{customdata[0]:.4f}  C(K₂)=%{customdata[1]:.4f}<extra></extra>",
                customdata=[[v["c1"], v["c2"]]] * 2,
            )
        )

    fig.add_vline(x=100.0, line_dash="dash", line_color="grey",
                  annotation_text="ATM", annotation_position="top right")
    fig.update_layout(
        title=f"Smile with Arbitrage Flags — {selected_label}",
        xaxis_title="Moneyness (%)",
        yaxis_title="Implied Vol (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Local volatility charts
# ---------------------------------------------------------------------------

def plot_local_vol_3d(
    lv_grid: LocalVolGrid,
    title: str = "Dupire Local Volatility Surface",
) -> go.Figure:
    """3-D surface of Dupire local vol, styled to match the IV surface."""
    LV_pct = lv_grid.LV_grid * 100

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=lv_grid.K_grid,
            y=lv_grid.T_grid,
            z=LV_pct,
            colorscale="RdYlGn_r",
            colorbar=dict(title="LV (%)", thickness=14, len=0.7),
            connectgaps=True,
            hovertemplate=(
                "Moneyness: %{x:.1f}%<br>"
                "TTE: %{y:.3f} yr<br>"
                "Local Vol: %{z:.2f}%<extra></extra>"
            ),
            name="Local Vol",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(title="Moneyness (%)"),
            yaxis=dict(title="Time to Expiry (yr)"),
            zaxis=dict(title="Local Vol (%)"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.7)),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.8),
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_lv_vs_iv_slice(
    lv_grid: LocalVolGrid,
    expiry_idx: int,
    expiry_label: str = "",
) -> go.Figure:
    """Local vol vs implied vol for a single expiry slice.

    The Derman–Kani approximation for the local vol skew is roughly:
        β_loc ≈ 2 × β_IV   (local vol skew ≈ twice the implied vol skew)
    This plot illustrates that relationship directly.
    """
    m   = lv_grid.moneyness
    lv  = lv_grid.LV_grid[expiry_idx] * 100
    iv  = lv_grid.IV_grid[expiry_idx] * 100
    T   = lv_grid.expiries[expiry_idx]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=m, y=lv,
            mode="lines",
            name="Local Vol",
            line=dict(color="crimson", width=2.5),
            hovertemplate="Moneyness: %{x:.1f}%<br>LV: %{y:.2f}%<extra>Local Vol</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m, y=iv,
            mode="lines",
            name="Implied Vol",
            line=dict(color="royalblue", width=2, dash="dash"),
            hovertemplate="Moneyness: %{x:.1f}%<br>IV: %{y:.2f}%<extra>Implied Vol</extra>",
        )
    )

    # Derman–Kani approximate local vol: IV_atm + 2×(IV − IV_atm)
    atm_idx  = int(np.argmin(np.abs(m - 100.0)))
    iv_atm   = iv[atm_idx]
    dk_approx = iv_atm + 2.0 * (iv - iv_atm)
    fig.add_trace(
        go.Scatter(
            x=m, y=dk_approx,
            mode="lines",
            name="D–K approx  (IV_ATM + 2×skew)",
            line=dict(color="orange", width=1.5, dash="dot"),
            hovertemplate="Moneyness: %{x:.1f}%<br>D–K: %{y:.2f}%<extra>D–K approx</extra>",
        )
    )

    fig.add_vline(x=100.0, line_dash="dash", line_color="grey",
                  annotation_text="ATM", annotation_position="top right")
    fig.update_layout(
        title=f"Local Vol vs Implied Vol — {expiry_label}  (T = {T:.3f} yr)",
        xaxis_title="Moneyness (%)",
        yaxis_title="Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=500,
    )
    return fig


def plot_atm_term_structure(
    lv_grid: LocalVolGrid,
    df_raw: pd.DataFrame | None = None,
) -> go.Figure:
    """ATM local vol vs ATM implied vol across the term structure.

    Also shows the theoretical relationship:
        σ²_loc(ATM, T) = σ²_IV(ATM, T) + 2·T·σ_IV · ∂σ_IV/∂T
    which means local vol is above IV when the term structure is rising,
    and below when it is falling (inverted).
    """
    from local_vol import atm_comparison
    atm = atm_comparison(lv_grid)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=atm["time_to_expiry"],
            y=atm["lv_atm"] * 100,
            mode="lines+markers",
            name="ATM Local Vol",
            line=dict(color="crimson", width=2.5),
            marker=dict(size=6),
            hovertemplate="TTE: %{x:.3f} yr<br>ATM LV: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=atm["time_to_expiry"],
            y=atm["iv_atm"] * 100,
            mode="lines+markers",
            name="ATM Implied Vol",
            line=dict(color="royalblue", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="TTE: %{x:.3f} yr<br>ATM IV: %{y:.2f}%<extra></extra>",
        )
    )

    # Optionally overlay the raw ATM data points
    if df_raw is not None:
        raw_atm = (
            df_raw[df_raw["moneyness_pct"] == 100.0]
            .sort_values("time_to_expiry")
        )
        if not raw_atm.empty:
            fig.add_trace(
                go.Scatter(
                    x=raw_atm["time_to_expiry"],
                    y=raw_atm["implied_vol"] * 100,
                    mode="markers",
                    name="Bloomberg ATM quotes",
                    marker=dict(color="royalblue", size=8, symbol="circle-open",
                                line=dict(width=2)),
                    hovertemplate=(
                        "%{customdata}<br>ATM IV: %{y:.2f}%<extra>Bloomberg</extra>"
                    ),
                    customdata=raw_atm["expiry_label"],
                )
            )

    fig.update_layout(
        title="ATM Term Structure — Local Vol vs Implied Vol",
        xaxis_title="Time to Expiry (years)",
        yaxis_title="Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    return fig
