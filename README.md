# Local Volatility Surface Explorer & Monte Carlo Pricer

An interactive web application for exploring implied volatility surfaces,
computing Dupire local volatility, and pricing European options (vanilla,
digital, and barrier — calls and puts) via Monte Carlo simulation — all from
a Bloomberg OVME Excel snapshot.  Built with Python, Streamlit, and Plotly.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [Project Structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [Installation & Launch](#5-installation--launch)
6. [Input Data Format](#6-input-data-format)
7. [Application Walkthrough](#7-application-walkthrough)
   - 7.1 [Sidebar Controls](#71-sidebar-controls)
   - 7.2 [IV Surface Tab](#72-iv-surface-tab)
   - 7.3 [Vol Smiles Tab](#73-vol-smiles-tab)
   - 7.4 [Term Structure Tab](#74-term-structure-tab)
   - 7.5 [Heatmap Tab](#75-heatmap-tab)
   - 7.6 [Local Volatility Tab](#76-local-volatility-tab)
   - 7.7 [MC Pricer Tab](#77-mc-pricer-tab)
   - 7.8 [Arbitrage Checks Tab](#78-arbitrage-checks-tab)
8. [Arbitrage Checks — Mathematics](#8-arbitrage-checks--mathematics)
   - 8.1 [Calendar Spread Check](#81-calendar-spread-check)
   - 8.2 [Butterfly Spread Check](#82-butterfly-spread-check)
   - 8.3 [Vertical Spread Check](#83-vertical-spread-check)
9. [Surface Interpolation](#9-surface-interpolation)
10. [Local Volatility — Theory & Implementation](#10-local-volatility--theory--implementation)
    - 10.1 [Dupire Formula](#101-dupire-formula)
    - 10.2 [Why SVI?](#102-why-svi)
    - 10.3 [Numerical Pipeline](#103-numerical-pipeline)
11. [Module Reference](#11-module-reference)
12. [Test Suite](#12-test-suite)
13. [Known Limitations & Data Notes](#13-known-limitations--data-notes)
14. [Future Extensions](#14-future-extensions)

---

## 1. Overview

This tool has two equal primary functions:

1. **Implied Volatility Surface Explorer** — loads a Bloomberg OVME implied
   volatility matrix from an Excel workbook, builds a smooth bicubic-spline
   surface, and renders it through a zero-code Streamlit interface.  Three
   independent no-arbitrage checks run automatically and flag any violations
   with coloured badges and downloadable reports.

2. **Local Volatility Monte Carlo Pricer** — fits Gatheral SVI to each expiry
   slice, applies the full Dupire formula (total-variance form with PCHIP
   derivatives and a T=0 anchor) to extract a dense local vol surface, then
   prices European vanilla, digital, and barrier options (calls and puts)
   using a batched log-Euler Monte Carlo simulation with adaptive convergence.

The app supports a **daily file workflow**: place one Bloomberg Excel file per
day in the `data_vol_surface/` folder and switch between dates directly in the
sidebar without restarting the server.

The bundled dataset covers the **S&P 500 Index (SPX)** with:

| Property | Value |
|---|---|
| Available snapshots | 14 Apr 2026 |
| Near-term forward | ~6 785 |
| Expiry range | Next day → 17 Dec 2032 (33 slices) |
| Moneyness levels | 60 % · 80 % · 90 % · 95 % · 97.5 % · 100 % · 102.5 % · 105 % · 110 % · 120 % · 130 % · 140 % (12 levels) |
| IV range | ~10 % → ~70 % |

---

## 2. Features

| Goal | What the app does |
|---|---|
| **Daily file selector** | Reads all `.xlsx` files from `data_vol_surface/`, sorts by date embedded in filename (`dd_mm_yyyy`), and lets the user pick a snapshot from the sidebar |
| **Data ingestion** | Parses the Bloomberg wide-format Excel sheet; validates schema and raises descriptive errors on failure |
| **Surface building** | Interpolates the raw (N_T × 12) grid onto a dense 60 × 60 regular grid using bicubic spline in log(T) space |
| **3-D IV visualisation** | Interactive rotating surface with optional raw Bloomberg quote overlay |
| **Vol smile slices** | Per-expiry IV vs moneyness curves with quick-select tenor buckets and optional Gatheral SVI overlay |
| **SVI fitting** | Fits `w(x) = a + b·(ρ·(x−m) + √((x−m)²+σ²))` per expiry slice; displays 5 parameters and residual in a table |
| **Term structure** | ATM implied vol vs time-to-expiry on a log(T) x-axis |
| **Heatmap** | Colour-coded IV matrix with annotated values |
| **Dupire local vol** | Full Gatheral-formulation local vol surface via SVI analytical derivatives and PCHIP dw/dT; three sub-views (3-D surface, LV vs IV slice, ATM term structure) |
| **Arbitrage checks** | Calendar spread, butterfly spread, and vertical spread checks with severity scores |
| **Violation overlay** | Select any expiry to see its smile with violations marked directly on the chart |
| **Export** | Download violation reports as a multi-sheet Excel file |

---

## 3. Project Structure

```
vol_surface_excel/
├── data_vol_surface/                  # One Bloomberg Excel file per snapshot date
│   └── vol_surface_14_04_2026.xlsx
├── pyproject.toml                     # uv / PEP 517 project config
├── README.md
├── src/
│   ├── data_loader.py        # Excel ingestion, validation, long-format output
│   ├── iv_surface_builder.py    # Grid interpolation (bicubic spline) + SVI fitting
│   ├── montecarlo.py         # Monte Carlo pricer (vanilla / digital / barrier, call or put) via Dupire local vol
│   ├── local_vol.py          # Dupire local vol via SVI + PCHIP
│   ├── plots.py              # All Plotly chart functions
│   ├── arbitrage_checks.py   # Calendar / butterfly / vertical checks
│   └── streamlit_app.py      # UI entry-point
└── tests/
    ├── test_data_loader.py
    ├── test_surface_builder.py
    └── test_arbitrage_checks.py
```

**Data flow:**

```
data_vol_surface/*.xlsx   (or uploaded file)
          │
          ▼
   data_loader.py  ──────────────────────────────────┐
          │                                           │
          ▼                                           ▼
 iv_surface_builder.py                            local_vol.py
          │                                           │    \
          └──────────────────┬────────────────────────┘     ▼
                             ▼                         montecarlo.py
                         plots.py  ◄──  arbitrage_checks.py
                             │
                             ▼
                      streamlit_app.py
```

---

## 4. Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | Required for `str \| Path` and `list[str]` syntax |
| pandas | ≥ 2.2 | Data wrangling |
| numpy | ≥ 1.26 | Numerical arrays |
| openpyxl | ≥ 3.1 | Excel I/O |
| scipy | ≥ 1.13 | Spline interpolation, PCHIP, Black-76, SVI optimisation |
| plotly | ≥ 5.22 | Interactive charts |
| streamlit | ≥ 1.35 | Web UI |

All versions are pinned in `pyproject.toml`.  Dev extras (`pytest`, `ruff`,
`mypy`) are under `[project.optional-dependencies] dev`.

---

## 5. Installation & Launch

[uv](https://github.com/astral-sh/uv) is the package manager for this project.
It handles virtual environments automatically — you never need to activate one
manually.

**Install uv** (once, system-wide):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

**Clone / open the project folder**, then:

```bash
# Create the virtual environment and install all dependencies
uv sync

# Launch the Streamlit app
uv run streamlit run src/streamlit_app.py
```

Open your browser at **http://localhost:8501**.

Press `Ctrl + C` in the terminal to stop the app.

**Run the test suite:**

```bash
uv run pytest tests/ -v
```

**Install dev dependencies** (linting, type-checking, testing):

```bash
uv sync --extra dev
```

> `uv sync` reads `pyproject.toml` and installs exact versions into a local
> `.venv` folder.  `uv run` invokes commands inside that environment
> transparently.

---

## 6. Input Data Format

### File naming convention

Place Bloomberg Excel files in the `data_vol_surface/` folder.  Files **must**
follow the naming pattern:

```
vol_surface_DD_MM_YYYY.xlsx
```

Examples: `vol_surface_08_04_2026.xlsx`, `vol_surface_09_04_2026.xlsx`.

The app parses the date from the filename to sort files chronologically (most
recent first) in the sidebar selector.  Files that do not match this pattern
are still listed but sorted to the bottom.

You can also bypass the folder entirely by uploading a file directly via the
**Upload a workbook** button in the sidebar — this takes priority over the
folder selector.

### Sheet structure

The app expects a single sheet named **`VolSurface`** in Bloomberg OVME wide
format:

| Row | Content |
|---|---|
| 1 | Headers: `Exp Date` · `ImpFwd` · `60.0%` · `80.0%` · … · `140.0%` |
| 2 | Absolute strike prices per moneyness level (reference only — skipped by the parser) |
| 3 + | Data: expiry date string · forward price · implied vols in % |

### Column definitions

| Column | Example | Description |
|---|---|---|
| `Exp Date` | `16 Jun 2026` | Option expiry date (parsed with `dayfirst=True`) |
| `ImpFwd` | `6831.64` | Forward price F for that expiry slice |
| `60.0%` … `140.0%` | `31.64` | Implied vol in **percentage points** at moneyness K/F = 60 % … 140 % |

### Moneyness convention

All strike levels are expressed as **(K / F) × 100**, where F is the forward
price in the `Imp` column of the same row.  100 % = at-the-money forward
(ATMF).

### Vol convention

Values in the Excel file are in percentage points.  The parser divides by 100
before storing them internally (so `22.03` → `σ = 0.2203`).

---

## 7. Application Walkthrough

### 7.1 Sidebar Controls

| Control | Default | Description |
|---|---|---|
| **Upload a workbook** | — | Upload any Bloomberg OVME `.xlsx` file; overrides the folder selector |
| **Select daily file** | Most recent | Dropdown of all files in `data_vol_surface/`, sorted most-recent-first; disabled while a file is uploaded |
| **Snapshot date** | Today | Date used to compute time-to-expiry: `T = (expiry_date − snap_date).days / 365.25` |
| **Risk-free rate r (%)** | 4.5 % | Discount rate used in Black-76 call pricing for the vertical spread check and MC pricer |
| **Moneyness range (%)** | 80 % – 120 % | Filter applied to the smile slices and arbitrage flag charts |

After loading, five metric cards at the top of the page show: **Ticker**,
**Spot / Fwd**, **Expiries** count, **Strikes** count, and **Interpolation**
(always "Bicubic Spline").

---

### 7.2 IV Surface Tab

An interactive rotating 3-D surface showing implied vol (%) as a function of
moneyness (%) and time-to-expiry (years).

- **Rotate:** click and drag
- **Zoom:** scroll wheel
- **Hover:** displays exact moneyness, TTE, and IV at any grid point
- **Show raw data points toggle:** overlays the original Bloomberg quotes
  as coloured scatter markers.  Marker colour uses the same `RdYlGn_r` scale
  as the surface (red = high IV, green = low IV) so you can judge immediately
  how well the interpolation tracks the actual quotes.

The time axis labels are sampled at positions **evenly spaced in log(T)**,
giving readable ticks across both the front end (days to weeks) and the long
end (years).

---

### 7.3 Vol Smiles Tab

Per-expiry implied vol vs moneyness curves.  All selected slices are drawn on
one chart so you can compare smile shape and skew across maturities.

**Expiry selector:**

| Button | Selects |
|---|---|
| **All** | All 33 expiries |
| **None** | Clears the selection |
| **≤1M (N)** | Expiries with TTE ≤ 1 month |
| **1M–1Y (N)** | Expiries between 1 month and 1 year |
| **>1Y (N)** | Long-dated expiries (> 1 year) |

After pressing a quick-select button you can still fine-tune the selection
manually using the multi-select dropdown.

The ATM line (moneyness = 100 %) is shown as a grey dashed reference.

**SVI overlay toggle:**

When enabled, a Gatheral SVI smile `w(x) = a + b·(ρ·(x−m) + √((x−m)²+σ²))`
is fitted per expiry using L-BFGS-B optimisation and overlaid as a dashed
curve.  Below the chart an expandable table shows the 5 SVI parameters for
each expiry:

| Parameter | Interpretation |
|---|---|
| **a** | Overall level of total variance (ATM) |
| **b** | Slope of the wings (vol-of-vol proxy) |
| **ρ** | Skewness — negative for equity markets (put skew) |
| **m** | Horizontal shift of the minimum of the smile |
| **σ** | Smile curvature / width at the minimum |
| **Residual** | Sum of squared errors between SVI and raw quotes |

A **Raw data expander** shows the underlying Bloomberg quote table for
the selected expiries.

---

### 7.4 Term Structure Tab

ATM implied vol (moneyness = 100 %) plotted against time-to-expiry on a
**log(T) x-axis**.  Using a log scale is essential because the expiry axis
spans from 1 day to 6+ years — a linear axis would compress all the
short-dated data into a narrow band.

A metric below the chart shows the ATM vol change from expiry 1 to expiry 2
(i.e. the short-end slope of the term structure).

---

### 7.5 Heatmap Tab

The full IV matrix rendered as a colour-coded grid.  Each cell shows its
value in percentage points.  Rows are sorted by ascending time-to-expiry
(nearest expiry at the top).

Useful for spotting outliers or suspicious data at a glance before running the
formal arbitrage checks.

---

### 7.6 Local Volatility Tab

Displays the **Dupire local volatility surface** derived from the implied vol
data.  See [Section 10](#10-local-volatility--theory--implementation) for the
mathematical details.

Four coverage metrics are shown at the top:

| Metric | Meaning |
|---|---|
| **Min LV** | Minimum local vol across the valid surface |
| **Max LV** | Maximum local vol across the valid surface |
| **Arbitrage holes** | Number of grid points where `g ≤ _G_FLOOR` (denominator guard) — the Dupire formula is ill-defined there |
| **Valid coverage** | Percentage of the dense grid that has a finite local vol value |

Three sub-tabs are available:

#### 3-D Local Vol

Interactive rotating surface of local vol (%) vs moneyness × TTE.  The colour
scale is `RdYlGn_r` (same as the IV surface).  White gaps appear where the
Dupire formula is ill-conditioned (arbitrage holes).

#### LV vs IV Slice

For a selected expiry: overlays local vol, implied vol, and the Derman–Kani
linear approximation on a single moneyness chart.

- **Red solid** = Dupire local vol
- **Blue dashed** = Implied vol (from the interpolated surface)
- **Orange dotted** = D–K approximation: `LV_ATM + 2×(IV − IV_ATM)`

The approximation shows that the local vol skew is approximately **twice** the
implied vol skew (valid for nearly-normal smiles).

#### ATM Term Structure

ATM local vol and ATM implied vol plotted together against time-to-expiry:

- **Crimson solid** = ATM local vol `σ_loc(100%, T)`
- **Blue dashed** = ATM implied vol (interpolated surface)
- **Open circles** = raw Bloomberg ATM quotes

Where the IV term structure is **upward-sloping**, local vol lies above implied
vol.  Where it is **flat or inverted**, local vol ≈ implied vol or falls below
it.  This is the classic relationship between the two surfaces.

---

### 7.7 MC Pricer Tab

Prices European options on the Dupire local vol surface via Monte Carlo
simulation.  The local vol grid (resolution 100) is shared with the Local
Volatility tab — no extra compute if already cached.

**Supported payoffs:**

| Option type | Style | Combinable with |
|---|---|---|
| Call / Put | Vanilla | Barrier (up/down × in/out, American or European monitoring) |
| Call / Put | Digital (cash-or-nothing) | — (digitals pay only on terminal ITM; barrier controls are hidden in the UI) |

**Inputs:**

| Input | Default | Description |
|---|---|---|
| **Option type** | Call | Segmented control: `Call` or `Put` |
| **Digital** | Off | Toggle — when on, payoff becomes cash-or-nothing (fixed amount if ITM at expiry, else 0).  Mutually exclusive with barriers: the Barrier controls are hidden while Digital is on. |
| **Cash payout** (digital only) | 1.0 | Amount paid when `S_T` is ITM at expiry |
| **Barrier** | None | Segmented control: `None`, `Up-and-Out`, `Up-and-In`, `Down-and-Out`, `Down-and-In` |
| **Barrier level B** (barrier only) | S₀ × 1.10 (up) or 0.90 (down) | Absolute barrier level |
| **Barrier monitoring** (barrier only) | American | `American` (every Euler step) or `European` (terminal spot only) |
| **Spot S₀** | From dataset | Pre-filled from the loaded Bloomberg snapshot |
| **Strike K** | ATM (= S₀) | Absolute strike level |
| **Maturity T (years)** | 1.0000 | Option maturity; step 0.0025 yr ≈ 1 day; surface min ~0.04 yr |
| **Dividend yield q (%)** | 0.000 | Continuous dividend yield; 3 decimal places |
| **Steps per year** | 252 (daily) | Euler discretisation steps; alternative: 52 (weekly) |
| **Convergence ε** | 0.001 | Relative stopping threshold |

The risk-free rate is shared with the sidebar (same value used for the
arbitrage checks).

**Barrier monitoring styles:**

- **American** — the barrier is checked at every Euler step.  Discrete daily
  sampling under-estimates the hit probability vs continuous monitoring, but
  reproduces the behaviour of most traded barrier contracts.
- **European** — the barrier is checked only at expiry (`S_T` alone decides
  hit / no-hit).  A knock-out option then behaves like a vanilla one with a
  payoff extinguished only if the terminal spot breaches the barrier.

**Convergence rule:**

Paths run in batches of 500.  After each batch, stop when:

```
|P(n) − P(n−500)| / max(P(n), 1e-8) < ε   and   n ≥ 5 000
```

Hard cap: 50 000 paths.

**Surface coverage warnings** are shown before running if the strike moneyness
or maturity falls outside the local vol surface domain.  A barrier sanity
check warns if the barrier is on the wrong side of spot at inception.

**Outputs:** Price (label dynamically reflects flavour and style, e.g.
"(American) Up-and-Out Digital Call Price") · Standard error · 95%
confidence interval · Paths simulated · Convergence badge (green = converged,
amber = cap reached) · % of steps clamped to the surface boundary.

---

### 7.8 Arbitrage Checks Tab

Three independent no-arbitrage conditions are tested automatically each time
the risk-free rate slider is moved.  Results are displayed as:

- **Summary row** — green PASS / red FAIL badge and a one-line summary for
  each check
- **Detail expander** — full table of violating (K, T) pairs with severity
  scores (cells highlighted in red)
- **Violation overlay chart** — select any expiry to see its smile with
  violations marked directly:
  - **×** = calendar spread violation
  - **◆** = butterfly spread violation
  - **dashed line** = vertical spread violation (call prices inverted)
- **Download button** — exports all violations to a multi-sheet Excel report
  (`arbitrage_violations.xlsx`), one sheet per check type

An expandable info box explains why Bloomberg raw data can show violations
even though the market is liquid (event premiums, bid-ask noise, extraction
artefacts).

---

## 8. Arbitrage Checks — Mathematics

A volatility surface is arbitrage-free if and only if all three conditions
below hold simultaneously.  Each check is independent.

### 8.1 Calendar Spread Check

**Economic intuition:** a calendar spread buys a long-dated option and sells a
short-dated one at the same strike.  The long-dated option has strictly more
optionality, so it must always be worth at least as much.

**Condition:** the *total variance* `w(K, T) = σ²(K, T) · T` must be
non-decreasing in T for every fixed moneyness K:

```
For all K and T₁ < T₂:   w(K, T₂) ≥ w(K, T₁)
```

**Violation:** `w(K, T₂) − w(K, T₁) < 0` for any consecutive expiry pair
`(T₁, T₂)` at the same moneyness K.

**Severity score:** `|Δw| / w(K, T₁)` — relative breach normalised by the
shorter-dated total variance.

---

### 8.2 Butterfly Spread Check

**Economic intuition:** a butterfly spread (long ITM + long OTM − 2 × ATM)
has value equal to the second derivative of the call price with respect to
strike scaled by K².  A negative value implies a negative risk-neutral
probability density, which is impossible for any valid probability measure.

**Condition:** implied vol must be convex in moneyness at every interior
strike:

```
d²IV / dK² ≥ 0   for all interior K at each expiry T
```

**Implementation:** non-uniform centred finite differences are used because
the moneyness grid is non-uniform (60, 80, 90, 95, 97.5, 100, 102.5, 105, 110, 120, 130, 140):

```
d²IV/dK²  ≈  2 / (h₁ + h₂)  ×  [IV(K+h₂)/h₂  −  IV(K)·(1/h₁ + 1/h₂)  +  IV(K−h₁)/h₁]
```

where `h₁ = K − K_prev` and `h₂ = K_next − K`.

**Tolerance:** violations are reported only when `d²IV/dK² < −1×10⁻⁴` to
account for bid-ask spread noise.

---

### 8.3 Vertical Spread Check

**Economic intuition:** a bull call spread (long K₁, short K₂ > K₁) has a
maximum payoff of K₂ − K₁ and a minimum payoff of zero, so it must always
have a strictly positive cost.  Therefore call prices must be non-increasing
in strike.

**Condition:**

```
C(K₁, T) ≥ C(K₂, T)   for all K₁ < K₂ at each expiry T
```

**Implementation:** call prices are computed from implied vols using the
**Black-76 formula** (appropriate for forward-based moneyness):

```
C = e^(−rT) · [F · N(d₁) − K · N(d₂)]

d₁ = [ln(F/K) + ½σ²T] / (σ√T)
d₂ = d₁ − σ√T
```

where `K = (moneyness_pct / 100) × F` and `r` is the risk-free rate set in
the sidebar.

**Severity score:** breach magnitude / ATM call price — expressed as a
fraction of the at-the-money premium.

---

## 9. Surface Interpolation

The raw data is a complete (N_T × 12) grid interpolated onto a dense 60 × 60
display grid using `scipy.interpolate.RectBivariateSpline` (bicubic,
`kx = ky = 3`, `s = 0` for exact fit through every data point).

**Key design choice — log(T) axis:** the expiry axis spans from 1 day to 6.7
years.  Interpolating in linear T would compress all the short-dated data
(where the skew spike lives) into a tiny region, causing the bicubic spline to
under-represent the front-end.  Working in `log(T)` gives each decade of
maturity equal weight and accurately reproduces the steep vol term structure
at the short end.

If `RectBivariateSpline` raises an exception (e.g. due to near-duplicate
grid points), the code falls back to a linear spline (`kx = ky = 1`) — less
smooth but always numerically stable.

---

## 10. Local Volatility — Theory & Implementation

### 10.1 Dupire Formula

Dupire (1994) showed that, given the full surface of European call prices
`C(K, T)`, there exists a unique diffusion

```
dS_t / S_t = (r − q) dt + σ_loc(S_t, t) dW^Q
```

that prices every European option correctly.  The local volatility function is
given by:

```
σ²_loc(K, T) = (∂C/∂T) / (½ K² · ∂²C/∂K²)
```

In Gatheral (2006) total-variance form, with `x = ln(K/F)` and
`w(x, T) = σ²_IV(x, T) · T`:

```
σ²_loc(x, T) = (∂w/∂T) / g(x, T)
```

where the denominator `g` is the full expression:

```
g(x, T) = 1 − (x/w)·(∂w/∂x)
             + ¼·(∂w/∂x)²·(−¼ − 1/w + x²/w²)
             + ½·(∂²w/∂x²)
```

Note: the common approximation `g ≈ 1` is only valid when the smile is
symmetric around the forward.  This implementation uses the exact `g`
expression at every grid point.

**ATM case (x = 0):**

```
σ²_loc(0, T) = (∂w/∂T)|_{x=0}  /  g(0, T)

g(0, T) = 1 + ¼·(∂w/∂x)²·(−¼ − 1/w)  +  ½·(∂²w/∂x²)
```

Both the skew term `(∂w/∂x)²` and curvature term `(∂²w/∂x²)` contribute.

### 10.2 Why SVI?

With only ~12 moneyness quotes per expiry (60 %–140 % in the Bloomberg sheet),
numerical second derivatives of the implied vol smile are highly sensitive to
the non-uniform strike spacing.  The cluster around 95–100 % causes
`CubicSpline` to produce wild `d²w/dx²` values, making the denominator `g`
ill-conditioned.

Instead, we fit a **Gatheral SVI** parametric smile per expiry slice:

```
w(x) = a + b · (ρ·z + √(z² + σ²))    where z = x − m
```

The analytical derivatives are:

```
∂w/∂x   = b · (ρ + z/d)              where d = √(z² + σ²)
∂²w/∂x² = b · σ² / d³                always ≥ 0 when b, σ > 0
```

Because `∂²w/∂x² ≥ 0` by construction (for a well-fitted SVI), the
denominator `g` is much better-conditioned than with numerical finite
differences.

### 10.3 Numerical Pipeline

The local vol computation proceeds in four steps:

**Step 1 — SVI fit per expiry (x-derivatives)**

For each of the `N_T` raw expiry slices, fit SVI parameters `{a, b, ρ, m, σ}`
by minimising the least-squares residual.  Evaluate `w`, `∂w/∂x`, `∂²w/∂x²`
analytically on a fine grid of 200 log-moneyness points `x_fine`.

If the SVI fit fails (residual too large or `w ≤ 0`), fall back to
linear interpolation of raw `w` values plus numerical `np.gradient` for
the derivatives.

**Step 2 — PCHIP for dw/dT (T-derivatives)**

For each column `j` of `x_fine`, fit a PCHIP (Piecewise Cubic Hermite
Interpolating Polynomial) spline and evaluate its first derivative `∂w/∂T`.
PCHIP is shape-preserving and monotone where the data is monotone, so it
avoids spurious oscillations that would cause `∂w/∂T < 0`.

A synthetic anchor `(T=0, w=0)` is prepended before fitting.  This is exact
— total variance vanishes at zero expiry by definition — and it gives PCHIP a
well-conditioned left boundary, producing stable `∂w/∂T` estimates at the
first real expiry where the surface changes most rapidly.

Any residual negative values of `∂w/∂T` (due to numerical noise) are floored
to `_DWT_FLOOR = 1e-6` rather than masked as arbitrage violations, which
prevents unnecessary holes in the surface.

**Step 3 — Dupire formula on fine (T, x) grid**

Apply the full Gatheral `g` formula and compute:

```
σ²_loc = dw_dT_safe / g     where g > _G_FLOOR = 0.02
```

Points where `g ≤ _G_FLOOR` are marked as arbitrage violations and set to
`NaN`.  Points above the hard ceiling `_LV_CAP = 150 %` are also clipped to
`NaN` (numerical artefacts).

**Step 4 — Interpolate onto dense display grid**

`NaN` values are filled column-by-column by linear interpolation over valid
T-neighbours before the final `RegularGridInterpolator` maps the result onto
the dense `(n_k × n_t)` display grid.  This ensures a smooth, gap-free
surface for rendering, while the arbitrage hole count metric still reflects
how many fine-grid points were ill-conditioned.

The dense expiry axis `t_dense` is **log-spaced** (`np.exp(np.linspace(log(T_min), log(T_max), n_t))`).
This concentrates grid points at short maturities where the local vol surface
changes fastest, giving much finer resolution in the front-end region that
matters most for short-dated option pricing.

---

## 11. Module Reference

### `data_loader.py`

| Function / Class | Description |
|---|---|
| `load_workbook(path, snap_date)` | Main entry point.  Returns `VolDataset(vol_df, metadata)` |
| `_parse_vol_surface(raw, snap)` | Melts the wide-format sheet into a tidy long DataFrame |
| `_extract_metadata(raw, vol_df, snap)` | Builds the metadata dict (title, ticker, spot, n_expiries, …) |
| `validate(df)` | Returns a list of error strings; empty list = valid |
| `VolDataset` | `namedtuple('VolDataset', ['vol_df', 'metadata'])` |

**Output DataFrame columns:**

| Column | Type | Description |
|---|---|---|
| `expiry_date` | date | Calendar expiry date |
| `expiry_label` | str | Human-readable label, e.g. `"16 Jun 2026"` |
| `time_to_expiry` | float | T in years from the snapshot date |
| `moneyness_pct` | float | K/F × 100 |
| `forward_price` | float | Forward price F for this expiry slice |
| `implied_vol` | float | Implied vol in decimal (e.g. `0.2203`) |

---

### `iv_surface_builder.py`

| Function / Class | Description |
|---|---|
| `build_surface(df, n_k, n_t)` | Interpolates raw grid onto dense `(n_k × n_t)` grid using bicubic spline.  Returns `SurfaceGrid` |
| `interpolate_slice(strikes, vols, k_new, method)` | 1-D interpolation for a single expiry smile |
| `svi_fit(strikes, vols, F, T)` | Fits Gatheral SVI `{a, b, ρ, m, σ}` by L-BFGS-B least squares |
| `SurfaceGrid` | `namedtuple('SurfaceGrid', ['K_grid', 'T_grid', 'IV_grid', 'moneyness', 'expiries', 'expiry_labels'])` |

---

### `local_vol.py`

| Function / Class | Description |
|---|---|
| `build_local_vol(df, n_k, n_t)` | Computes the full Dupire local vol surface.  Returns `LocalVolGrid` |
| `atm_comparison(lv_grid)` | Returns a DataFrame of ATM IV, ATM LV, and `lv/iv` ratio per expiry |
| `_svi_fit_slice(x, w_obs)` | Internal: fits SVI to `(x, w)` pairs; returns param dict or `None` |
| `_svi_eval(p, x)` | Internal: evaluates `(w, dw/dx, d²w/dx²)` analytically from SVI params |
| `LocalVolGrid` | `namedtuple('LocalVolGrid', ['K_grid', 'T_grid', 'LV_grid', 'IV_grid', 'moneyness', 'expiries', 'arb_mask'])` |

**Key constants:**

| Constant | Value | Role |
|---|---|---|
| `_LV_CAP` | 1.50 | Hard ceiling (150 %) — values above are numerical artefacts |
| `_G_FLOOR` | 0.02 | Denominator guard — points with `g ≤ 0.02` are masked |
| `_W_FLOOR` | 1e-8 | Floor on total variance `w` before computing `g` |
| `_DWT_FLOOR` | 1e-6 | Floor on `∂w/∂T` to prevent holes from numerical noise |
| `_T_MIN_CUTOFF` | 0.04 | ~2 weeks — very short expiries excluded (unstable derivatives) |
| `_N_X_FINE` | 200 | Number of log-moneyness points on the fine SVI evaluation grid |

---

### `plots.py`

| Function | Returns |
|---|---|
| `plot_surface_3d(grid, df, title)` | 3-D IV surface with optional raw data overlay |
| `plot_smile_slices(df, selected_labels, k_min, k_max, svi_fits)` | Multi-expiry smile chart with optional SVI overlay |
| `plot_term_structure(df)` | ATM IV vs TTE line chart (log x-axis) |
| `plot_heatmap(df)` | Annotated IV heatmap |
| `plot_arbitrage_flags(df, violations, selected_label, k_min, k_max)` | Smile with violation markers |
| `plot_local_vol_3d(lv_grid, title)` | 3-D Dupire local vol surface |
| `plot_lv_vs_iv_slice(lv_grid, expiry_idx, expiry_label)` | LV vs IV for one expiry + D–K approximation |
| `plot_atm_term_structure(lv_grid, df_raw)` | ATM LV and IV vs TTE with raw Bloomberg overlay |

---

### `montecarlo.py`

| Function / Class | Description |
|---|---|
| `price_european_option(lv_grid, S0, K, T, r, q, option_type, is_digital, digital_payout, barrier_type, barrier_level, barrier_style, steps_per_year, eps, seed)` | Runs the MC simulation for vanilla / digital / barrier options (call or put) and returns `MCResult` |
| `_build_lv_interpolator(lv_grid)` | Internal: builds a `RegularGridInterpolator` from `LocalVolGrid`, filling NaN holes |
| `MCResult` | `namedtuple('MCResult', ['price', 'std_error', 'n_paths', 'converged', 'clamp_pct'])` |

**Option flavours supported:**

| Flavour | Triggered by | Payoff |
|---|---|---|
| Vanilla call/put | `is_digital=False`, `barrier_type=None` | `max(S_T − K, 0)` / `max(K − S_T, 0)` |
| Digital call/put | `is_digital=True` | `digital_payout · 1_{S_T > K}` / `digital_payout · 1_{S_T < K}` |
| Barrier (any of the above) | `barrier_type` ∈ `up_out`, `up_in`, `down_out`, `down_in` + `barrier_level` | Knock-in / knock-out applied to the underlying vanilla/digital payoff |
| Barrier monitoring | `barrier_style` = `"american"` (default, every Euler step) or `"european"` (terminal spot only) | — |

**Key constants:**

| Constant | Value | Role |
|---|---|---|
| `_MIN_PATHS` | 5 000 | Minimum paths before convergence can trigger |
| `_MAX_PATHS` | 50 000 | Hard cap — `converged=False` if reached |
| `_BATCH_SIZE` | 500 | Paths per batch; also the convergence window |
| `_EPS_FLOOR` | 1e-8 | Prevents division by zero when price ≈ 0 |

---

### `arbitrage_checks.py`

| Function | Returns |
|---|---|
| `check_calendar_spread(df)` | `CheckResult` |
| `check_butterfly_spread(df, eps)` | `CheckResult` |
| `check_vertical_spread(df, r)` | `CheckResult` |
| `run_all_checks(df, r)` | `list[CheckResult]` — always `[calendar, butterfly, vertical]` |

```python
CheckResult = namedtuple('CheckResult', ['name', 'passed', 'violations', 'details'])
```

Each violation dict contains at minimum: `expiry_label`, `moneyness_pct`,
`time_to_expiry`, `severity`, plus check-specific fields.

---

## 12. Test Suite

Tests live in `tests/` and are run with `uv run pytest tests/ -v`.

| File | Tests |
|---|---|
| `test_data_loader.py` | Schema validation, long-format output columns, metadata extraction, time-to-expiry computation |
| `test_surface_builder.py` | Surface grid shape, no negative vols, front-end spike preservation, 1-D interpolation |
| `test_arbitrage_checks.py` | Calendar / butterfly / vertical check logic on synthetic data |

All tests load data from `data_vol_surface/vol_surface.xlsx` (relative to the
project root) rather than a file at the root level.

---

## 13. Known Limitations & Data Notes

**Front-end anomalies (very short expiries)**

The 1-day expiry can show `IV(90 %) > IV(80 %)`, i.e. the 90 % strike is
more expensive than the 80 % strike.  This reflects a binary-event premium
around specific strikes for an option expiring the next day (e.g. an FOMC or
CPI date).  This contributes to butterfly violations for that expiry.

**Duplicate vol rows**

Some Bloomberg extracts contain pairs of expiries with identical vol data but
different calendar dates (e.g. 16 Jun 2026 and 18 Jun 2026).  The loader
takes the mean (leaving values unchanged) and retains both as separate slices
since their time-to-expiry values differ.

**Single-sheet workbook**

The app expects one sheet named `VolSurface`.  Multi-sheet workbooks or
differently-named sheets will cause a load error.

**No bid/ask spread**

Bloomberg OVME exports mid implied vols only.  The butterfly check tolerance
(`ε = 1×10⁻⁴`) accounts for typical bid-ask noise but cannot be calibrated
to the actual spread.

**Black-76 pricing for vertical check**

Call prices are computed from implied vols using Black-76 with the user-
supplied risk-free rate.  Results are sensitive to the rate for deep ITM/OTM
options or very long maturities.  The default of 4.5 % is appropriate for
USD rates as of April 2026.

**Local vol at very short expiries**

The `_T_MIN_CUTOFF = 0.04` (≈ 2 weeks) filter removes the very shortest
expiries from the local vol computation.  These slices have too few T-
neighbours for PCHIP to estimate `∂w/∂T` reliably and are excluded to keep
the surface stable.

**SVI no-arbitrage**

The SVI fit in the Vol Smiles tab uses simple parameter bounds and does **not**
enforce the full Gatheral no-static-arbitrage conditions (Lee moment formula,
Durrleman condition).  For a fully arbitrage-free parametric fit, consider
SSVI or a constrained SVI calibration.

---

## 14. Future Extensions

| ID | Description |
|---|---|
| F-01 | **Delta-space display** — re-express the moneyness axis in Black-Scholes delta (Δ) for comparison with broker conventions |
| F-02 | **Multi-asset support** — detect the ticker from the title row and allow uploading multiple workbooks simultaneously |
| F-03 | **Historical comparison** — upload two snapshots and display the surface difference (vol change Δσ per node) |
| F-04 | **Greeks surface** — derive and display vega, vanna, and volga surfaces from the fitted IV grid |
| F-05 | **Gatheral density** — compute and plot the risk-neutral density implied by the butterfly condition |
| F-06 | **PNG / SVG export** — add a download button for each chart |
| F-07 | **CI pipeline** — add GitHub Actions running `uv run pytest tests/` and `ruff check src/` on every push |
| F-08 | **SSVI / constrained SVI** — replace the unconstrained SVI fit with a fully arbitrage-free parametric form |
