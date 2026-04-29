# Local Volatility & Heston Stochastic Volatility Pricer

An interactive web application for exploring implied volatility surfaces,
computing Dupire local volatility, calibrating the Heston stochastic
volatility model, and pricing European options (vanilla, digital, and
barrier — calls and puts) via Monte Carlo simulation — all from a Bloomberg
OVME Excel snapshot.  Built with Python, Streamlit, and Plotly.

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
   - 7.6 [LV Surface Tab](#76-lv-surface-tab)
   - 7.7 [LV MC Pricer Tab](#77-lv-mc-pricer-tab)
   - 7.8 [Heston Pricer Tab](#78-heston-pricer-tab)
   - 7.9 [Arbitrage Checks Tab](#79-arbitrage-checks-tab)
8. [Arbitrage Checks — Mathematics](#8-arbitrage-checks--mathematics)
   - 8.1 [Calendar Spread Check](#81-calendar-spread-check)
   - 8.2 [Butterfly Spread Check](#82-butterfly-spread-check)
   - 8.3 [Vertical Spread Check](#83-vertical-spread-check)
9. [Surface Interpolation](#9-surface-interpolation)
10. [Local Volatility — Theory & Implementation](#10-local-volatility--theory--implementation)
    - 10.1 [Dupire Formula](#101-dupire-formula)
    - 10.2 [Why per-slice SVI?](#102-why-per-slice-svi)
    - 10.3 [Numerical Pipeline](#103-numerical-pipeline)
11. [Heston Stochastic Volatility — Theory & Implementation](#11-heston-stochastic-volatility--theory--implementation)
    - 11.1 [Model Dynamics](#111-model-dynamics)
    - 11.2 [Semi-Analytic Pricing](#112-semi-analytic-pricing)
    - 11.3 [Calibration Strategy](#113-calibration-strategy)
    - 11.4 [Monte Carlo Scheme](#114-monte-carlo-scheme)
12. [Yield Curve — Theory & Implementation](#12-yield-curve--theory--implementation)
    - 12.1 [Data Source](#121-data-source)
    - 12.2 [Bootstrap](#122-bootstrap)
    - 12.3 [Interpolation](#123-interpolation)
    - 12.4 [Use in Pricing](#124-use-in-pricing)
13. [Module Reference](#13-module-reference)
14. [Test Suite](#14-test-suite)
15. [Known Limitations & Data Notes](#15-known-limitations--data-notes)

---

## 1. Overview

This tool has three primary functions:

1. **Implied Volatility Surface Explorer** — loads a Bloomberg OVME implied
   volatility matrix from an Excel workbook, builds a smooth bicubic-spline
   surface, and renders it through a zero-code Streamlit interface.  Three
   independent no-arbitrage checks run automatically and flag any violations
   with coloured badges and downloadable reports.

2. **Local Volatility Monte Carlo Pricer** — fits Gatheral SVI per expiry
   slice (the per-slice fit family Bloomberg OVME's LV surface uses, vs the
   global SSVI alternative), applies the full Dupire formula (total-variance
   form with PCHIP derivatives and a T=0 anchor) to extract a dense local
   vol surface, then prices European vanilla, digital, barrier, and
   one-touch options (calls and puts) using a batched log-Euler Monte Carlo
   simulation with adaptive convergence.  The MC layer uses discrete
   monitoring (one check per Euler step, no Brownian-bridge correction), so
   short-dated barrier and one-touch prices carry a known under-detection
   bias against Bloomberg's PDE-based OVME.

3. **Heston Stochastic Volatility Pricer** — calibrates the five Heston
   parameters (v₀, κ, θ, ξ, ρ) to the market IV surface using differential
   evolution, prices vanilla Europeans via semi-analytic Fourier inversion,
   and prices exotic payoffs (digital, barrier, one-touch) via
   full-truncation Euler Monte Carlo.

The app supports a **daily file workflow**: place one Bloomberg Excel file per
day in the `data_vol_surface/` folder and switch between dates directly in the
sidebar without restarting the server.

The bundled dataset covers the **S&P 500 Index (SPX)** with:

| Property | Value |
|---|---|
| Available snapshots | 14 Apr 2026, 20 Apr 2026 |
| Near-term forward | ~6 785 (14 Apr) / ~7 117 (20 Apr) |
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
| **Dupire local vol** | Full Gatheral-formulation local vol surface built from per-expiry SVI fits (the per-slice fit family Bloomberg OVME uses, not the global SSVI alternative) with analytical x-derivatives and PCHIP dw/dT; three sub-views (3-D surface, LV vs IV slice, ATM term structure) |
| **Heston calibration** | Fits the five Heston parameters (v₀, κ, θ, ξ, ρ) to the market IV surface via differential evolution + L-BFGS-B polish; configurable maturity and moneyness filters |
| **Heston pricing** | Semi-analytic European call/put via Fourier inversion (Albrecher formulation); full-truncation Euler MC for vanilla, digital, barrier, and one-touch payoffs |
| **One-touch pricing** | Pay-at-hit one-touch (FX market convention) — up or down direction — under both Dupire LV and Heston engines, with per-path discount from hit time τ |
| **Yield curve** | Auto-fetches the US Treasury CMT par yield curve from treasury.gov for the snapshot date, bootstraps to continuous-compounded zero rates (T-bills passed through, T-notes solved by recursive coupon-bond bootstrap), and exposes an editable 9-tenor table for user overrides.  All pricing and arbitrage checks consume `r(T)` per option maturity via log-linear interpolation in discount-factor space. |
| **Arbitrage checks** | Calendar spread, butterfly spread, and vertical spread checks with severity scores |
| **Violation overlay** | Select any expiry to see its smile with violations marked directly on the chart |
| **Export** | Download violation reports as a multi-sheet Excel file |

---

## 3. Project Structure

```
vol_surface_excel/
├── data_vol_surface/                  # One Bloomberg Excel file per snapshot date
│   ├── vol_surface_14_04_2026.xlsx
│   └── vol_surface_20_04_2026.xlsx
├── pyproject.toml                     # uv / PEP 517 project config
├── README.md
├── src/
│   ├── data_loader.py        # Excel ingestion, validation, long-format output
│   ├── iv_surface_builder.py    # Grid interpolation (bicubic spline) + per-slice SVI overlay
│   ├── montecarlo.py         # Monte Carlo pricer (vanilla / digital / barrier / one-touch) via Dupire local vol
│   ├── local_vol.py          # Dupire local vol via per-slice SVI + PCHIP dw/dT
│   ├── heston.py             # Heston model: characteristic function, Fourier pricing, calibration, MC
│   ├── rates.py              # US Treasury yield curve fetch, bootstrap, log-DF interpolation
│   ├── dividends.py          # Implied dividend curve via parity extraction from Bloomberg forwards
│   ├── plots.py              # All Plotly chart functions
│   ├── arbitrage_checks.py   # Calendar / butterfly / vertical checks
│   └── streamlit_app.py      # UI entry-point
└── tests/
    ├── test_data_loader.py
    ├── test_surface_builder.py
    ├── test_arbitrage_checks.py
    ├── test_heston.py        # Black-76, CF, pricing, smile, Heston MC + OT (38 tests)
    ├── test_montecarlo.py    # Dupire LV one-touch tests (10 tests)
    ├── test_rates.py         # BEY conversion, coupon-bond bootstrap, log-DF interp, 4Y interp post-fill (36 tests)
    └── test_dividends.py     # Implied div extraction, OLS canonical-tenor fit, calibration q-invariance (29 tests)
```

**Data flow:**

```
data_vol_surface/*.xlsx   (or uploaded file)
          │
          ▼
   data_loader.py  ─────────────────────────────────────────────┐
          │                                                      │
          ▼                                                      ▼
 iv_surface_builder.py ─────────► heston.py              local_vol.py
          │                           │                          │    \
          └──────────────────┬────────┴──────────────────────────┘     ▼
                             ▼                                    montecarlo.py
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
| scipy | ≥ 1.13 | Spline interpolation, PCHIP, Black-76, per-slice SVI optimisation |
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
| 2 | Absolute strike prices per moneyness level — used by the loader to recover the spot via `S = median(K_i / m_pct_i)` (more accurate than the earliest forward) |
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
| **Rate curve** | Bootstrapped from US Treasury CMT on the snap date | Editable 10-tenor zero-rate table (1M, 2M, 3M, 4M, 6M, 1Y, 2Y, 3Y, 4Y, 5Y).  Auto-fetched from treasury.gov for the snapshot date and bootstrapped to continuous-compounded zeros; the 4Y row (which CMT doesn't publish) is filled by log-DF interpolation between 3Y and 5Y.  User can override any cell.  See [§12](#12-yield-curve--theory--implementation). |
| **Dividend curve** | Implied from Bloomberg forwards and the rate curve | Editable 10-tenor table at the same canonical tenors as the rate curve.  Values come from an OLS line `q(T) = a + b·T` fit (T-weighted) to all per-expiry parity-implied points `q = r(T) − ln(F(T)/S)/T` with `T ≥ 1/52`.  The fit is dominated by the long end where dividend timing has averaged out, so the displayed term structure is smooth.  Editor remounts (and re-fills from the OLS) whenever a rate cell is edited, since `q` is *implied from* `r`.  See `src/dividends.py` for the parity-extraction and OLS details. |
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

### 7.6 LV Surface Tab

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

### 7.7 LV MC Pricer Tab

Prices European options on the Dupire local vol surface via Monte Carlo
simulation.  The local vol grid (resolution 100) is shared with the Local
Volatility tab — no extra compute if already cached.

**Supported payoffs:**

| Payoff family | Variants | Combinable with |
|---|---|---|
| Vanilla | Call / Put | Barrier (up/down × in/out, American or European monitoring) |
| Digital (cash-or-nothing) | Call / Put | — (digitals pay only on terminal ITM; barrier controls are hidden in the UI) |
| One-Touch | Up / Down | — (the OT *is* a barrier product; pay-at-hit, FX market convention) |

**Inputs:**

| Input | Default | Description |
|---|---|---|
| **Payoff** | Vanilla | Segmented control: `Vanilla` / `Digital` / `One-Touch` |
| **Option type** (vanilla / digital) | Call | Segmented control: `Call` or `Put` |
| **Direction** (one-touch only) | Up | Segmented control: `Up` (B above spot) or `Down` (B below spot) |
| **Cash payout** (digital / one-touch) | 1.0 | For digital: paid at expiry if ITM.  For OT: paid on first barrier touch (pay-at-hit). |
| **Barrier** (vanilla only) | None | `None`, `Up-and-Out`, `Up-and-In`, `Down-and-Out`, `Down-and-In` |
| **Barrier level B** (barrier or OT) | S₀ × 1.10 (up) or 0.90 (down) | Absolute barrier level |
| **Barrier monitoring** (barrier only) | American | `American` (every Euler step) or `European` (terminal spot only) |
| **Spot S₀** | From dataset | Pre-filled from the loaded Bloomberg snapshot |
| **Strike K** (vanilla / digital) | ATM (= S₀) | Absolute strike level — hidden for OT |
| **Maturity T (years)** | 1.0000 | Option maturity entered to 4 decimals (e.g. 0.0822 ≈ 30 days); surface min `_T_MIN_CUTOFF`, currently `14 / 365` ≈ 14 days |
| **Risk-free rate r (%)** | from curve at T | Reads `r(T)` off the bootstrapped Treasury curve by default; uncheck "Use curve r at T" to enter a flat scalar manually |
| **Dividend yield q (%)** | from curve at T | Reads `q(T)` off the parity-implied dividend curve by default; uncheck "Use curve q at T" to enter a flat scalar manually |
| **Steps per year** | 252 (daily) | Euler discretisation steps; alternative: 52 (weekly) |
| **Convergence ε** | 0.001 | Relative stopping threshold; capped at 0.01 (1 %) — looser tolerances are not useful in practice (95 % CI half-width ≈ 2 %) |

The risk-free rate is shared with the sidebar (same value used for the
arbitrage checks).

**Barrier monitoring styles:**

- **American** — the barrier is checked at every Euler step.  Discrete daily
  sampling under-estimates the hit probability vs continuous monitoring, but
  reproduces the behaviour of most traded barrier contracts.
- **European** — the barrier is checked only at expiry (`S_T` alone decides
  hit / no-hit).  A knock-out option then behaves like a vanilla one with a
  payoff extinguished only if the terminal spot breaches the barrier.

**One-touch convention (pay-at-hit).**

The cash payout is delivered **the moment** spot first touches the barrier,
discounted from the hit time `τ` back to today: `PV = E[e^{−rτ}·X·1{hit}]`.
This is the standard FX-market convention.  Internally the MC tracks a
per-path `first_hit_step` (initialised to −1 for "not hit", set to the
step index of the first crossing) and applies a path-specific discount at
payoff time rather than the global `e^{−rT}`.  Day-zero hits (spot already
past the barrier at inception) discount with `τ = 0`, so the price equals
the cash payout exactly with no discount.  Monitoring is American only
(European-monitored OT degenerates into a digital and isn't a real product).

**Convergence rule:**

Paths run in batches of 500.  After each batch, stop when the running
relative standard error meets the precision target:

```
SE(n) / max(|P(n)|, 1e-8) < ε   and   n ≥ 5 000
```

`SE(n)` is computed from the running variance of antithetic *pair averages*
`Y_j = (X_a + X_b)/2` (each batch contains 250 such pairs, with `X_a` and
`X_b` driven by `Z` and `−Z` respectively): `Var(P̂) = Var(Y) / M` where
`M = N/2` is the pair count.  This captures the antithetic correlation
correctly — for a vanilla call where the pair correlation `ρ_pair ≈ −0.95`,
the pair-aware SE is ~5× tighter than the marginal SE; for digital /
one-touch payoffs `ρ_pair ≈ 0` and the two coincide.  This is a true 1-σ
half-width bound on the estimator, not a batch-to-batch stability check.
Hard cap: 200 000 paths.

**Surface coverage warnings** are shown before running if the strike moneyness
or maturity falls outside the local vol surface domain.  A barrier sanity
check warns if the barrier is on the wrong side of spot at inception.

**Outputs:** Price (label dynamically reflects flavour and style, e.g.
"(American) Up-and-Out Digital Call Price" or "Up One-Touch Price") ·
Standard error · 95%
confidence interval · Paths simulated · Convergence badge (green = converged,
amber = cap reached) · % of steps clamped to the surface boundary.

---

### 7.8 Heston Pricer Tab

Calibrates the five Heston stochastic-volatility parameters to the full market
IV surface, then prices European (Fourier) and exotic (Monte Carlo) options
under that calibrated measure.  See
[Section 11](#11-heston-stochastic-volatility--theory--implementation) for the
mathematical details.

**Calibration controls:**

| Control | Default | Description |
|---|---|---|
| **Min T (years)** | 0.05 (≈ 18 days) | Excludes ultra-short expiries where smile noise dominates |
| **Moneyness range (%)** | 70 – 130 | Restricts the fit to liquid strikes; deep-OTM points are removed |
| **Run calibration** | — | Launches differential evolution + L-BFGS-B polish on the filtered surface |

Calibration typically takes **~60–90 s** on a filtered SPX surface (~400
points after filters).  Parameters and fit quality are cached for the duration
of the session so subsequent pricing does not re-calibrate.

**Calibrated parameters:**

Five metric cards display the best-fit values with useful transforms in the
subtitle:

| Metric | Meaning | Subtitle |
|---|---|---|
| **v₀** | Initial variance | σ₀ = √v₀ (equivalent spot vol) |
| **κ** | Mean-reversion speed of variance | — |
| **θ** | Long-run variance | σ∞ = √θ (equivalent long-term vol) |
| **ξ** | Vol-of-vol | — |
| **ρ** | Spot–vol correlation | — |

Below the metrics three diagnostic badges are shown:

- **RMSE (pp)** — root-mean-square IV error on the calibrated region
  (filtered by Min T and moneyness range), expressed in percentage points
- **MaxErr (pp)** — worst single-point IV error on the calibrated region
- **Feller** — green if `2κθ > ξ²` (variance stays strictly positive under
  the SDE), amber if violated (variance can touch zero; the MC scheme uses
  full truncation to handle this safely)

Two sub-tabs are available:

#### Smile Fits

Side-by-side comparison of the market SVI smile and the calibrated Heston
smile for a selected expiry.  Market vols (from the dense IV surface) appear
as markers; the Heston model smile is overlaid as a solid line.  Residuals
are shown beneath as a thin strip.  Useful for spotting whether the global
fit is biased systematically at certain maturities.

#### MC Pricer

Same payoff menu as the [LV MC Pricer tab](#77-lv-mc-pricer-tab) (vanilla /
digital / barrier × call / put × American / European monitoring, plus
one-touch up/down with pay-at-hit cash settlement), but paths are generated
from the full-truncation Heston Euler scheme instead of the Dupire local vol
surface.

For **vanilla European calls and puts**, the semi-analytic Fourier price is
also shown (no simulation noise); compare it to the MC price to gauge
convergence.  For digitals, barriers, and one-touch, only the MC price is
available — none of these payoffs has a closed-form Heston counterpart in
this codebase.

Outputs mirror the Dupire tab: price, standard error, 95 % CI, paths
simulated, convergence badge.  There is no boundary-clamping metric here
because the Heston scheme does not rely on a finite grid.

---

### 7.9 Arbitrage Checks Tab

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

**Economic intuition:** the value of an infinitesimal butterfly spread is
proportional to `d²C/dK²`, which by the Breeden–Litzenberger identity equals
`e^{-rT}·f(K, T)` where `f` is the risk-neutral density of the terminal
spot.  A density must be non-negative, so call prices must be convex in
strike at every expiry.  Convexity of the *implied vol* in `K` is **neither
necessary nor sufficient** — typical equity put-skews are concave in `K`
on the put wing yet entirely arbitrage-free, so an IV-space convexity test
flags false positives and misses real arbitrages.

**Condition:**

```
d²C / dK² ≥ 0   for all interior K at each expiry T
```

**Implementation:** convert each slice's quoted IVs to Black-76 call prices
using the slice-specific zero rate `r(T)` from the rate curve, then apply a
non-uniform centred finite second difference (the moneyness grid is non-
uniform: 60, 80, 90, 95, 97.5, 100, 102.5, 105, 110, 120, 130, 140):

```
d²C/dK²  ≈  2/(h₁+h₂)  ×  [C(K+h₂)/h₂  −  C(K)·(1/h₁ + 1/h₂)  +  C(K−h₁)/h₁]
```

where `h₁ = K − K_prev` and `h₂ = K_next − K` are absolute strike gaps
(Breeden–Litzenberger lives in `K`-space, so we work with `K = m_pct/100·F`
not the moneyness percentage).

**Tolerance:** violations are reported only when `d²C/dK² < −1×10⁻⁴` to
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

### 10.2 Why per-slice SVI?

With only ~12 moneyness quotes per expiry (60 %–140 % in the Bloomberg sheet),
numerical second derivatives of the implied vol smile are highly sensitive to
the non-uniform strike spacing.  The cluster around 95–100 % causes
`CubicSpline` to produce wild `d²w/dx²` values, making the denominator `g`
ill-conditioned.

Two parametric alternatives exist:

  · **Per-slice SVI** (Gatheral): five parameters `(a, b, ρ, m, σ)` per
    expiry.  Each slice is fit independently and retains its own shape
    — that is what produces the front-end skew bumps and off-ATM
    curvature variation visible in Bloomberg OVME's local-vol surface.
    No cross-slice constraint, so adjacent slices can in principle
    cross in `w(·, T)` (calendar arbitrage); in practice this is rare
    on real Bloomberg quotes and the Dupire formula flags any
    occurrence as `g ≤ 0`, which we mask to NaN.
  · **SSVI** (Surface SVI, Gatheral–Jacquier 2014): one ATM curve `θ(T)`
    plus three globals `(ρ, η, γ)` shared across all expiries, with
    `θ·φ(θ)·(1+|ρ|) ≤ 4` and monotone `θ(T)` giving calendar / butterfly
    no-arb by construction.  Theoretically clean but only `3 + N_T`
    degrees of freedom for the entire surface — per-slice idiosyncrasies
    get averaged into one global skew function and disappear.  The
    resulting LV surface is visibly too smooth versus a desk-grade
    benchmark like Bloomberg OVME.

We use **per-slice SVI** here, matching Bloomberg's choice and most
production desks.

The SVI form per slice:

```
w(x) = a + b · (ρ·z + √(z² + σ²))    where z = x − m
```

Analytical x-derivatives:

```
∂w/∂x   = b · (ρ + z/d)              where d = √(z² + σ²)
∂²w/∂x² = b · σ² / d³                always ≥ 0 when b, σ > 0
```

Because `∂²w/∂x² ≥ 0` by construction (for a well-fitted SVI), the
denominator `g` is much better-conditioned than with numerical finite
differences.

### 10.3 Numerical Pipeline

The local vol computation proceeds in four steps:

**Step 1 — Per-slice SVI fit (x-derivatives)**

For each of the `N_T` raw expiry slices, fit SVI parameters `{a, b, ρ, m, σ}`
by minimising relative squared error.  Five SPX-flavoured starts feed
L-BFGS-B; the best residual wins.  Evaluate `w`, `∂w/∂x`, `∂²w/∂x²`
analytically on a fine grid of 200 log-moneyness points `x_fine`.

If a slice's SVI fit is poor (mean relative squared error > 0.04, or
fitted `w` not strictly positive on the fine grid), fall back to linear
interpolation of raw `w` plus numerical `np.gradient` for the derivatives.
This preserves local fidelity to the noisy quotes rather than forcing a
bad parametric fit.

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

## 11. Heston Stochastic Volatility — Theory & Implementation

### 11.1 Model Dynamics

The Heston (1993) model posits that the spot price `S_t` and its instantaneous
variance `v_t` follow the joint SDE

```
dS_t  = (r − q) · S_t · dt  +  √v_t · S_t · dW_t^S
dv_t  = κ · (θ − v_t) · dt  +  ξ · √v_t · dW_t^v
⟨dW^S, dW^v⟩ = ρ · dt
```

with five free parameters:

| Symbol | Role | Typical sign / range |
|---|---|---|
| **v₀** | Initial variance | ≥ 0 |
| **κ** | Speed of mean reversion of `v` toward `θ` | > 0 |
| **θ** | Long-run variance | > 0 |
| **ξ** | Vol-of-vol (diffusion coefficient of `v`) | > 0 |
| **ρ** | Correlation between spot and variance shocks | typically < 0 for equities |

The **Feller condition** `2κθ > ξ²` guarantees that `v_t` stays strictly
positive almost surely; when it is violated, `v_t` can reach zero but cannot
become negative thanks to the `√v` diffusion.  The app reports Feller status
explicitly and uses a full-truncation scheme for MC to handle either regime.

### 11.2 Semi-Analytic Pricing

Under Heston, European call prices admit the closed-form representation

```
C(S, K, T) = S · e^(−qT) · P₁  −  K · e^(−rT) · P₂
```

where `P₁` and `P₂` are risk-neutral probabilities obtained by Fourier
inversion of the characteristic function `f_j(u; T, x, v)`:

```
P_j = ½ + (1/π) · ∫₀^∞ Re[ e^(−i u ln K) · f_j(u) / (i · u) ] du     (j = 1, 2)
```

This implementation uses the **Albrecher et al. (2007) "little trap"**
formulation of the characteristic function, which avoids the branch-cut
instability of the original Heston form across the complex square root of
`d(u)`.  Its closed form is:

```
f_j(u) = exp( C_j(u, T) + D_j(u, T) · v₀ + i · u · x )

where x = ln(S) + (r − q)·T

C_j(u, T) = (r − q) · i · u · T
          + (κθ / ξ²) · [ (b_j − ρξiu − d) · T
                          − 2 · ln((1 − c · e^(−dT)) / (1 − c)) ]

D_j(u, T) = [(b_j − ρξiu − d) / ξ²] · (1 − e^(−dT)) / (1 − c · e^(−dT))

d   = √( (ρξiu − b_j)²  −  ξ² · (2·u_j·iu − u²) )
c   = (b_j − ρξiu − d) / (b_j − ρξiu + d)

u_1 = +½,  b_1 = κ − ρξ
u_2 = −½,  b_2 = κ
```

Puts are priced by put-call parity:

```
P = C  −  S·e^(−qT)  +  K·e^(−rT)
```

The Fourier integral is computed numerically via `scipy.integrate.quad` with
generous absolute and relative tolerances (`1e-8`); a warm-up call amortises
JIT / import overhead across batched pricing.  A vectorised routine prices an
entire smile (one T, many K) in a single pass by re-using the CF evaluation.

### 11.3 Calibration Strategy

The calibration minimises a **vega-weighted price error** objective across
the filtered market surface:

```
minimise  Σ_{i} [ (C_model(K_i, T_i) − C_market(K_i, T_i)) / vega_i ]²
```

Since `ΔC ≈ vega · Δσ` to first order, dividing the price error by each
point's Black-76 vega converts price RMSE into an approximate **implied vol
RMSE** — so the optimiser effectively fits the surface in IV space without
paying the cost of a Newton inversion at every point during the search.

**Vega floor:** extremely deep OTM points have `vega ≈ 0`, which would
explode their weight in the objective.  A **single global 25th-percentile
floor** is applied: any point whose raw vega falls below that level is
clipped to the floor (so the *weight* `1/vega` is capped from above),
preventing noise from dominating the fit.  With the T / moneyness filters
already applied upstream, a per-slice floor offered no additional benefit
and is harder to defend than one transparent global scalar.

**Filters:**

| Filter | Default | Purpose |
|---|---|---|
| **Min T** | 0.05 yr (≈ 18 days) | Drops ultra-short expiries where smile noise and event premia dominate |
| **Moneyness range** | 70 % – 130 % | Drops deep-OTM and deep-ITM quotes with negligible vega or illiquid pricing |

Filters are configurable in the sidebar.  The reported RMSE and MaxErr refer
to the **calibrated region only**.

**Optimiser:** SciPy's `differential_evolution` (global, stochastic) with
parameter bounds

```
v₀ ∈ [0.001, 0.50]   κ ∈ [0.1, 10.0]   θ ∈ [0.001, 0.50]
ξ  ∈ [0.05, 3.0]     ρ ∈ [−0.99, 0.20]    (equity skew is always negative)
```

followed by an automatic L-BFGS-B polish from the DE best point (`polish=True`).
The initial population is seeded via `x0 = [σ_ATM_front², 2.0, σ_ATM_long², 0.5,
−0.70]` — a crude but effective warm start built from the front-month and
long-dated ATM implied vols.  Typical wall-clock time on the bundled SPX
surface (filtered to ~400 points) is **~60–90 s**.  To keep the DE inner loop
tractable, at most `max_slices=12` expiries are used for fitting (evenly
spaced across the filtered range); the final fit-quality report is then
re-evaluated on the **full** surface with a finer 1000-point φ-grid.

### 11.4 Monte Carlo Scheme

Both Fourier and MC prices are available for vanilla European payoffs, but
digital and barrier options are priced exclusively by MC.  The scheme is the
**full-truncation Euler discretisation** of Lord–Koekkoek–van Dijk (2010),
which handles violated-Feller regimes robustly by setting `max(v_t, 0)`
wherever the variance process would otherwise go negative:

```
v_{t+Δt} = v_t + κ·(θ − v_t^+)·Δt  +  ξ·√(v_t^+)·√Δt · Z_v
S_{t+Δt} = S_t · exp[ (r − q − ½·v_t^+)·Δt  +  √(v_t^+)·√Δt · Z_S ]

where v_t^+ = max(v_t, 0)
      Z_S, Z_v ~ bivariate normal with correlation ρ
```

Running accumulators of antithetic *pair averages* `Y_j = (X_a + X_b)/2` are
used in place of a growing Python list, so memory stays constant as paths are
added.  The pair-aware running SE absorbs the antithetic correlation
correctly (see the LV MC section above for the derivation).  Each batch of
500 paths is drawn as 250 jointly-negated `(Z₁, Z₂)` Gaussian trajectories,
which preserves the `(ρ, √(1−ρ²))` rotation while flipping the sign of every
Brownian increment.  The same SE-based precision rule as the Dupire MC applies:

```
SE(n) / max(|P(n)|, 1e-8) < ε   and   n ≥ 5 000
```

with a hard cap of 400 000 paths and a default batch size of 500 (the cap
is higher than the Dupire MC's 200 000 because Heston path generation is
cheaper per step — no local-vol grid interpolation — and digital /
one-touch payoffs routinely need the extra headroom under a true
precision bound).

For **vanilla European calls and puts**, the Fourier price serves as a
reference — the test suite asserts that the MC price lies within `3 × SE` of
the Fourier value, which is a strong end-to-end sanity check on both
pipelines.

---

## 12. Yield Curve — Theory & Implementation

A flat single risk-free rate is a poor approximation when option maturities
span four orders of magnitude (1 day to 5+ years).  The app builds a
proper continuously-compounded zero curve for the snapshot date and
queries it at each option's maturity, so a 3-month digital and a 5-year
barrier discount with their *own* rates rather than sharing a sidebar slider.

### 12.1 Data Source

The U.S. Department of the Treasury publishes the **Daily Treasury Par
Yield Curve Rates** (CMT — Constant Maturity Treasury) as a CSV download
keyed by calendar year:

```
https://home.treasury.gov/resource-center/data-chart-center/
interest-rates/daily-treasury-rates.csv/{YYYY}/all
    ?type=daily_treasury_yield_curve
    &field_tdr_date_value={YYYY}
    &page&_format=csv
```

This is the canonical, free, no-API-key source for U.S. risk-free rates.
Each row is one business day; columns are the published tenors (1m, 2m,
3m, 4m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y).  We keep only the tenors
at or below 5y matching the project scope:

```
1M, 2M, 3M, 4M, 6M, 1Y, 2Y, 3Y, 5Y     (9 tenors)
```

The fetch is cached on the snapshot date (`@st.cache_data`) — for any
historical date the Treasury rates never change, so the network round-trip
happens at most once per session per file.  If the snapshot date falls on
a weekend or holiday (Treasury doesn't publish), we use the most recent
preceding business day.  If the fetch fails entirely (offline, future date,
treasury.gov down), the UI falls back to a manual-entry mode pre-populated
with a flat 4.5 % curve and shows a clear warning banner.

### 12.2 Bootstrap

Treasury reports yields on a **bond-equivalent** basis (semi-annual
compounding).  Two regimes:

- **T-bills** (≤ 1y) are zero-coupon — the par yield equals the zero rate.
  Convert from BEY to continuous compounding once:
  ```
  r_cc = 2 · ln(1 + y_BEY / 2)
  ```

- **T-notes** (2y, 3y, 5y) are coupon bonds.  The published yield is a
  *par yield* — the coupon rate that would price the bond at par.  We
  bootstrap recursively in increasing maturity order: solve for the
  continuous zero rate `r(T)` such that a semi-annual coupon bond with
  coupon rate `y_par/2` and face value 1 prices to exactly 1.  Coupon
  dates between previously bootstrapped knots are filled by linear
  interpolation in `log(DF)` space using a *candidate* `r(T)` — the
  resulting equation is then a single 1-D root in `r(T)`, solved via
  Brent's method.

The self-consistency property of the bootstrap is verified in the test
suite: each par bond, repriced under its own bootstrapped curve, is
within `1e-9` of par.

### 12.3 Interpolation

`RateCurve.zero_rate(T)` queries the curve at any maturity by linear
interpolation in `log(DF)` space:

```
log DF(T)  =  log DF(T_below)
            + (T − T_below) / (T_above − T_below) · (log DF(T_above) − log DF(T_below))
```

This is the **standard market convention** because it is equivalent to
piecewise-constant forward rates between knot points — a smoother curve
than linear-in-rates without the oscillations of a cubic spline, and the
implied forward rates remain non-negative and well-behaved.  Beyond the
curve range we extrapolate flat (the boundary rate), which is a
conservative choice for European pricing.

### 12.4 Use in Pricing

For every option of maturity `T`, callers query `curve.zero_rate(T)` and
use the resulting scalar `r` as a constant rate for that pricing call.
This is option (α) in the standard taxonomy — appropriate for European
products where rate-vol risk is not being managed separately.  The
alternative (using instantaneous forward rates `f(t)` in the drift at
each MC step) is more sophisticated but only matters for strongly
path-dependent payoffs (American exercise, accrual notes); not relevant
here.

Specifically:

- **MC pricing** (Dupire LV and Heston): `r = curve.zero_rate(option_T)` is
  passed as the constant rate for the entire path.  Drift `(r − q)·dt` and
  terminal discount `e^{−rT}` use the same single rate.
- **Heston Fourier pricing**: same — `r` is per-option.
- **Heston calibration**: each expiry slice is priced and discounted with
  its own `r(T_slice)`.  Both vega-weighted price errors during DE and
  the IV residuals reported in the detail table use the slice-appropriate
  rate.
- **Vertical spread arbitrage check**: each expiry slice gets its own
  Black-76 rate when computing call prices for the no-arbitrage test.
  The check report explicitly shows the rate range used (e.g. `r∈[4.30 %,
  4.75 %]`).

A user override of any cell in the curve table re-flows automatically
through every pricing call: edit the 1y rate, recompute, every option
with `T ≈ 1y` re-prices on the next click.

---

## 13. Module Reference

### `data_loader.py`

| Function / Class | Description |
|---|---|
| `load_workbook(path, snap_date)` | Main entry point.  Returns `VolDataset(vol_df, metadata)` |
| `_parse_vol_surface(raw, snap)` | Melts the wide-format sheet into a tidy long DataFrame |
| `_extract_spot_from_strike_grid(raw)` | Internal: derives the reference spot from row 1's absolute-strike grid (`S = median(K_i / m_pct_i)` across columns).  More accurate than `F(earliest)`, which differs from spot by `exp((r−q)·T_earliest)` |
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
| `_svi_fit_slice(x, w_obs)` | Internal: fits Gatheral SVI to one slice's `(x, w)` pairs; returns param dict or `None` if rejected |
| `_svi_eval(p, x)` | Internal: evaluates `(w, ∂w/∂x, ∂²w/∂x²)` analytically from SVI params at fixed T |
| `LocalVolGrid` | `namedtuple('LocalVolGrid', ['K_grid', 'T_grid', 'LV_grid', 'IV_grid', 'moneyness', 'expiries', 'arb_mask'])` |

**Key constants:**

| Constant | Value | Role |
|---|---|---|
| `_LV_CAP` | 1.50 | Hard ceiling (150 %) — values above are numerical artefacts |
| `_G_FLOOR` | 0.02 | Denominator guard — points with `g ≤ 0.02` are masked |
| `_W_FLOOR` | 1e-8 | Floor on total variance `w` before computing `g` |
| `_DWT_FLOOR` | 1e-6 | Floor on `∂w/∂T` to prevent holes from numerical noise |
| `_T_MIN_CUTOFF` | `14 / 365` | ~14 calendar days (~2 weeks) — very short expiries excluded (unstable PCHIP derivatives on too-few T-neighbours) |
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

### `heston.py`

| Function / Class | Description |
|---|---|
| `heston_call(S, K, T, r, q, v0, kappa, theta, xi, rho, n_phi=1000)` | Semi-analytic European call price via Fourier inversion of the Albrecher CF (scalar K) |
| `heston_smile(S, T, r, q, params, m_pct_grid, n_phi=1000)` | Vectorised smile: computes Heston IVs on a moneyness-% grid for one expiry in a single pass (one CF evaluation, batched across strikes); used for plotting market vs model |
| `calibrate(vol_df, S, r, q, max_slices=12, min_T=0.05, m_range=(70.0, 130.0))` | Global calibration: DE + L-BFGS-B polish on vega-weighted price error.  Returns `HestonCalibResult` |
| `mc_heston(params, S0, K, T, r, q, option_type, is_digital, digital_payout, barrier_type, barrier_level, barrier_style, is_one_touch, one_touch_direction, steps_per_year, eps, seed)` | Full-truncation Euler MC for vanilla / digital / barrier / one-touch payoffs (call or put).  Returns `MCResult` |
| `_heston_cf(phi, T, v0, kappa, theta, xi, rho, j)` | Internal: Heston log-spot CF `f_j(φ)` — Albrecher "little trap" form |
| `_heston_call_fwd_batch(F, K_arr, T, r, v0, kappa, theta, xi, rho, phi)` | Internal: vectorised call prices over strikes at fixed expiry using the market forward |
| `_b76_call`, `_b76_vega`, `_b76_call_vec`, `_b76_vega_vec`, `_b76_iv`, `_b76_iv_vec` | Internal: Black-76 pricing, vega, and Newton-Raphson IV inversion — scalar and vectorised variants |
| `HestonParams` | `namedtuple('HestonParams', ['v0', 'kappa', 'theta', 'xi', 'rho'])` |
| `HestonCalibResult` | `namedtuple('HestonCalibResult', ['params', 'rmse_iv', 'max_err_iv', 'mean_err_iv', 'n_points', 'success', 'detail_df'])` — `detail_df` holds per-point market vs model IVs across the full surface |

**Puts** are not a separate function — they are derived by the caller via
put-call parity: `P = C − S·e^(−qT) + K·e^(−rT)` (see the Heston MC Pricer
sub-tab in [streamlit_app.py](src/streamlit_app.py)).  The MC pricer
(`mc_heston`) handles both calls and puts directly via `option_type`.

**Feller condition** is computed on-demand by the caller from
`2·κ·θ > ξ²` on the returned `HestonParams` — it is not stored in
`HestonCalibResult` because it is a trivial one-line derivation.

**Key constants:**

| Constant | Value | Role |
|---|---|---|
| `_MC_MIN_PATHS` | 5 000 | Minimum paths before convergence can trigger |
| `_MC_MAX_PATHS` | 400 000 | Hard cap — `converged=False` if reached (higher than Dupire MC because Heston paths are cheaper per step and digital / OT payoffs need extra headroom under the SE-based precision rule) |
| `_MC_BATCH` | 500 | Paths per batch; also the convergence window |
| `_MC_EPS_FLOOR` | 1e-8 | Prevents division by zero when price ≈ 0 |

---

### `montecarlo.py`

| Function / Class | Description |
|---|---|
| `price_european_option(lv_grid, S0, K, T, r, q, option_type, is_digital, digital_payout, barrier_type, barrier_level, barrier_style, is_one_touch, one_touch_direction, steps_per_year, eps, seed)` | Runs the MC simulation for vanilla / digital / barrier / one-touch options (call or put) and returns `MCResult` |
| `_build_lv_interpolator(lv_grid)` | Internal: builds a `RegularGridInterpolator` from `LocalVolGrid`, filling NaN holes |
| `MCResult` | `namedtuple('MCResult', ['price', 'std_error', 'n_paths', 'converged', 'clamp_pct'])` |

**Option flavours supported:**

| Flavour | Triggered by | Payoff |
|---|---|---|
| Vanilla call/put | `is_digital=False`, `barrier_type=None`, `is_one_touch=False` | `max(S_T − K, 0)` / `max(K − S_T, 0)` |
| Digital call/put | `is_digital=True` | `digital_payout · 1_{S_T > K}` / `digital_payout · 1_{S_T < K}` |
| Barrier (any of the above) | `barrier_type` ∈ `up_out`, `up_in`, `down_out`, `down_in` + `barrier_level` | Knock-in / knock-out applied to the underlying vanilla/digital payoff |
| Barrier monitoring | `barrier_style` = `"american"` (default, every Euler step) or `"european"` (terminal spot only) | — |
| One-touch | `is_one_touch=True`, `one_touch_direction` ∈ `"up"`, `"down"`, plus `barrier_level` and `digital_payout` | `digital_payout · e^{−r·τ} · 1{barrier hit during [0, T]}` — pay-at-hit, FX market convention.  Mutually exclusive with `is_digital` and `barrier_type`. |

**Key constants:**

| Constant | Value | Role |
|---|---|---|
| `_MIN_PATHS` | 5 000 | Minimum paths before convergence can trigger |
| `_MAX_PATHS` | 200 000 | Hard cap — `converged=False` if reached |
| `_BATCH_SIZE` | 500 | Paths per batch; also the convergence window |
| `_EPS_FLOOR` | 1e-8 | Prevents division by zero when price ≈ 0 |

---

### `arbitrage_checks.py`

| Function | Returns |
|---|---|
| `check_calendar_spread(df)` | `CheckResult` |
| `check_butterfly_spread(df, r, eps)` | `CheckResult` — `r` (scalar or `RateCurve`) is required: each slice's IVs are converted to Black-76 call prices using `r(T)` and `d²C/dK²` is checked (Breeden–Litzenberger) |
| `check_vertical_spread(df, r)` | `CheckResult` — `r` may be a scalar (flat) or a `RateCurve` (slice-by-slice) |
| `run_all_checks(df, r)` | `list[CheckResult]` — always `[calendar, butterfly, vertical]`; `r` is required, no silent default |

```python
CheckResult = namedtuple('CheckResult', ['name', 'passed', 'violations', 'details'])
```

Each violation dict contains at minimum: `expiry_label`, `moneyness_pct`,
`time_to_expiry`, `severity`, plus check-specific fields.  Vertical-spread
violations also carry `r_used` (the slice-specific rate that produced the
Black-76 call price).

---

### `rates.py`

| Function / Class | Description |
|---|---|
| `RateCurve` | `NamedTuple('RateCurve', ['tenors_yr', 'zero_rates', 'source', 'snap_date_iso'])` with methods `.zero_rate(T)`, `.discount_factor(T)`, `.as_table()` |
| `fetch_treasury_yields(snap_date, timeout=15)` | Download the Daily Treasury Par Yield Curve CSV for `snap_date.year`, locate the row matching `snap_date` (or the most recent preceding business day), return `{tenor_label: par_yield_pct}`.  Raises `TreasuryFetchError` on network/parse failure |
| `bootstrap_zero_rates(par_yields_pct)` | T-bills passed through with BEY → continuous conversion; T-notes solved by recursive coupon-bond bootstrap with Brent's method.  Returns `{T_years: continuous zero rate}` |
| `build_curve_from_yields(par_yields_pct, snap_date, source='treasury')` | Convenience wrapper: bootstrap + assemble a `RateCurve` |
| `build_curve_from_zero_rates(zeros_pct, snap_date, source='manual')` | Build a `RateCurve` from already-continuous zero rates (used when the user types overrides into the sidebar table) |
| `flat_curve(rate, snap_date)` | Constant-rate curve at every canonical tenor — fallback / test helper |
| `get_rate(curve_or_scalar, T)` | Duck-typed accessor — returns `curve.zero_rate(T)` if it has the attribute, else `float(curve_or_scalar)`.  Used by `arbitrage_checks` and `heston.calibrate` to support both flat and curved inputs |
| `_interp_log_df(T, tenors, rates)` | Internal: linear interpolation in `log(DF)` space (piecewise-flat forwards), with flat extrapolation outside the curve |
| `TENORS`, `TENOR_LABELS`, `TENOR_YEARS` | Canonical tenor list: `1M, 2M, 3M, 4M, 6M, 1Y, 2Y, 3Y, 4Y, 5Y` (10 tenors).  4Y is included even though US Treasury CMT doesn't quote it — `build_curve_from_yields` post-fills it by log-DF interpolation between the surrounding bootstrapped knots |
| `TreasuryFetchError` | Raised by `fetch_treasury_yields` on any network or parse failure |

---

### `dividends.py`

| Function / Class | Description |
|---|---|
| `DivCurve` | `NamedTuple('DivCurve', ['tenors_yr', 'div_yields', 'labels', 'source', 'snap_date_iso'])` with methods `.div_yield(T)` and `.as_table()`.  `labels` are tenor strings ("1M", "2M", …) for canonical-tenor curves and expiry strings for raw per-expiry extraction |
| `extract_implied(vol_df, spot, rate_curve, snap_date, max_T)` | Per-expiry parity extraction: returns one knot per option expiry where `q(T) = r(T) − ln(F(T)/S)/T`.  Result is bumpy in the front end (Bloomberg often quotes `F(T_earliest) ≡ S` to the cent, so `q = r` artificially; ex-dividend timing jolts F at any tenor) — use the canonical-tenor extractor below for display |
| `extract_implied_at_canonical_tenors(vol_df, spot, rate_curve, ...)` | Builds the displayed sidebar curve by fitting an OLS line `q(T) = a + b·T` (linearised: `q·T = a·T + b·T²`, naturally `T`-weighted) to all per-expiry points with `1/52 ≤ T ≤ 5y`, then evaluating at `TENOR_LABELS`.  Long-end expiries dominate the fit; the front-end `F ≡ S` artefact is filtered |
| `build_curve_from_table(edited_df, snap_date, source)` | Reconstructs a `DivCurve` from an edited sidebar table (columns `Tenor`, `Years`, `q (%)`).  Used when the user overrides a cell |
| `flat_div_curve(yield_value, tenors_yr, snap_date)` | Constant-yield curve at every tenor — fallback / test helper |
| `get_div_yield(curve_or_scalar, T)` | Duck-typed accessor — returns `curve.div_yield(T)` if it has the attribute, else `float(curve_or_scalar)`.  Mirrors `rates.get_rate` |
| `_interp_qT(T, tenors, yields)` | Internal: linear interpolation in `q·T` space (piecewise-flat instantaneous div), with flat extrapolation outside the curve |

**Why `q` doesn't enter Heston calibration**: under moneyness-quoted IVs, `K = m_pct · F`, so scaling `F` by `λ = exp(−Δq·T)` scales every `K` by the same `λ`.  Heston/Black-76 prices and vegas are positively homogeneous of degree 1 in `(F, K)`, so the vega-weighted residual is invariant — calibrated parameters are bit-identical for any `q`.  `q` only matters where strikes are absolute, i.e. the MC pricers.  Regression test: `tests/test_dividends.py::TestHestonCalibrationQInvariance::test_calibration_invariant_under_q`.

**Where the basis to Bloomberg's view comes from**: Bloomberg builds `F` from a SOFR / futures-implied rate, not Treasury CMT, so our extracted `q` is shifted *down* by `(r_BB − r_treasury)` (~10 bp at 1M, ~50 bp at 5Y).  This is a structural rate-basis effect — the curve still reproduces Bloomberg's `F` exactly under our `r`, which is what self-consistent option pricing needs.  No free, no-key SOFR-OIS curve source is currently available.

---

## 14. Test Suite

Tests live in `tests/` and are run with `uv run pytest tests/ -v`.  The full
suite contains **132 tests** across seven files.

| File | Tests | Coverage |
|---|---|---|
| `test_data_loader.py` | 8 | Schema validation, long-format output columns, metadata extraction, time-to-expiry computation |
| `test_surface_builder.py` | 4 | Surface grid shape, no negative vols, front-end spike preservation, 1-D interpolation |
| `test_arbitrage_checks.py` | 7 | Calendar / butterfly / vertical check logic on synthetic data |
| `test_heston.py` | 38 | Black-76 (scalar + vectorised, IV roundtrip), Heston CF properties (`f_j(0)=1`, modulus, stress), call pricing (positivity, spot cap, intrinsic floor, put-call parity, monotonicity in K, zero-vol-of-vol collapses to Black-76), smile batch, MC (Fourier reference within 3 SE, seed reproducibility, barrier ordering), and **9 one-touch tests** (bounded by payout, immediate payment past barrier, unreachable barrier, monotonicity in B, seed, payout linearity, validation) |
| `test_montecarlo.py` | 10 | One-touch under Dupire LV: bounded by payout, immediate payment, unreachable barrier, monotonicity in B, seed reproducibility, payout linearity, rate sensitivity for down-touch, validation |
| `test_dividends.py` | 29 | Parity-based implied-q extraction (recovery to machine precision under flat / non-flat rate curves, scalar-rate duck-typing, max-T filter, error on empty window), `q·T` linear interpolation (at-node, between-nodes, flat extrapolation, single-node and empty-curve edge cases), `DivCurve` API + table round-trip, `flat_div_curve` / `build_curve_from_table` builders, `get_div_yield` scalar/curve duck-typing, `F = S·exp((r−q)·T)` self-consistency from extracted q, sidebar-table round-trip regression, and a **Heston-calibration q-invariance** regression: prices/vegas are homogeneous of degree 1 in `(F, K)` under moneyness-quoted IVs, so calibrated params must be bit-identical for `q = 0` vs `q = 5%` (guards against future re-introduction of an inert F-rebuild) |
| `test_rates.py` | 36 | BEY → continuous conversion, recursive coupon-bond bootstrap (2y/3y/5y self-consistency under flat, normal, inverted, mildly-negative, and *strongly-negative* −8 % curves), log-DF interpolation (at-node, between-nodes, flat extrapolation, geometric-mean DF property), `RateCurve` API, builder helpers, scalar/curve duck-typing, full-tenor completeness, tuple-roundtrip for the Streamlit cache key, and **canonical-label round-trip** through `as_table` (regression guard so user edits to 1M/2M/4M rows aren't silently dropped) |

Tests that need market data load from `data_vol_surface/vol_surface_14_04_2026.xlsx`
(the older of the two bundled snapshots — kept stable across commits so the
test fixtures don't drift when newer files are added).  The Treasury
network fetch itself is *not* covered by unit tests — those would belong in
a separate integration suite.

---

## 15. Known Limitations & Data Notes

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

Call prices are computed from implied vols using Black-76, with the
**slice-specific zero rate `r(T)` read off the Treasury-bootstrapped curve**
for each expiry.  The default rate set is fetched live from treasury.gov
for the snapshot date and bootstrapped to continuous zeros; the user can
override any tenor in the sidebar table.  Results are sensitive to the
rate for deep ITM/OTM options or very long maturities — using a per-slice
curve rather than a flat rate noticeably tightens the long end.

**Implied dividend curve — SOFR-Treasury basis (residual gap to Bloomberg)**

`q(T)` is extracted by parity from the Bloomberg forward and the bootstrapped
Treasury rate curve (`q(T) = r(T) − ln(F(T)/S)/T`), then OLS-fitted to a
straight line in T and evaluated at canonical tenors — see [§7.1](#71-sidebar-controls).
Bloomberg builds `F` from a SOFR / futures-implied rate, not Treasury CMT,
so our extracted `q` is shifted *down* by `(r_BB − r_treasury)` (~10 bp at
1M, ~50 bp at 5Y).  The curve still reproduces Bloomberg's `F` exactly
under our `r`, which is what self-consistent option pricing needs — the
residual gap to Bloomberg's "indicative dividend yield" is a SOFR-Treasury
basis effect, not a fit bug.  No free, no-key SOFR-OIS curve source exists
to close this.

**Local vol at very short expiries**

The `_T_MIN_CUTOFF = 14 / 365` (~14 calendar days, ~2 weeks) filter removes
the very shortest expiries from the local vol computation.  These slices have too
few T-neighbours for PCHIP to estimate `∂w/∂T` reliably and are excluded
to keep the surface stable.  Path simulations with `T < _T_MIN_CUTOFF`
(and the early steps of any longer path) clamp the time axis at the
boundary, so sub-cutoff OTs / barriers price against the LV slice at the
cutoff rather than a true `t = 0` vol.

**Per-slice SVI: no cross-slice arbitrage enforcement**

Both the Vol Smiles overlay and the local-vol pipeline fit **per-slice SVI**
(Gatheral form) — the same fit, used twice.  Per-slice SVI is the fit family
Bloomberg OVME and most production desks use for LV — not for arbitrage
guarantees but because it preserves per-expiry idiosyncrasies (front-end
skew bumps, off-ATM curvature variation) that a global parametric surface
like SSVI would smooth away.  Note that "matches Bloomberg" here refers to
the *fit family*, not necessarily the resulting prices: Bloomberg's OVME
uses PDE solvers for path-dependent payoffs whereas this codebase uses
discretely-monitored MC, which under-detects hits on short-dated barriers
and one-touches by `O(σ·√Δt)`.

Trade-off: per-slice SVI does **not** enforce calendar-spread no-arbitrage
between adjacent slices.  Where neighbouring fits cross in `w(·, T)`, the
Dupire denominator `g` goes ≤ 0 and the affected (K, T) point is masked
to NaN in the LV surface.  In practice this is rare on real Bloomberg
quotes and the holes appear at the wing edges where pricing accuracy is
already poor — the MC pricer linearly fills them in T before sampling
to avoid query failures.

**Heston calibration is a compromise fit**

A single set of five Heston parameters is projected onto the entire filtered
surface.  For equity indices this typically gives an RMSE of ~5–10 pp — good
enough to price exotics consistently with the overall smile level and skew,
but not tight enough to recover the precise shape at every expiry.  For
better slice-level precision, consider a time-dependent extension (e.g.
piecewise-constant θ and ξ).

**Feller violation under calibration**

The default parameter bounds allow `2κθ < ξ²` because forcing Feller almost
always degrades the IV fit for equity surfaces (high observed skew requires
large ξ).  The full-truncation Euler scheme handles the violated regime
without bias, but implied densities under calibrated parameters can place
more mass near `v=0` than a Feller-constrained fit would.
