# ⚾ MLB Sustainability Model

**Is a hitter's hot start (or slump) real, or is it luck that will fade?**

A Statcast-driven projection and diagnostic dashboard that separates *true talent change* from *random variance* for every qualified MLB hitter — built entirely on public data (Statcast via `pybaseball` + the MLB Stats API).

Instead of collapsing everything into one arbitrary score, the model classifies each player on two orthogonal axes — **who he truly is** (talent tier) and **what's happening right now** (performance tier + luck split) — then layers on momentum, surface-stat flags, trajectory, and a full visual diagnosis of *why*.

---

## What it does

For any 2023–2026 hitter, the model answers:

- **How far above/below his projected level is he?** — e.g. *"+14% better than projected"*
- **How much of that is skill vs luck?** — a z-score test turns sample size into a real probability, shown as a **Skill vs Luck split bar**
- **Where will he realistically finish?** — a Bayesian rest-of-season projection
- **Why is his line what it is?** — zone heatmaps, a ballpark spray chart, and pitch-mix-faced trends

---

## The three pages

### 📊 Explorer (Leaderboards)
Browse and group the whole league by a **Talent Tier × Performance Tier** grid. Quick-view presets (🚀 Real Breakouts, 💎 Buy-Low, ❄️ Genuine Slumps, 🔋 Power Outages, …) plus custom filters (tier, performance, form, flags, min PA). Every row carries the archetype, form arrow, flags, projection, and luck %.

### 👤 Player Detail
A full scouting card for one hitter:
- **Archetype header** — talent tier, magnitude vs projected level, skill/luck split bar, form arrow (▲▬▼), surface-stat chips
- **Rolling xwOBA chart** vs projected baseline (with a draggable range slider)
- **Career hitting summary** with `OPS+ / wOBA+ / xwOBA+` (league-normalized, cross-season comparable)
- **Skill Profile** — season-pickable radar, percentile bars, and a peripheral-support spider
- **🔬 Visual Diagnosis** — Gaussian-smoothed zone heatmaps (attack / whiff / damage), a spray chart on the player's real ballpark outline, and pitch-mix-faced trends — all selectable by season (2023–2026)

### 🧠 Behind the Scenes
The complete methodology in 25 expandable sections — every formula, threshold, and design decision, with worked examples. (Also available as a generated Word document.)

---

## How the model works (in brief)

| Stage | What happens |
|---|---|
| **Aggregation** | Per-season Statcast stats, regular-season only; park-neutralized wOBA; competitive-swing bat speed; hit-type & pitch-type splits |
| **Marcel baseline** | PA- and recency-weighted history, regressed to a PA-weighted league mean, age-adjusted (stat-specific aging curves) |
| **Reliability** | Bayesian regression-to-mean at two stages — building the baseline, and projecting forward |
| **Z-score engine** | Turns deviation + sample size into a luck probability: `p_luck = 1 − erf(|z| / √2)` |
| **Peripheral support** | Soft-cap `2·tanh(z/2)` blend of K%, BB%, barrel%, and skill-adjusted luck |
| **Classification** | Talent tier (baseline percentile) × performance tier (bucketed `|z|`), plus momentum, trajectory tags, and surface flags |
| **Projection** | `current_reliability · current + (1 − current_reliability) · baseline` |

Full detail lives in the in-app **Behind the Scenes** page.

**Accuracy:** PA / AB / BA / OBP / SLG / K% / BB% match Baseball Savant to the third decimal. xwOBA/xBA/xSLG use Savant's published `estimated_*_using_speedangle`. wOBA matches FanGraphs to ~±.003 after park adjustment. Bat speed matches Savant within ~0.2 mph (competitive-swing filter).

---

## Architecture

Two layers, separated by parquet files:

```
data_pull_baseball.py   ──writes──►   *.parquet   ──read by──►   mlb_model_sustain.py
   (pipeline: all math)                (data)                     (Streamlit dashboard)
```

The **pipeline** runs locally to pull Statcast and compute everything. The **dashboard** is a pure read-only view layer — it never runs the math, just reads the parquets. This keeps the heavy work offline and makes the app fast and cheap to host.

### Files

| File | Role |
|---|---|
| `mlb_model_sustain.py` | The Streamlit dashboard (deploy this) |
| `data_pull_baseball.py` | The pipeline — pulls Statcast, computes everything, writes parquets |
| `generate_methodology_doc.py` | Builds the Word-doc version of the methodology |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Forces dark theme (the UI hardcodes dark backgrounds) |
| `comparison.parquet` | One row per 2026 hitter — baselines, deviations, scores, flags |
| `player_history.parquet` | All four seasons stacked |
| `rolling_xwoba_2026.parquet` | Trailing 50-BBE rolling xwOBA |
| `pitch_mix_by_season.parquet` | Pitch-mix faced, by year |
| `pitches_{2023..2026}.parquet` | Slim per-pitch data for heatmaps + spray |

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run mlb_model_sustain.py
```

The dashboard opens at `http://localhost:8501` and reads the committed parquets — no data pull needed to view it.

### Refreshing the data

The dashboard is static — it shows whatever's in the parquets. To pull fresh games:

1. In `data_pull_baseball.py`, update the Statcast pull date range and re-pull (overwrites `statcast_2026.parquet`).
2. Run the pipeline:
   ```bash
   python data_pull_baseball.py
   ```
   This regenerates **all** derived parquets in one pass.
3. Commit & push the changed parquets (git only pushes what actually changed):
   ```bash
   git add *.parquet
   git commit -m "refresh data through <date>"
   git push
   ```

> Only the 2026-dependent files change on a mid-season refresh (`comparison`, `player_history`, `rolling_xwoba_2026`, `pitch_mix_by_season`, `pitches_2026`). The completed-season pitch files (2023–2025) stay frozen.

---

## Deploy (Streamlit Community Cloud)

1. Push the repo to GitHub (the `.gitignore` excludes the huge raw Statcast pulls — the app doesn't need them).
2. At [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo/branch.
3. **Main file:** `mlb_model_sustain.py`
4. **Advanced → Python 3.12** (matches the pinned dependencies).
5. Deploy. Future pushes auto-redeploy.

---

## Tech stack

`streamlit` · `pandas` · `numpy` · `plotly` · `matplotlib` · `scipy` · `pybaseball` · `duckdb` · `requests`

---

## Known limitations

- **Mid-season trajectory signals** (momentum, form, trajectory tags) are the noisiest and firm up as the season's sample grows.
- **Rookies** have no baseline → tier "Unproven", reliability 0; their deviations should be ignored.
- **No L/R splits or injury adjustments** in the current model.
- **Sprint speed** isn't modeled directly — it's inferred through a persistent personal `xwOBA_diff` gap, so a rookie speed demon isn't flagged as an "xwOBA-beater" until a few seasons accrue.
- `bat_speed` exists only from 2023 onward in the Statcast pulls.

See the in-app **Behind the Scenes → Known limitations** for the full list.

---

*Built for separating signal from noise — and for making it watchable.*
