# ⚽ CT's Big-5 Scouting Dashboard

> A transparent, role-aware scouting tool for every outfield player in Europe's top 5 leagues. Built to answer questions football data dashboards usually don't: **"Is this player good?" *and* "Is this actually his role?"**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://scouting-app-cua-chuong.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.57-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

🔗 **Live demo:** https://scouting-app-cua-chuong.streamlit.app/


---

## What it is

A single-screen scouting tool covering every outfield player in the Big Five leagues — Premier League, La Liga, Bundesliga, Serie A, Ligue 1 — with at least 900 minutes played this season. Pick a player, pick a tactical role, see how well they fit. Or flip it: pick a role, see the top 20 candidates across Europe. Or paste a custom shortlist (e.g. a club's transfer targets) and see who fits which role at a glance.

The whole thing runs on transparent math — every grade you see can be traced back to specific percentile ranks against the player's positional peers, with documented weighting. No black-box ratings.

Built in Streamlit, fully bilingual (English + Tiếng Việt), deployed on Streamlit Community Cloud.

---

## What makes it different

Most public scouting dashboards either give you raw stats (and leave interpretation to you), or give you a black-box "overall rating" (and you have to trust it). This one tries the harder middle ground: **a transparent grade that separates a player's quality from their stylistic fit for a specific role.**

The headline grade for any player-role combination is a blend of two numbers — both shown alongside:

- **Quality** — how good the player is at the stats this role demands. An importance-weighted average of percentile ranks against position peers.
- **Style Fit** — does the *shape* of the player's profile actually match the role archetype? Level-independent. Mathematically a cosine similarity between the player's centered category profile and the role's emphasis profile.

Why both? Because Quality alone can't tell you if a great player is the *right kind* of great. **Michael Olise** scores 92 Quality for Trequartista AND 80 Quality for Box-to-Box — both look great on paper. But his Style Fit is 92 for Trequartista and only 32 for Box-to-Box. The blend collapses him off the B2B leaderboard while keeping him at the top of Trequartista — exactly where he belongs. A pure box-to-box engine with Q 70 / S 75 outranks him on B2B (blend 73 > 56) without inflating attacking grades anywhere.

That single design choice fixes the most common public-dashboard failure mode — elite all-rounders carpeting every role leaderboard regardless of whether the role is their style.

---

## Features

The app has five pages, each one a different question you might ask the data.

| Page | Question it answers |
|---|---|
| 🔍 **Scouting Report** | "Pick a player. Show me how they fit role X." |
| 🏆 **Role Rankings** | "Pick a role. Show me the top 20 across Europe." |
| 📊 **Stat Leaderboards** | "Who's elite at this one specific stat?" |
| ⚖️ **Player Comparison** | "Compare these two head-to-head." |
| 📋 **Shortlist Analyser** | "Paste my custom longlist. Who fits what?" |

Cross-cutting features available on every page:

- **Position group selector** — 🛡 Defender / ⚽ Midfielder / 🎯 Forward. Each group has its own stat catalog and role-preset library. Hybrid players (e.g. Valverde at `MF,DF`, Bellingham as attacking 8) get bucketed by FBref's primary-position label.
- **League pressing adjustment** (toggle, off by default) — neutralises the structural difference between high-pressing leagues (Bundesliga) and possession-heavy ones (La Liga) when ranking defensive volume stats.
- **Cohort filter** (Scouting Report) — re-rank against U21, U23, U25, 30+, same-league, or same-age-bracket peers instead of the full Big Five.
- **Custom role presets** — 22 hand-tuned tactical archetypes across the three position groups, mirroring Football Manager's role taxonomy (Anchor Man, Regista, Trequartista, Ball-Playing Defender, Wing-Back, Poacher, Inside Forward, etc.). Plus a "Custom" slider mode for anything that doesn't fit.
- **Similar profiles** — for any selected player, surface the 8 most stylistically similar names using a category-equalised Manhattan-distance match on percentile shapes.

---

## Methodology

### The data

Every outfield player in the Big Five leagues with at least 900 minutes played in 2025–26, merged from three public sources:

- **[FBref](https://fbref.com)** — playing time, position, basic and advanced stats (passing, defense, possession, GCA, misc)
- **[Sofascore](https://www.sofascore.com)** (via [`ScraperFC`](https://github.com/oseymour/ScraperFC)) — duel, dribble, pressing, error, big-chance data
- **[Understat](https://understat.com)** (via [`soccerdata`](https://github.com/probberechts/soccerdata)) — xG, xA, xG chain, xG buildup, shot quality

About 1,400 players in the final pool after deduplication and minutes filter.

### Position grouping

Each player is assigned to one of three groups based on FBref's `pos_` classification:

- **🛡 Defender** = `DF` + `DF,MF`
- **⚽ Midfielder** = `MF` + `MF,DF`
- **🎯 Forward** = `MF,FW` + `FW` + `FW,MF`

Hybrids (MF/FW, etc.) join the group their forward / midfield / defensive primary output makes most analytical sense in — MF/FW players (Bellingham-types with forward output) go to Forward, MF/DF players (Valverde covering at CB) stay with Midfielders, etc.

**Why this matters:** percentile ranks are computed *within* the position group. A pure midfielder's "90th-percentile tackles" is benchmarked against ~540 midfielders, not against centre-backs or wingers. Fair pool, contextual numbers.

### Percentile ranks

Every stat in the catalog is converted to a percentile (0–100) within the active position group:

```
percentile = rank(value, ascending=True) / pool_size × 100
```

For "inverted" stats where lower raw = better (e.g. `Dispossessed/100T`, `Possession Lost/100T`, `Dribbled Past/90`), the percentile is flipped so a high number always means "good."

### Role presets — the slider system

Each role is defined by a set of stat weights on a **0-to-5 scale**:

- `0–2.5`: **ignore** this stat (it doesn't count toward the grade — neither good nor bad performance affects the role fit)
- `5`: **essential** to the role

This is a critical design choice. An earlier version *inverted* low sliders (treating "I don't want X" as "being bad at X is good"), which inadvertently rewarded one-dimensional players. The current "ignore below 2.5" semantics keeps the math honest: stats the role doesn't ask for simply drop out of the calculation.

22 presets ship across the three position groups:

- **Midfielders (10):** Anchor Man, Ball-Winning Midfielder, Half-Back, Deep-Lying Playmaker, Regista, Central Midfielder, Box-to-Box, Mezzala, Advanced Playmaker, Trequartista
- **Defenders (6):** No-Nonsense Defender, Ball-Playing Defender, Centre-Back, Full-Back, Wing-Back, Inverted Wing-Back
- **Forwards (7):** Poacher, Target Forward, Pressing Forward, Deep-Lying Forward, Complete Forward, Inside Forward, Winger

Each preset has a written description that explains the archetype in plain English (and Vietnamese).

### Role-fit grading

For each player-role combination, the model produces three numbers:

1. **Quality (Q, 0–100)** — importance-weighted average of percentiles in the wanted stats, with categories equalised so the 11 defensive stats don't drown the 4 final-product stats.

```
For each category:
  importance = clip((slider − 2.5) / 2.5, 0, 1)
  cat_fit    = Σ(percentile × importance) / Σ(importance)
  cat_pull   = mean(importance)        # equalises across categories

grade = Σ(cat_fit × cat_pull) / Σ(cat_pull)
```

2. **Style Fit (S, 0–100)** — cosine similarity between the mean-centred player profile and the role's emphasis profile, mapped to 0–100. Independent of overall quality.

```
player_shape = cat_scores − mean(cat_scores)
role_shape   = cat_pull   − mean(cat_pull)
style_fit    = (cos(player_shape, role_shape) + 1) / 2 × 100
```

3. **Blend (the headline)** = `(Q + S) / 2`. The Football Manager ★ stars in the UI track the blend.

### Validation

The role presets were tuned and stress-tested against curated lists of canonical players — e.g. Anchor Man against Rodri, Casemiro, de Roon, Kamara, Højbjerg; Regista against Vitinha, Frenkie de Jong. The rule: a role's exemplars should cluster in the top quartile of that role's ranking, otherwise the weights are wrong. If Rodri doesn't grade as a top Anchor Man, the model has a problem.

All 22 shipping presets pass this test with median exemplar percentile ≥ 78th (most ≥ 86th). Box-to-Box jumped from 67th to 86th after a weighting revision during development.

---

## Setup

### Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

# Optional: virtualenv
python -m venv .venv
source .venv/bin/activate            # macOS/Linux
.venv\Scripts\activate               # Windows

pip install -r requirements.txt
streamlit run scouting_app.py
```

Open http://localhost:8501 in your browser.

### Data

The repo ships with `PERFECT_scouting_data_2026.csv` — a pre-built, merged dataset combining FBref, Sofascore (via ScraperFC), and Understat (via soccerdata). The CSV is what the app reads at startup.

The original scraping + merging scripts that built this CSV aren't included in this public repo (kept private for now). Anyone who wants to reproduce the dataset from scratch can use the public packages listed below — `soccerdata` covers FBref + Understat, `ScraperFC` covers Sofascore — and merge by player name with team-substring disambiguation for collisions.

The Streamlit app picks up CSV changes automatically: `load_scouts` is keyed on the file's modification time, so dropping in a refreshed CSV refreshes the app on next reload without a cache clear.

### File structure

```
.
├── scouting_app.py                 # Main Streamlit app (all 5 pages, all UI)
├── PERFECT_scouting_data_2026.csv  # Master dataset, pre-built
├── requirements.txt
└── README.md
```

---

## Tech stack

- **[Streamlit](https://streamlit.io)** — UI and deployment
- **[pandas](https://pandas.pydata.org)** + **[NumPy](https://numpy.org)** — data wrangling, stat derivations
- **[Plotly](https://plotly.com)** — radar charts, heatmaps, overlapped polar plots
- **[soccerdata](https://github.com/probberechts/soccerdata)** — FBref / Understat / WhoScored scraping
- **[ScraperFC](https://github.com/oseymour/ScraperFC)** — Sofascore scraping
- **[SciPy](https://scipy.org)** — Manhattan distance for similar-profile matching

---

## Limitations (honest list)

These exist because of data constraints, not because of laziness. If you're going to use the tool, you should know:

- **No team-possession context.** Defenders on bad teams have inflated raw defensive volume because they're defending more. The league-adjustment toggle helps neutralise this at league level, but not at team level — that would require team-possession data we don't have.
- **No positional / tracking data.** The model knows what a player did, not where on the pitch they did it. So a winger labelled "MF" by FBref (e.g. Olise, Salah, Cherki) still appears in the midfielder pool. Style Fit can partially compensate by surfacing their winger-shaped profile — but only partially.
- **Rate-stat noise at the 900-minute floor.** Players just above the threshold have noisier percentage stats (Pass%, Aerial%) than 2500-minute regulars. Most users won't hit this, but it's a real caveat for low-minutes players.
- **FBref's position labels can disagree with the eye test.** Several obvious wingers (Olise, Cherki, Foden, Greenwood) are labelled `MF` by FBref because of how they classify wide-mid-in-4-3-3 minutes. They'll show up in the Midfielder pool, not the Forward pool — Style Fit will still surface their winger-shaped profile inside that pool, but the labelling itself stays whatever FBref says.
- **Subjective preset weights.** The 22 role presets are tuned based on football judgement, then validated against canonical exemplars. Reasonable people will disagree on whether McTominay is a Mezzala or a Segundo Volante. The Custom Manual mode lets you build any role from scratch with sliders.

---

## Roadmap

Things I'd like to add eventually:

- Player images + team logos on every card (the data has the IDs already, just needs UI work)
- Shortlist export to CSV / PDF
- Persistent user shortlists across sessions
- Time-series tracking ("Rodri's role-fit grade over the season")
- A standalone goalkeeper page with its own model (PSxG-allowed, save %, claims, distribution)
- More granular position splits (CB vs FB, IF vs CF) once role libraries justify it

Issues and PRs welcome. If you find a player the model gets weirdly wrong, that's interesting — open an issue with the player name and the role you expected.

---

## Acknowledgments & data credits

This entire project rides on public data made available by:

- **[FBref](https://fbref.com)** — the spine of the dataset (positions, minutes, advanced stats)
- **[Sofascore](https://www.sofascore.com)** — duels, dribbles, pressing, errors
- **[Understat](https://understat.com)** — xG, xA, and shot quality
- **[soccerdata](https://github.com/probberechts/soccerdata)** by Pieter Robberechts — clean wrapper for FBref/Understat scraping
- **[ScraperFC](https://github.com/oseymour/ScraperFC)** by Owen Seymour — Sofascore scraping

The Football Manager role taxonomy (Anchor Man, Regista, Inverted Wing-Back, etc.) is from Sports Interactive's *Football Manager*. The model uses the same role names because they're the most widely understood tactical vocabulary among football fans and scouts; the weights are entirely my own, tuned and validated against statistical exemplars.

---

## License

MIT — do whatever you want with it, just credit where it came from.

---

## Get in touch

- **Twitter / X: @CTdoesanalytics
- **YouTube: www.youtube.com/@CTlovessports

If you build something on top of this, or if you find a bug, or if you just want to nerd out about football analytics — I'd love to hear about it.
