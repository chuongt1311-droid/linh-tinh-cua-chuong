"""
═══════════════════════════════════════════════════════════════════════════════
MLB SUSTAINABILITY MODEL — THE DASHBOARD (Streamlit front-end)
═══════════════════════════════════════════════════════════════════════════════

WHAT THIS FILE IS
─────────────────
This is the VIEW layer. It does NO modeling — all the math happens in
data_pull_baseball.py, which writes two parquet files. This file just reads
those files and draws them:
    comparison.parquet      → one row per 2026 hitter (baseline, deviation, score…)
    player_history.parquet  → every player's per-season stats 2023-2026

Run it with:  streamlit run mlb_model_sustain.py

ARCHITECTURE (pipeline → data → dashboard)
──────────────────────────────────────────
    data_pull_baseball.py  ──writes──>  *.parquet  ──read by──>  this file
We keep them separate so the heavy pybaseball/Marcel math runs once, and the
dashboard is just a fast file-reader. Regenerating the parquet auto-refreshes
the dashboard (see the mtime-based caching below).

THE TWO PAGES
─────────────
  📊 Leaderboards   → hot/cold/rookie tables ranked by sustain score
  👤 Player Detail  → deep dive on one player:
        - ESPN-style header (photo, vitals, team logo)
        - Sustainability verdict card (the star of the page) with projections
        - Career hitting summary table (per season + totals, with team logos)
        - Skill Profile tabs:
            • Year-by-Year radar      (percentile vs that season's league)
            • Current vs Baseline radar
            • Peripheral Support spider (do the skills back the trend?)
            • Hit-Type Mix            (GB/LD/FB/PU by season + league ref)
            • vs Pitch Type           (xwOBA vs FB/Breaking/Offspeed + league ref)
        - Stat trend over time (any stat, 2023-2026, vs league avg)
        - Glossary (definitions + benchmarks + computed league references)

KEY DASHBOARD CONCEPTS YOU'LL SEE BELOW
───────────────────────────────────────
  1. mtime-based caching — every loader takes the parquet's modification time
     as a (hidden) argument. When the file changes, the cache key changes, so
     the dashboard reloads fresh data automatically. No manual "clear cache".

  2. The QUALIFIED pool — "vs league" comparisons (percentiles, league averages)
     are computed against QUALIFIED hitters only (300+ PA in completed seasons;
     everyone in 2026 since it's mid-season). Partial-season players still get
     a data point, but they're ranked against full-timers. Every percentile /
     radar / league-line filters to this pool, with a fallback to the full df
     if the qualified subset would be empty.

  3. league_refs — actual league-average hit-type rates and pitch-type xwOBA,
     computed (PA/BBE-weighted) from YOUR 2024-25 data, not hardcoded. Used for
     the reference overlays and the glossary so the numbers always match the data.

  4. Custom HTML cards — the header and sustain card are hand-built HTML strings
     rendered via st.markdown(..., unsafe_allow_html=True). IMPORTANT: these must
     be single continuous strings with NO blank lines — Streamlit's markdown
     parser treats a blank line as the end of an HTML block and dumps the rest
     as escaped text. (This bit us once; that's why the card HTML is one big
     concatenation.)
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from pathlib import Path
import duckdb
from pybaseball import statcast, spraychart

# Gaussian smoothing for the zone heatmaps. scipy is optional; if it's missing
# we no-op (plotly's zsmooth='best' still does bilinear interpolation, which
# isn't as clean but doesn't crash).
try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except ImportError:
    def _gaussian_filter(arr, sigma):
        return arr

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MLB Sustainability", page_icon="⚾", layout="wide")

HERE = Path(__file__).parent
COMPARISON_PATH = HERE / 'comparison.parquet'
HISTORY_PATH    = HERE / 'player_history.parquet'
ROLLING_PATH    = HERE / 'rolling_xwoba_2026.parquet'
PITCHMIX_PATH   = HERE / 'pitch_mix_by_season.parquet'

def pitches_path(year):
    """Per-year slim pitch file — loaded lazily so we never hold all 4 years."""
    return HERE / f'pitches_{year}.parquet'

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
#
# We pass the file's modification time as the cache key. When the pipeline
# regenerates a parquet, mtime changes → cache key changes → cache miss → fresh
# load. No need to manually clear the cache anymore.
#
# CRITICAL: the `mtime` argument MUST NOT start with an underscore. In
# @st.cache_data, any argument prefixed with `_` is DELIBERATELY EXCLUDED from
# the cache key (Streamlit's mechanism for passing unhashable objects like DB
# connections). If we named it `_mtime`, the file's modification time would be
# ignored, the cache key would never change, and the loader would serve ONE
# result forever — even after you push fresh parquets. Named `mtime` (a plain
# hashable float), it becomes part of the key: when the file changes, mtime
# changes, the key changes, and the cache reloads the new data.
# ═══════════════════════════════════════════════════════════════════════════════
def _file_mtime(path):
    """Returns the file's last-modified time, or None if missing."""
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return None

@st.cache_data
def load_comparison(mtime):
    return pd.read_parquet(COMPARISON_PATH)

@st.cache_data
def load_history(mtime):
    return pd.read_parquet(HISTORY_PATH)

@st.cache_data
def load_rolling(mtime):
    try:
        return pd.read_parquet(ROLLING_PATH)
    except (FileNotFoundError, OSError):
        return pd.DataFrame(columns=['batter', 'game_date', 'rolling_xwOBA'])

@st.cache_data
def load_pitches(year, mtime):
    """Slim pitch-level data for ONE season (heatmaps + spray). Cached per
    (year, mtime), so switching the season dropdown loads that year once, and a
    fresh parquet (new mtime) busts the cache."""
    try:
        return pd.read_parquet(pitches_path(year))
    except (FileNotFoundError, OSError):
        return pd.DataFrame()

@st.cache_data
def load_pitchmix(mtime):
    try:
        return pd.read_parquet(PITCHMIX_PATH)
    except (FileNotFoundError, OSError):
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# LEAGUE REFERENCE VALUES — computed from YOUR dataset, not hardcoded.
# Reference pool = qualified (300+ PA) hitters in completed recent seasons
# (2024 + 2025). We pick those because 2026 is mid-season noise and 2023 is
# starting to drift from the current run environment. Two completed years
# is the right sample for "this is what the league does."
#
# Aggregations are PROPERLY WEIGHTED:
#   - Hit-type rates: BBE-weighted (so a player with 400 BBE counts 4x more
#     than one with 100 BBE in the league average)
#   - Pitch-type xwOBA: PA-weighted (same logic; full-time guys carry more)
# Otherwise a weak hitter with 80 PA against breaking pitches would skew the
# average as much as Trout with 300.
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def compute_league_refs(mtime):
    h = pd.read_parquet(HISTORY_PATH)
    pool = h[(h['season'].isin([2024, 2025])) & (h.get('qualified', 1) == 1)]
    refs = {'hit_type': {}, 'pitch_xwoba': {}}

    bbe_total = pool['is_batted_ball'].sum()
    for ht in ['GB', 'LD', 'FB', 'PU']:
        col = f'{ht}_rate'
        if col in pool.columns and bbe_total > 0:
            refs['hit_type'][ht] = (pool[col] * pool['is_batted_ball']).sum() / bbe_total

    for cat in ['Fastball', 'Breaking', 'Offspeed']:
        xcol, pcol = f'xwOBA_vs_{cat}', f'PAs_vs_{cat}'
        if xcol in pool.columns and pcol in pool.columns:
            valid = pool[[xcol, pcol]].dropna()
            total_pas = valid[pcol].sum()
            if total_pas > 0:
                refs['pitch_xwoba'][cat] = (valid[xcol] * valid[pcol]).sum() / total_pas
    return refs

@st.cache_data(ttl=86400)
def get_player_info(mlbam_id):
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{int(mlbam_id)}", timeout=5)
        p = r.json()['people'][0]
        # Nested objects can be present-but-None (e.g. a free agent has
        # currentTeam = null). `p.get('currentTeam', {})` would return None in
        # that case — NOT the {} default — so a following .get() would raise and
        # wipe the ENTIRE info dict. `(p.get(k) or {})` is null-safe: it yields
        # {} whether the key is missing OR explicitly None.
        team     = p.get('currentTeam') or {}
        position = p.get('primaryPosition') or {}
        bats     = p.get('batSide') or {}
        throws   = p.get('pitchHand') or {}
        return {
            'fullName':  p.get('fullName', '—'),
            'height':    p.get('height', '—'),
            'weight':    p.get('weight', '—'),
            'age':       p.get('currentAge', '—'),
            'birthDate': p.get('birthDate', '—'),
            'birthCity': p.get('birthCity', ''),
            'birthCountry': p.get('birthCountry', ''),
            'team':      team.get('name', '—'),
            'position':  position.get('abbreviation', '—'),
            'bats':      bats.get('code', '—'),
            'throws':    throws.get('code', '—'),
            'number':    p.get('primaryNumber', ''),
        }
    except Exception:
        return None

def headshot_url(mlbam_id):
    return (
        f"https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/w_426,q_auto:best/"
        f"v1/people/{int(mlbam_id)}/headshot/67/current"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM CODE → (full name, MLB team ID for logos)
# Logos via https://www.mlbstatic.com/team-logos/{id}.svg
# ═══════════════════════════════════════════════════════════════════════════════
TEAM_INFO = {
    'AZ':  ('Arizona Diamondbacks',  109),
    'ATL': ('Atlanta Braves',        144),
    'BAL': ('Baltimore Orioles',     110),
    'BOS': ('Boston Red Sox',        111),
    'CHC': ('Chicago Cubs',          112),
    'CWS': ('Chicago White Sox',     145),
    'CIN': ('Cincinnati Reds',       113),
    'CLE': ('Cleveland Guardians',   114),
    'COL': ('Colorado Rockies',      115),
    'DET': ('Detroit Tigers',        116),
    'HOU': ('Houston Astros',        117),
    'KC':  ('Kansas City Royals',    118),
    'LAA': ('Los Angeles Angels',    108),
    'LAD': ('Los Angeles Dodgers',   119),
    'MIA': ('Miami Marlins',         146),
    'MIL': ('Milwaukee Brewers',     158),
    'MIN': ('Minnesota Twins',       142),
    'NYM': ('New York Mets',         121),
    'NYY': ('New York Yankees',      147),
    'OAK': ('Oakland Athletics',     133),
    'ATH': ('Athletics',             133),   # 2025+ relocation
    'PHI': ('Philadelphia Phillies', 143),
    'PIT': ('Pittsburgh Pirates',    134),
    'SD':  ('San Diego Padres',      135),
    'SEA': ('Seattle Mariners',      136),
    'SF':  ('San Francisco Giants',  137),
    'STL': ('St. Louis Cardinals',   138),
    'TB':  ('Tampa Bay Rays',        139),
    'TEX': ('Texas Rangers',         140),
    'TOR': ('Toronto Blue Jays',     141),
    'WSH': ('Washington Nationals',  120),
}

def team_logo_url(code):
    if pd.isna(code) or code not in TEAM_INFO:
        return None
    return f"https://www.mlbstatic.com/team-logos/{TEAM_INFO[code][1]}.svg"

def team_full_name(code):
    return TEAM_INFO.get(code, (code, None))[0] if not pd.isna(code) else '—'

# ═══════════════════════════════════════════════════════════════════════════════
# STAT DEFINITIONS — single source of truth for tooltips + glossary
# ═══════════════════════════════════════════════════════════════════════════════
STAT_DEFS = {
    'xwOBA':       dict(name='xwOBA',     defn='Expected weighted on-base average. Predicted from exit velo + launch angle. Best single "true talent" stat.', bench='MVP: .400+  ·  Elite: .370  ·  Avg: .315  ·  Poor: <.290'),
    'xBA':         dict(name='xBA',       defn='Expected batting average from contact quality. What the batter "deserved" to hit.', bench='Elite: .290+  ·  Avg: .248  ·  Poor: <.230'),
    'xSLG':        dict(name='xSLG',      defn='Expected slugging % from contact quality. Power version of xBA.', bench='MVP: .550+  ·  Elite: .480  ·  Avg: .405  ·  Poor: <.360'),
    'wOBA':        dict(name='wOBA',      defn='Weighted on-base average. Park-neutralized. Single number for total offense.', bench='MVP: .400+  ·  Elite: .370  ·  Avg: .315  ·  Poor: <.290'),
    'avg_exit_velo':dict(name='Exit Velo',defn='Average speed of the ball off the bat (mph). Raw measure of hard contact.', bench='Elite: 93+ mph  ·  Avg: 89  ·  Poor: <85'),
    'barrel_rate': dict(name='Barrel%',   defn='% of batted balls in the optimal exit-velo + launch-angle combo. Gold standard for damaging contact.', bench='MVP: 18%+  ·  Elite: 14%  ·  Avg: 9%  ·  Poor: <5%'),
    'hard_hit_rate':dict(name='HardHit%', defn='% of batted balls hit at 95+ mph.', bench='Elite: 50%+  ·  Avg: 40%  ·  Poor: <30%'),
    'avg_bat_speed':dict(name='Bat Speed',defn='Avg bat speed through the zone (mph). Only tracked from 2024.', bench='Elite: 76+ mph  ·  Avg: 71  ·  Poor: <68'),
    'K_rate':      dict(name='K%',        defn='Strikeout rate. Lower is better. Stabilizes fast (~60 PA).', bench='Elite (low): <15%  ·  Avg: 22%  ·  Poor: 28%+'),
    'BB_rate':     dict(name='BB%',       defn='Walk rate. Higher is better. Stabilizes ~120 PA.', bench='Elite: 14%+  ·  Avg: 8.5%  ·  Poor: <5%'),
    'BABIP':       dict(name='BABIP',     defn='Batting Avg on Balls in Play. Mix of contact quality + luck. Persistent deviation from .300 = skill.', bench='Elite: .350+  ·  Avg: .300  ·  Poor: <.260'),
    'HR_rate':     dict(name='HR Rate',   defn='Home runs per plate appearance. Direct power metric.', bench='MVP: 7%+  ·  Elite: 5%  ·  Avg: 3.3%  ·  Poor: <2%'),
    'chase_rate':  dict(name='Chase%',    defn='% of pitches outside the zone that the batter swings at. Lower is better.', bench='Elite (low): <22%  ·  Avg: 28%  ·  Poor: 35%+'),
    'whiff_rate':  dict(name='Whiff%',    defn='% of swings that miss. Lower is better.', bench='Elite (low): <20%  ·  Avg: 25%  ·  Poor: 32%+'),
    # Model-side stats (no good/bad benchmarks, just "what they are")
    'baseline_xwOBA': dict(name='Baseline xwOBA', defn='Marcel-style projection of the player from their 2023-25 history, PA-weighted, regressed to league mean, age-adjusted to 2026.', bench=''),
    'xwOBA_deviation':dict(name='Deviation',      defn='Current 2026 xwOBA minus baseline. Positive = hotter than expected, negative = colder.', bench=''),
    'projected_ros_xwOBA': dict(name='Projected RoS xwOBA', defn='Rest-of-season projection. Bayesian blend of 2026 current and aged baseline, weighted by 2026 PA.', bench=''),
    'signal_xwOBA':   dict(name='Signal',          defn='How much we trust the picture. Baseline trust × current sample trust. >0.20 = trust the deviation; <0.10 = sample too small.', bench=''),
}

def tooltip_for(stat_key):
    d = STAT_DEFS.get(stat_key)
    if not d: return ''
    txt = d['defn']
    if d.get('bench'):
        txt += ' — ' + d['bench']
    return txt

# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT TIER RELABELING — descriptive sentences
# (UNCERTAIN renamed to make it actually mean something)
# ═══════════════════════════════════════════════════════════════════════════════
VERDICT_RENAMES = {
    'REAL DEAL':       'Hot streak is fully sustainable',
    'TRENDING REAL':   'Hot streak is mostly sustainable',
    'UNCERTAIN':       'Mixed signals — could go either way',
    'LIKELY FLUKE':    'Hot streak likely a fluke',
    'MIRAGE':          'Hot streak is mostly luck',
    'GENUINELY COLD':  'Slump is fully real',
    'SLUMP REAL':      'Slump is mostly real',
    'UNLUCKY COLD':    'Slump is mostly bad luck',
    'BUY LOW':         'Slump is pure bad luck — buy low',
    'STABLE':          'Performing at expected level',
    'NO BASELINE':     'Rookie — no baseline yet',
}
def relabel_verdict(v): return VERDICT_RENAMES.get(v, v)

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE SUSTAIN THEME
# Returns (color, emoji, vibe-word) based on the COMBINATION of trend + score.
# Hot+high = fire, hot+low = sus/illusion, cold+high = freezing, cold+low = buy low.
# ═══════════════════════════════════════════════════════════════════════════════
def get_sustain_theme(score, trend, category):
    if category == 'ROOKIE':
        return ('#6b7280', '🌱', 'Rookie — no baseline yet')
    if trend == 'STABLE':
        return ('#94a3b8', '😐', 'Performing at expected level')

    if trend == 'HOT':
        if score >= 75: return ('#dc2626', '🔥🚀', 'Hot streak is fully sustainable')   # FUEGO red
        if score >= 60: return ('#ea580c', '🔥📈', 'Hot streak is mostly sustainable')  # warm orange
        if score >= 40: return ('#eab308', '🤔',  'Mixed signals — could go either way') # yellow caution
        if score >= 25: return ('#a855f7', '👀',  'Hot but sus — peripherals don\'t back it') # purple suspicion
        return            ('#7c3aed', '🎭',  'Hot streak is mostly luck — mirage')      # deep purple illusion

    # COLD
    if score >= 75: return ('#1e40af', '🥶❄️', 'Slump is fully real — ice cold')        # deep blue freezing
    if score >= 60: return ('#2563eb', '📉😞', 'Slump is mostly real')                  # blue + sad
    if score >= 40: return ('#0ea5e9', '🤷',   'Mixed signals — could go either way')   # sky blue
    if score >= 25: return ('#14b8a6', '🍀',   'Slump is mostly bad luck')              # teal
    return            ('#10b981', '💎',   'Pure bad luck — buy low')                    # emerald opportunity

# ═══════════════════════════════════════════════════════════════════════════════
# PERCENTILE / COLOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def percentile_color(pct):
    if pct >= 90: return '#a50026'
    if pct >= 75: return '#d73027'
    if pct >= 60: return '#f46d43'
    if pct >= 40: return '#999999'
    if pct >= 25: return '#74add1'
    if pct >= 10: return '#4575b4'
    return '#313695'

def percentile_rank(series, value, invert=False):
    series = series.dropna()
    if pd.isna(value) or len(series) == 0:
        return 50
    pct = (series < value).mean() * 100
    return 100 - pct if invert else pct

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECTED TRADITIONAL STATS
# Marcel only built baselines for skill stats (xwOBA, xBA, K%, BB%, HR%, barrel%).
# For BA/OBP/SLG/OPS, we use the PLAYER'S OWN career average as the prior
# (PA-weighted across 2023-25), then Bayesian-blend with 2026 current using
# the same regression-PA logic. Rookies fall back to current 2026 number.
# ═══════════════════════════════════════════════════════════════════════════════
def project_basic_stats(player, history_df):
    """Returns dict of projected rest-of-season values for BA/OBP/SLG/OPS/HR_rate."""
    hist = history_df[
        (history_df['batter'] == player['batter']) &
        (history_df['season'] < 2026)
    ]
    pa = player['PA']
    REG = {'BA': 600, 'OBP': 400, 'SLG': 600, 'HR_rate': 170}
    out = {}
    for stat, reg_pa in REG.items():
        cur = player.get(stat)
        if pd.isna(cur):
            out[stat] = np.nan
            continue
        if len(hist) == 0:
            # Rookie / no history — just use current value as projection
            out[stat] = cur
            continue
        prior = (hist[stat] * hist['PA']).sum() / hist['PA'].sum()
        cur_rel = pa / (pa + reg_pa)
        out[stat] = cur_rel * cur + (1 - cur_rel) * prior
    out['OPS'] = out.get('OBP', np.nan) + out.get('SLG', np.nan) if not pd.isna(out.get('OBP')) and not pd.isna(out.get('SLG')) else np.nan
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER ROW: photo · vitals · SUSTAIN CARD
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING xwOBA TREND CHART (compact, sits under the player portrait)
# Shows the within-season trajectory: the player's trailing 50-BBE xwOBA vs his
# projected baseline (dashed line). Line goes green if currently above baseline,
# red if below — instant "is he trending above or below where he should be."
# ═══════════════════════════════════════════════════════════════════════════════
def rolling_xwoba_chart(player_id, baseline_xwoba, rolling_df):
    d = rolling_df[rolling_df['batter'] == player_id].sort_values('game_date')
    if len(d) < 3:
        return None
    last = d['rolling_xwOBA'].iloc[-1]
    line_color = '#22c55e' if (pd.notna(baseline_xwoba) and last >= baseline_xwoba) else '#ef4444'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d['game_date'], y=d['rolling_xwOBA'], mode='lines',
        line=dict(color=line_color, width=2.5),
        hovertemplate='%{x|%b %d}: %{y:.3f}<extra></extra>',
    ))
    if pd.notna(baseline_xwoba):
        fig.add_hline(
            y=baseline_xwoba, line=dict(color='#888', dash='dash', width=1.5),
            annotation_text=f'projected baseline ({baseline_xwoba:.3f})',
            annotation_position='top left', annotation_font=dict(color='#888', size=11),
        )
    fig.update_layout(
        height=300, margin=dict(t=34, b=20, l=44, r=16),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        # rangeslider = the draggable "slide" mini-strip under the chart. It shows
        # the FULL series; drag its handles to zoom the main panel into any window.
        xaxis=dict(
            showgrid=False, color='#bbb', tickfont=dict(size=11), nticks=8,
            rangeslider=dict(visible=True, thickness=0.12, bgcolor='#161b22'),
        ),
        yaxis=dict(showgrid=True, gridcolor='#222', color='#bbb', tickfont=dict(size=11), fixedrange=True),
        showlegend=False,
        title=dict(text='Rolling xwOBA (every 50 batted balls) vs projected baseline',
                   font=dict(size=13, color='#ccc'), x=0.0),
    )
    return fig


def render_header_row(player, info, history_df, rolling_df=None):
    # Top-level split: LEFT (photo + vitals + rolling chart) | RIGHT (sustain card).
    # Nesting photo+vitals inside the left column lets the rolling chart sit
    # directly UNDER them — visible without scrolling — while spanning the full
    # left width and stopping before the card on the right.
    col_left, col_sustain = st.columns([3.2, 2.6])

    with col_left:
        col_photo, col_vitals = st.columns([1.2, 2])

        with col_photo:
            st.image(headshot_url(player['batter']), use_container_width=True)

        with col_vitals:
            full_name = info['fullName'] if info else player['name'].title()
            parts = full_name.upper().split(' ', 1)
            first = parts[0]
            last  = parts[1] if len(parts) > 1 else ''
            st.markdown(f"""
                <div style="line-height:1.1; margin-bottom:10px;">
                    <div style="font-size:22px; color:#888; letter-spacing:1px;">{first}</div>
                    <div style="font-size:36px; font-weight:900; letter-spacing:1px;">{last}</div>
                </div>
            """, unsafe_allow_html=True)

            if info:
                current_team_code = player.get('team')
                logo = team_logo_url(current_team_code)
                logo_html = (
                    f'<img src="{logo}" style="height:28px; vertical-align:middle; margin-right:8px;">'
                    if logo else ''
                )
                # Prefer the MLB API team name; if it's missing (free-agent/null
                # currentTeam, e.g. Moniak), fall back to the parquet-derived team.
                team_name = info['team']
                if team_name in ('—', '', None):
                    team_name = team_full_name(current_team_code)
                team_line = f'{logo_html}<b>{team_name}</b>'
                if info['number']:
                    team_line += f" &nbsp;·&nbsp; #{info['number']}"
                team_line += f" &nbsp;·&nbsp; {info['position']}"
                st.markdown(team_line, unsafe_allow_html=True)

                vital_rows = [
                    ('HT/WT',      f"{info['height']}, {info['weight']} lbs"),
                    ('BIRTHDATE',  f"{info['birthDate']} ({info['age']})"),
                    ('BAT/THR',    f"{info['bats']}/{info['throws']}"),
                    ('BIRTHPLACE', f"{info['birthCity']}, {info['birthCountry']}"),
                ]
                rows_html = "".join(
                    f"""<tr>
                        <td style="color:#888; font-size:11px; letter-spacing:1px; padding:3px 18px 3px 0; vertical-align:top;">{lbl}</td>
                        <td style="font-size:14px; padding:3px 0;">{val}</td>
                    </tr>"""
                    for lbl, val in vital_rows
                )
                st.markdown(f"<table style='margin-top:8px;'>{rows_html}</table>", unsafe_allow_html=True)

        # ── Rolling xwOBA chart: directly under photo+vitals, full left width ──
        if rolling_df is not None:
            rc = rolling_xwoba_chart(player['batter'], player.get('baseline_xwOBA'), rolling_df)
            if rc is not None:
                # Interactive: drag the rangeslider strip to zoom any window,
                # scroll to zoom, drag to pan, double-click to reset.
                st.plotly_chart(rc, use_container_width=True,
                                config={'displayModeBar': True, 'scrollZoom': True,
                                        'displaylogo': False})
                st.caption("Drag the **slider strip** below the chart (or scroll to zoom, drag to pan, "
                           "double-click to reset) to inspect any stretch of the season.")

    with col_sustain:
        render_sustain_with_model(player, history_df)

# ═══════════════════════════════════════════════════════════════════════════════
# SUSTAIN CARD — score + verdict + projected basic stats + model details
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE LOOKUPS — Tier × State headline (replaces the single sustain score)
# ═══════════════════════════════════════════════════════════════════════════════
# TALENT TIER: who the player truly is (from baseline_xwOBA percentile).
#   (color, emoji, display label)
TIER_INFO = {
    'Elite':         ('#fbbf24', '🌟', 'ELITE HITTER'),
    'Above-Average': ('#22c55e', '✅', 'ABOVE-AVERAGE HITTER'),
    'Average':       ('#9ca3af', '➖', 'AVERAGE HITTER'),
    'Below-Average': ('#60a5fa', '🔻', 'BELOW-AVERAGE HITTER'),
    'Unproven':      ('#94a3b8', '🆕', 'UNPROVEN (ROOKIE)'),
}

# CURRENT STATE: what's happening now vs that talent, and whether it's real.
#   (color, plain-English one-liner)
STATE_INFO = {
    'Surging':       ('#22c55e', 'Above his level and the underlying skills back it — real'),
    'Heating Up':    ('#84cc16', 'Above his level, skills are mixed — not yet confirmed'),
    'Hot but Lucky': ('#f59e0b', "Above his level but skills don't support it — regression likely"),
    'True to Level': ('#9ca3af', 'Performing right at his established level'),
    'Cooling Off':   ('#60a5fa', 'Below his level, skills are mixed — not yet confirmed'),
    'Unlucky':       ('#06b6d4', "Below his level but skills don't support the slump — buy-low"),
    'Declining':     ('#ef4444', 'Below his level and the skills confirm it — real'),
}

# PERFORMANCE TIER → plain-language label + color. This is the z-score magnitude
# in words a casual instantly gets (replaces the confusing Surging/Heating Up
# vocabulary on the card). Direction + degree, nothing else.
#   (display label, color)
PERF_DISPLAY = {
    'As Expected':       ('Performing at Projected Level', '#9ca3af'),
    'Notably Over':      ('Better Than Expected',          '#84cc16'),
    'Notably Under':     ('Worse Than Expected',           '#f59e0b'),
    'Unusually High':    ('Much Better Than Expected',     '#22c55e'),
    'Unusually Low':     ('Much Worse Than Expected',      '#ef4444'),
    'Unfathomably High': ('Far Better Than Expected',      '#16a34a'),
    'Unfathomably Low':  ('Far Worse Than Expected',       '#b91c1c'),
}

# MOMENTUM: within-season trajectory (rolling recent xwOBA vs season).
#   (chip text, background, text color)
MOMENTUM_CHIP = {
    'heating': ('📈 Heating Up',   '#064e3b', '#6ee7b7'),
    'cooling': ('📉 Cooling Down', '#7c2d12', '#fdba74'),
    'steady':  ('➡️ Steady',       '#374151', '#cbd5e1'),
}

# TRAJECTORY TAGS: forward-looking edge cases (State × Form × Tier).
#   (emoji, color, one-line meaning)
TRAJECTORY_INFO = {
    'Bottoming Out':       ('📈', '#84cc16', 'Was slumping, recent form turning up — may have hit bottom'),
    'Improving but Capped':('⬆️', '#9ca3af', 'Getting better, but the ceiling is low — still a below-average bat'),
    'Breakout Building':   ('🌱', '#22c55e', 'At his level but recent form is rising — a breakout may be forming'),
    'Fading':              ('⚠️', '#f59e0b', 'Looks fine now, but recent form is slipping — early warning'),
    'Bubble Bursting':     ('💥', '#ef4444', 'Riding hot luck AND cooling — the correction has already started'),
    'Free Fall':           ('❄️', '#3b82f6', 'Cold and getting colder'),
}

# ═══════════════════════════════════════════════════════════════════════════════
# SURFACE-STAT TAG CHIPS
# Renders the boolean flags from the pipeline as colored chips. This is the
# "tell the full story" layer — catches obvious surface-stat events (power
# outage, K spike, luck) that the xwOBA-based sustain verdict alone misses.
# A player can show several chips at once.
# ═══════════════════════════════════════════════════════════════════════════════
def build_flag_chips(player):
    # (flag column, emoji, label, background, text color)
    specs = [
        ('flag_power_outage', '🔋', 'Power Outage',  '#7f1d1d', '#fca5a5'),
        ('flag_power_surge',  '💥', 'Power Surge',   '#064e3b', '#6ee7b7'),
        ('flag_k_spike',      '⚡', 'K% Spiking',    '#7c2d12', '#fdba74'),
        ('flag_k_improve',    '✅', 'K% Improving',  '#064e3b', '#6ee7b7'),
        ('flag_buy_low',      '💎', 'Buy-Low (unlucky)',  '#1e3a8a', '#93c5fd'),
        ('flag_sell_high',    '📉', 'Sell-High (lucky)',  '#78350f', '#fcd34d'),
    ]
    chips = []
    for col, emoji, label, bg, fg in specs:
        if bool(player.get(col, False)):
            # Power outage chip gets the actual HR count appended for punch.
            extra = ''
            if col == 'flag_power_outage':
                extra = f" — {int(player.get('is_hr', 0))} HR"
            chips.append(
                f'<span style="display:inline-block;background:{bg};color:{fg};'
                f'padding:4px 10px;margin:3px 4px 3px 0;border-radius:12px;'
                f'font-size:12px;font-weight:600;">{emoji} {label}{extra}</span>'
            )
    return ''.join(chips)


def render_sustain_with_model(player, history_df):
    score   = player['sustain_score']
    trend   = player['trend']
    catg    = player['category']

    # ── Talent tier (who he is) ──
    tier  = player.get('talent_tier', 'Unproven')
    tier_color, tier_emoji, tier_label = TIER_INFO.get(tier, TIER_INFO['Unproven'])

    # ── Performance magnitude (the STAR number): how far above/below his
    #    projected level, as a %. (current xwOBA − baseline) / baseline.
    base = player.get('baseline_xwOBA', np.nan)
    cur  = player.get('xwOBA', np.nan)
    magnitude_pct = ((cur - base) / base * 100) if (pd.notna(base) and base > 0 and pd.notna(cur)) else 0.0
    mag_color = '#22c55e' if magnitude_pct > 1 else ('#ef4444' if magnitude_pct < -1 else '#9ca3af')
    mag_sign  = '+' if magnitude_pct >= 0 else '−'   # explicit minus glyph
    better_worse = 'better' if magnitude_pct >= 0 else 'worse'

    # ── Performance tier in plain language (replaces Surging/Heating Up) ──
    perf = player.get('performance_tier', 'As Expected')
    perf_label, perf_color = PERF_DISPLAY.get(perf, PERF_DISPLAY['As Expected'])

    # Card border follows the performance color.
    color = perf_color

    # FORM arrow — the player's short-term trajectory (rolling 50-BBE vs season).
    # ↑ heating up, → steady, ↓ cooling down. Shown right on the headline next
    # to the archetype so the eye gets "who he is + which way he's trending" at once.
    mom_label_key = player.get('momentum_label', 'steady')
    FORM_ARROW = {'heating': '▲', 'cooling': '▼', 'steady': '▬'}
    FORM_COLOR = {'heating': '#22c55e', 'cooling': '#ef4444', 'steady': '#9ca3af'}
    FORM_WORD  = {'heating': 'Trending Up', 'cooling': 'Trending Down', 'steady': 'Steady'}
    form_arrow = FORM_ARROW.get(mom_label_key, '▬')
    form_color = FORM_COLOR.get(mom_label_key, '#9ca3af')
    form_word  = FORM_WORD.get(mom_label_key, 'Steady')
    form_html = (
        f'<span style="color:{form_color};font-size:20px;font-weight:900;" '
        f'title="Recent form (last 50 batted balls vs season)">{form_arrow}</span>'
        f'<span style="color:{form_color};font-size:13px;font-weight:600;margin-left:4px;">'
        f'{form_word}</span>'
    )
    # Breakout badge (young + Surging)
    breakout_badge = ''
    if player.get('is_breakout', False):
        breakout_badge = (
            '<span style="display:inline-block;background:#7c3aed;color:#fff;'
            'padding:4px 10px;border-radius:12px;font-size:12px;font-weight:700;'
            'margin-right:4px;">🚀 BREAKOUT</span>'
        )

    # ── SKILL vs LUCK split bar (the second STAR) — shown for EVERY player ──
    # p_luck = chance the deviation is pure chance → that's the LUCK share.
    # 100 − p_luck = chance it's a real signal → the SKILL share. We render both
    # halves of one horizontal bar: skill (green) left, luck (amber) right, with
    # the % labels above each side. The standard probability-split viz.
    pluck = player.get('p_luck_xwOBA', np.nan)
    if pd.isna(pluck):
        luck_block = ''
    else:
        luck_pct  = int(round(pluck * 100))
        skill_pct = 100 - luck_pct
        SKILL_COLOR, LUCK_COLOR = '#22c55e', '#f59e0b'
        plain = ('almost certainly real' if luck_pct < 5 else
                 'probably real'         if luck_pct < 20 else
                 'could be luck'         if luck_pct < 50 else
                 'likely just normal noise')
        luck_block = (
            f'<div style="margin-top:12px;">'
            f'<div style="font-size:11px;color:#888;letter-spacing:2px;margin-bottom:4px;">SKILL vs LUCK</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:15px;font-weight:800;margin-bottom:4px;">'
            f'<span style="color:{SKILL_COLOR};">{skill_pct}% Skill</span>'
            f'<span style="color:{LUCK_COLOR};">{luck_pct}% Luck</span>'
            f'</div>'
            f'<div style="display:flex;height:18px;border-radius:9px;overflow:hidden;background:#1f1f1f;">'
            f'<div style="width:{skill_pct}%;background:{SKILL_COLOR};"></div>'
            f'<div style="width:{luck_pct}%;background:{LUCK_COLOR};"></div>'
            f'</div>'
            f'<div style="font-size:11px;color:#888;margin-top:4px;">→ this hot start / slump is {plain}</div>'
            f'</div>'
        )

    # ── Forward-looking trajectory note (State × Form edge cases) ──
    traj = player.get('trajectory_tag', '')
    traj_block = ''
    if traj and traj in TRAJECTORY_INFO:
        t_emoji, t_color, t_meaning = TRAJECTORY_INFO[traj]
        traj_block = (
            f'<div style="margin-top:8px;padding:6px 10px;background:{t_color}22;'
            f'border-left:3px solid {t_color};border-radius:4px;font-size:12px;color:#ddd;">'
            f'{t_emoji} <b>{traj}</b> — {t_meaning}</div>'
        )

    # Projected basic stats — computed on the fly
    proj = project_basic_stats(player, history_df)

    # Surface-stat flag chips — built here so they live INSIDE the sustain card,
    # right under the verdict, where the eye lands first.
    chips = build_flag_chips(player)
    chips_block = (
        f'<div style="margin-top:10px;">{chips}</div>' if chips else ''
    )

    # ─── xwOBA-beater caveat (the speed/contact edge case) ───────────────
    # If the pipeline flagged this player as someone who reliably out-hits their
    # contact quality (speed / placement / shift-beating), we add a caveat that
    # combines that with their CONTACT FOUNDATION trend. This is the two-part
    # read: the propping is real skill, but is the foundation holding or cracking?
    beater_caveat = ''
    if player.get('xwoba_beater', False):
        foundation = player.get('contact_foundation', 'stable')
        if foundation == 'eroding':
            # The "looks OK now but worrying" case — value propped up by skill
            # while the underlying contact quality is slipping.
            beater_caveat = (
                '<div style="margin-top:10px;padding:8px 10px;background:#78350f33;'
                'border-left:3px solid #f59e0b;border-radius:4px;font-size:12px;color:#fcd34d;">'
                '⚠️ <b>Beats xwOBA via speed/contact</b> — wOBA is propped up by real skill, '
                'but contact quality (xwOBA) is <b>slipping vs his norm</b>. Holds up for now; '
                'sustainability at risk if the trend continues.</div>'
            )
        else:
            label = 'and contact is improving' if foundation == 'improving' else 'and contact is holding'
            beater_caveat = (
                '<div style="margin-top:10px;padding:8px 10px;background:#064e3b33;'
                'border-left:3px solid #10b981;border-radius:4px;font-size:12px;color:#6ee7b7;">'
                f'🦵 <b>Beats xwOBA via speed/contact</b> {label} — wOBA is skill-driven '
                'and should hold even though raw contact quality looks modest.</div>'
            )

    # All HTML built as single-line strings — blank lines break Streamlit's markdown parser.
    def cmp_row(label, current_val, projected_val, fmt, tooltip):
        if pd.isna(current_val): return ''
        proj_str = fmt(projected_val) if not pd.isna(projected_val) else '—'
        tip = tooltip.replace('"', "'")
        return (
            f'<tr title="{tip}" style="cursor:help;">'
            f'<td style="color:#aaa;padding:2px 12px 2px 0;font-size:12px;">{label}</td>'
            f'<td style="font-weight:600;font-size:13px;text-align:right;padding-right:6px;">{fmt(current_val)}</td>'
            f'<td style="color:#888;padding:0 4px;">→</td>'
            f'<td style="font-weight:600;font-size:13px;color:{color};">{proj_str}</td>'
            f'</tr>'
        )

    proj_rows = (
        cmp_row('AVG', player.get('BA'),  proj.get('BA'),  lambda v: f"{v:.3f}",
                "Projected rest-of-season batting average. Bayesian blend of current and career average.") +
        cmp_row('OBP', player.get('OBP'), proj.get('OBP'), lambda v: f"{v:.3f}",
                "Projected rest-of-season on-base percentage.") +
        cmp_row('SLG', player.get('SLG'), proj.get('SLG'), lambda v: f"{v:.3f}",
                "Projected rest-of-season slugging.") +
        cmp_row('OPS', (player.get('OBP') or 0) + (player.get('SLG') or 0),
                proj.get('OPS'), lambda v: f"{v:.3f}",
                "Projected OPS = projected OBP + projected SLG.") +
        cmp_row('HR%', player.get('HR_rate'), proj.get('HR_rate'), lambda v: f"{v*100:.1f}%",
                "Projected rest-of-season home run rate per plate appearance.")
    )

    def model_row(label, value_str, tooltip):
        tip = tooltip.replace('"', "'")
        return (
            f'<tr title="{tip}" style="cursor:help;">'
            f'<td style="color:#aaa;padding:2px 12px 2px 0;font-size:12px;border-bottom:1px dotted #444;">{label}</td>'
            f'<td style="font-weight:600;font-size:13px;padding:2px 0;">{value_str}</td>'
            f'</tr>'
        )

    dev = player['xwOBA_deviation']
    dev_color = '#22c55e' if dev >= 0 else '#ef4444'
    model_html = (
        model_row('Baseline xwOBA', f"{player['baseline_xwOBA']:.3f}", tooltip_for('baseline_xwOBA')) +
        model_row('Current xwOBA',  f"{player['xwOBA']:.3f}",          "Player's actual 2026 xwOBA so far.") +
        model_row('Deviation',      f'<span style="color:{dev_color};">{dev:+.3f}</span>', tooltip_for('xwOBA_deviation')) +
        model_row('Projected RoS',  f"{player['projected_ros_xwOBA']:.3f}", tooltip_for('projected_ros_xwOBA')) +
        model_row('Signal',         f"{player['signal_xwOBA']:.2f}",   tooltip_for('signal_xwOBA'))
    )

    # Entire card as ONE continuous HTML string — no blank lines, no leading
    # whitespace lines. Streamlit's markdown parser sees blank lines as the end
    # of an HTML block and bails out, so we keep everything on one logical line.
    # Entire card as ONE continuous HTML string — no blank lines (Streamlit's
    # markdown parser treats a blank line as the end of an HTML block).
    # HEADLINE is now the archetype (Tier — State), not the single score.
    # The old 0-100 sustain score is DEMOTED to a supporting "Skill-Support" line.
    html = (
        f'<div style="background:linear-gradient(135deg,{color}1a 0%,{color}33 100%);'
        f'border:2px solid {color};border-radius:14px;padding:14px 18px;">'
        # Tier label (who he is) on the left, FORM arrow pushed to the right.
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f'<span style="font-size:13px;font-weight:800;color:{tier_color};letter-spacing:1px;">{tier_emoji} {tier_label}</span>'
        f'<span style="white-space:nowrap;"><span style="font-size:11px;color:#666;margin-right:4px;">FORM</span>{form_html}</span>'
        f'</div>'
        # THE STAR #1: magnitude vs projected level — huge number.
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:8px;">'
        f'<span style="font-size:46px;font-weight:900;color:{mag_color};line-height:1;">{mag_sign}{abs(magnitude_pct):.0f}%</span>'
        f'<span style="font-size:13px;color:#bbb;line-height:1.2;">{better_worse} than his<br>projected level</span>'
        f'</div>'
        # Performance tier label (plain language)
        f'<div style="font-size:16px;font-weight:700;color:{perf_color};margin-top:4px;">{perf_label}</div>'
        # THE STAR #2: luck %.
        f'{luck_block}'
        # Breakout + flag chips
        f'<div style="margin-top:10px;">{breakout_badge}{chips}</div>'
        f'{traj_block}'
        f'{beater_caveat}'
        # Demoted score: Skill-Support (small, supporting)
        f'<div style="margin-top:12px;font-size:11px;color:#777;">'
        f'Skill-Support {score:.0f}/100 '
        f'<span style="color:#555;">· do the peripherals (K%, barrels…) back the move</span></div>'
        f'<div style="font-size:10px;color:#888;letter-spacing:2px;margin-top:12px;">PROJECTED REST-OF-SEASON</div>'
        f'<table style="width:100%;margin-top:4px;">{proj_rows}</table>'
        f'<div style="font-size:10px;color:#888;letter-spacing:2px;margin-top:12px;">'
        f'MODEL DETAILS&nbsp;<span style="color:#666;">(hover for explanations)</span></div>'
        f'<table style="width:100%;margin-top:4px;">{model_html}</table>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4-SEASON HITTING TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def render_season_table(player_id, history_df):
    rows = history_df[history_df['batter'] == player_id].sort_values('season')
    if len(rows) == 0:
        st.info("No multi-season history for this player.")
        return

    team_codes  = rows.get('team', pd.Series([None]*len(rows))).tolist()
    team_logos  = [team_logo_url(t) for t in team_codes]
    team_labels = [t if pd.notna(t) else '—' for t in team_codes]

    # "+" stats (league-normalized, 100 = avg) make seasons comparable across
    # different run environments — a 962 OPS in one year isn't necessarily
    # better than an 890 in another. We show them right next to the raw stats.
    def _plus(col):
        return rows[col].round(0).astype('Int64') if col in rows.columns else pd.NA

    display = pd.DataFrame({
        'Season': rows['season'].astype(int).astype(str),
        'Team Logo': team_logos,
        'Team': team_labels,
        'PA':  rows['PA'].astype(int),
        'AB':  rows['AB'].astype(int),
        'H':   rows['is_hit'].astype(int),
        'HR':  rows['is_hr'].astype(int),
        'BB':  rows['is_bb'].astype(int),
        'SO':  rows['is_k'].astype(int),
        'AVG': rows['BA'].round(3),
        'OBP': rows['OBP'].round(3),
        'SLG': rows['SLG'].round(3),
        'OPS': (rows['OBP'] + rows['SLG']).round(3),
        'OPS+':   _plus('OPS_plus'),
        'wOBA+':  _plus('wOBA_plus'),
        'xwOBA+': _plus('xwOBA_plus'),
    })
    totals = pd.DataFrame({
        'Season':    [f'{len(rows)} Seasons'],
        'Team Logo': [None],
        'Team':      ['—'],
        'PA':  [int(rows['PA'].sum())], 'AB': [int(rows['AB'].sum())],
        'H':   [int(rows['is_hit'].sum())], 'HR': [int(rows['is_hr'].sum())],
        'BB':  [int(rows['is_bb'].sum())], 'SO': [int(rows['is_k'].sum())],
        'AVG': [round(rows['is_hit'].sum() / rows['AB'].sum(), 3) if rows['AB'].sum() else 0],
        'OBP': [round((rows['OBP'] * rows['PA']).sum() / rows['PA'].sum(), 3)],
        'SLG': [round((rows['SLG'] * rows['AB']).sum() / rows['AB'].sum(), 3) if rows['AB'].sum() else 0],
        'OPS': [None],
        # "+" stats: PA-weighted career averages (so a big season counts more).
        'OPS+':   [int(round((rows['OPS_plus']   * rows['PA']).sum() / rows['PA'].sum())) if 'OPS_plus'   in rows.columns and rows['PA'].sum() else pd.NA],
        'wOBA+':  [int(round((rows['wOBA_plus']  * rows['PA']).sum() / rows['PA'].sum())) if 'wOBA_plus'  in rows.columns and rows['PA'].sum() else pd.NA],
        'xwOBA+': [int(round((rows['xwOBA_plus'] * rows['PA']).sum() / rows['PA'].sum())) if 'xwOBA_plus' in rows.columns and rows['PA'].sum() else pd.NA],
    })
    totals['OPS'] = round(totals['OBP'].iloc[0] + totals['SLG'].iloc[0], 3)
    full = pd.concat([display, totals], ignore_index=True)

    st.dataframe(
        full, hide_index=True, use_container_width=True,
        column_config={
            'Team Logo': st.column_config.ImageColumn("Logo", width="small"),
            'Team': st.column_config.TextColumn("Team"),
        },
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PERCENTILE BARS
# Auto-includes chase/whiff if those columns exist in comparison.parquet,
# silently skips if they don't (so the pipeline change is optional).
# ═══════════════════════════════════════════════════════════════════════════════
PERCENTILE_SPECS = [
    ('xwOBA',         lambda v: f"{v:.3f}",        False),
    ('xBA',           lambda v: f"{v:.3f}",        False),
    ('xSLG',          lambda v: f"{v:.3f}",        False),
    ('wOBA',          lambda v: f"{v:.3f}",        False),
    ('avg_exit_velo', lambda v: f"{v:.1f} mph",    False),
    ('barrel_rate',   lambda v: f"{v*100:.1f}%",   False),
    ('hard_hit_rate', lambda v: f"{v*100:.1f}%",   False),
    ('avg_bat_speed', lambda v: f"{v:.1f} mph",    False),
    ('BABIP',         lambda v: f"{v:.3f}",        False),
    ('HR_rate',       lambda v: f"{v*100:.1f}%",   False),
    ('K_rate',        lambda v: f"{v*100:.1f}%",   True),
    ('BB_rate',       lambda v: f"{v*100:.1f}%",   False),
    ('chase_rate',    lambda v: f"{v*100:.1f}%",   True),   # lower=better
    ('whiff_rate',    lambda v: f"{v*100:.1f}%",   True),   # lower=better
]

def render_percentile_bar(label, value_str, pct, tooltip=""):
    color = percentile_color(pct)
    pct_int = int(round(pct))
    tip = tooltip.replace('"', "'")
    return f"""
    <div title="{tip}" style="display:flex; align-items:center; gap:12px; padding:5px 0; cursor:help;">
        <div style="width:110px; font-size:13px; color:#ddd;">{label}</div>
        <div style="flex:1; position:relative; height:12px; background:#1f1f1f; border-radius:6px;">
            <div style="position:absolute; left:0; top:0; width:{pct}%; height:100%;
                        background:linear-gradient(90deg, #2c5282, {color}); border-radius:6px;"></div>
            <div style="position:absolute; left:calc({pct}% - 13px); top:-7px;
                        width:26px; height:26px; line-height:26px; text-align:center;
                        background:{color}; color:white; border-radius:50%;
                        font-size:11px; font-weight:bold;
                        box-shadow:0 1px 3px rgba(0,0,0,0.5);">{pct_int}</div>
        </div>
        <div style="width:100px; text-align:right; font-size:13px; color:#fff; font-weight:600; white-space:nowrap;">{value_str}</div>
    </div>
    """

def render_percentile_section(player_row, season_pool):
    """Render the Savant-style percentile bars for ONE player in ONE season.

    Parameters
    ----------
    player_row : a single player's stat row FOR THE SELECTED SEASON. This comes
        from player_history (filtered to that season), NOT from comparison.
        That's what makes the section season-selectable — pass 2023's row to see
        2023's percentiles, 2026's row to see 2026's, etc.
    season_pool : every player's row for that SAME season. We rank player_row
        against this pool. We then narrow it to QUALIFIED players (so the
        percentile means "vs full-time hitters that year"), with a fallback to
        the full pool if the qualified subset would be empty.

    The percentile for each stat = what % of the season's qualified pool this
    player beats. Inverted stats (K%, Chase%, Whiff%) flip so that "good" is
    always a high percentile / red bar.
    """
    pool = season_pool
    if 'qualified' in season_pool.columns:
        candidate = season_pool[season_pool['qualified'] == 1]
        if len(candidate) > 0:
            pool = candidate

    # For xwOBA / wOBA we append the league-normalized "+" value to the
    # displayed number (e.g. ".390  (142+)") so the bar carries both the raw
    # stat AND the cross-season-comparable "+" figure. 100 = league average.
    PLUS_COL = {'xwOBA': 'xwOBA_plus', 'wOBA': 'wOBA_plus'}

    bars_html = ""
    for col, fmt, invert in PERCENTILE_SPECS:
        if col not in pool.columns:
            continue   # stat not present this season (rare)
        val = player_row.get(col)
        if pd.isna(val):
            continue   # player has no value for this stat (e.g., bat speed in a missing year)
        pct = percentile_rank(pool[col], val, invert=invert)
        label = STAT_DEFS.get(col, {}).get('name', col)
        value_str = fmt(val)
        # Append the "+" stat where we have one.
        plus_col = PLUS_COL.get(col)
        if plus_col and plus_col in player_row.index and pd.notna(player_row.get(plus_col)):
            value_str += f"  ({int(round(player_row[plus_col]))}+)"
        bars_html += render_percentile_bar(label, value_str, pct, tooltip_for(col))
    st.markdown(bars_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RADAR CHARTS — multi-season + current vs baseline
# ═══════════════════════════════════════════════════════════════════════════════
RADAR_AXES = [
    ('xwOBA',         'xwOBA',       False),
    ('xBA',           'xBA',         False),
    ('barrel_rate',   'Barrel%',     False),
    ('hard_hit_rate', 'HardHit%',    False),
    ('avg_exit_velo', 'Exit Velo',   False),
    ('K_rate',        'K%',          True),
    ('BB_rate',       'BB%',         False),
]
RADAR_LABELS = [a[1] for a in RADAR_AXES]
SEASON_COLORS = {
    2023: ('#374151', 'rgba(55, 65, 81, 0.15)'),
    2024: ('#5b21b6', 'rgba(91, 33, 182, 0.18)'),
    2025: ('#1e3a8a', 'rgba(30, 58, 138, 0.20)'),
    2026: ('#991b1b', 'rgba(153, 27, 27, 0.30)'),
}

def _style_radar(fig, height=440):
    fig.update_layout(
        polar=dict(
            bgcolor='#0e1117',
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color='#888'), gridcolor='#333'),
            angularaxis=dict(tickfont=dict(size=11, color='#ddd'), gridcolor='#333'),
        ),
        paper_bgcolor='#0e1117',
        showlegend=True, height=height,
        margin=dict(t=30, b=20, l=40, r=40),
        legend=dict(font=dict(color='#ddd')),
    )

def radar_multi_season(player_id, history_df):
    """Percentiles computed vs the QUALIFIED (300+ PA) pool each season so the
    'vs league' comparison stays meaningful even though partial-season players
    are now included in the dataset."""
    fig = go.Figure()
    for season in sorted(history_df['season'].unique()):
        sdf = history_df[history_df['season'] == season]
        # Pool is qualified players for that season; fall back if empty.
        pool = sdf
        if 'qualified' in sdf.columns:
            candidate = sdf[sdf['qualified'] == 1]
            if len(candidate) > 0: pool = candidate
        prow = sdf[sdf['batter'] == player_id]
        if len(prow) == 0: continue
        prow = prow.iloc[0]
        pcts = [percentile_rank(pool[a[0]], prow[a[0]], a[2]) for a in RADAR_AXES]
        lc, fc = SEASON_COLORS.get(season, ('#888', 'rgba(136,136,136,0.15)'))
        fig.add_trace(go.Scatterpolar(
            r=pcts + [pcts[0]], theta=RADAR_LABELS + [RADAR_LABELS[0]],
            fill='toself', name=str(season),
            line=dict(color=lc, width=2.5), fillcolor=fc,
        ))
    _style_radar(fig)
    return fig

def radar_current_vs_baseline(player_row, league_df):
    # Qualified pool with empty-fallback.
    pool = league_df
    if 'qualified' in league_df.columns:
        candidate = league_df[league_df['qualified'] == 1]
        if len(candidate) > 0: pool = candidate
    current_pcts  = [percentile_rank(pool[a[0]], player_row[a[0]], a[2]) for a in RADAR_AXES]
    baseline_pcts = []
    for stat, _, invert in RADAR_AXES:
        bcol = f'baseline_{stat}'
        if bcol in pool.columns:
            baseline_pcts.append(percentile_rank(pool[bcol], player_row[bcol], invert))
        else:
            baseline_pcts.append(50)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=baseline_pcts + [baseline_pcts[0]], theta=RADAR_LABELS + [RADAR_LABELS[0]],
        fill='toself', name='Baseline',
        line=dict(color='#475569', width=2, dash='dot'),
        fillcolor='rgba(71, 85, 105, 0.15)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=current_pcts + [current_pcts[0]], theta=RADAR_LABELS + [RADAR_LABELS[0]],
        fill='toself', name='2026',
        line=dict(color='#991b1b', width=3),
        fillcolor='rgba(153, 27, 27, 0.30)',
    ))
    _style_radar(fig)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PERIPHERAL SUPPORT SPIDER — does each peripheral back the current trend?
# 4 axes. Each: 0 = strongly opposes trend, 50 = neutral, 100 = strongly supports.
# Recomputed on the fly from existing comparison columns; no pipeline change needed.
# ═══════════════════════════════════════════════════════════════════════════════
PERIPHERAL_SIGMAS = {'K_rate': 0.040, 'BB_rate': 0.020, 'barrel_rate': 0.040, 'xwOBA_diff': 0.020}

def peripheral_spider(player):
    direction = player.get('direction', 0)
    if direction == 0:
        return None   # no trend → peripheral support is undefined

    def signed_support(val, sigma):
        # Soft-cap version: 2 × tanh(z/2) maps any z-score to [-2, +2] with
        # diminishing returns. A 3σ move now visibly outruns a 1σ move on the
        # spider, instead of both being mashed to "fully supports."
        # Spider axis = 50 + 25 × support (since support spans ±2, this still
        # covers the 0..100 range for visualization).
        z = val / sigma
        soft = 2 * np.tanh(z / 2)
        return 50 + 25 * soft * direction

    # K_rate inverted: K going DOWN supports hot, K going UP supports cold
    k_supp       = signed_support(-player.get('K_rate_deviation', 0),       PERIPHERAL_SIGMAS['K_rate'])
    bb_supp      = signed_support( player.get('BB_rate_deviation', 0),      PERIPHERAL_SIGMAS['BB_rate'])
    barrel_supp  = signed_support( player.get('barrel_rate_deviation', 0),  PERIPHERAL_SIGMAS['barrel_rate'])
    contact_supp = signed_support( player.get('xwOBA_diff', 0),             PERIPHERAL_SIGMAS['xwOBA_diff'])

    labels = ['K Rate', 'BB Rate', 'Barrel %', 'Contact Quality']
    values = [k_supp, bb_supp, barrel_supp, contact_supp]

    trend_color = '#dc2626' if direction > 0 else '#2563eb'

    fig = go.Figure()
    # Neutral reference ring at 50
    fig.add_trace(go.Scatterpolar(
        r=[50]*5, theta=labels + [labels[0]],
        line=dict(color='#666', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill='toself', name='Peripheral support',
        line=dict(color=trend_color, width=3),
        fillcolor=f"rgba({'220, 38, 38' if direction>0 else '37, 99, 235'}, 0.30)",
    ))
    _style_radar(fig, height=380)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# HIT-TYPE BREAKDOWN CHART
# Horizontal stacked bar per season. Shows the player's contact-profile evolution.
# Wider GB segment = ground-ball hitter. Wider FB segment = power-tilted swing.
# Wider LD segment = elite contact (line drives are the best outcome).
# ═══════════════════════════════════════════════════════════════════════════════
HIT_TYPE_COLORS = {
    'GB': '#92400e',  # brown — ground balls
    'LD': '#16a34a',  # green — line drives (the good ones)
    'FB': '#2563eb',  # blue — fly balls
    'PU': '#9ca3af',  # gray — popups (the bad ones)
}
HIT_TYPE_LABELS = {
    'GB': 'Ground Balls',
    'LD': 'Line Drives',
    'FB': 'Fly Balls',
    'PU': 'Popups',
}

def hit_type_chart(player_id, history_df, league_refs=None):
    """Stacked horizontal bar — one bar per season + an optional 'League Avg'
    reference bar at the top so viewers can see how the player compares to a
    typical qualified hitter."""
    rows = history_df[history_df['batter'] == player_id].sort_values('season', ascending=False)
    if len(rows) == 0 or 'GB_rate' not in rows.columns:
        return None

    season_labels = rows['season'].astype(int).astype(str).tolist()

    # Decide whether to prepend the league reference row. We label it with a
    # ⚖️ marker so it's instantly distinguishable from the player's own seasons
    # (otherwise it looks like just another year and gets missed).
    show_league = bool(league_refs and league_refs.get('hit_type'))
    LEAGUE_LABEL = '⚖️ League Avg'
    y_labels = ([LEAGUE_LABEL] if show_league else []) + season_labels

    fig = go.Figure()
    for ht in ['GB', 'LD', 'FB', 'PU']:
        col = f'{ht}_rate'
        if col not in rows.columns:
            continue
        season_values = (rows[col] * 100).round(1).tolist()
        if show_league:
            league_value = round(league_refs['hit_type'].get(ht, 0) * 100, 1)
            values = [league_value] + season_values
        else:
            values = season_values

        # Slightly fade the league row's bars so the player's own seasons pop.
        # opacity per y-row: league row dimmer, seasons full strength.
        bar_opacity = [0.55 if lbl == LEAGUE_LABEL else 1.0 for lbl in y_labels]

        fig.add_trace(go.Bar(
            y=y_labels,
            x=values,
            name=HIT_TYPE_LABELS[ht],
            orientation='h',
            marker=dict(color=HIT_TYPE_COLORS[ht], opacity=bar_opacity),
            text=[f"{v:.1f}%" if v >= 5 else '' for v in values],
            textposition='inside',
            textfont=dict(color='white', size=12),
            hovertemplate=f'{HIT_TYPE_LABELS[ht]}: %{{x:.1f}}%<extra></extra>',
        ))

    fig.update_layout(
        barmode='stack',
        height=80 + 60 * len(y_labels),
        xaxis=dict(title='% of batted balls', range=[0, 100],
                   color='#ddd', gridcolor='#333'),
        yaxis=dict(title='', color='#ddd', gridcolor='#333',
                   categoryorder='array', categoryarray=y_labels[::-1]),  # top→bottom: League, then most recent season
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        showlegend=True,
        legend=dict(orientation='h', y=-0.18, font=dict(color='#ddd')),
        margin=dict(t=20, b=80, l=80, r=20),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PITCH-TYPE PERFORMANCE CHART
# Bar chart showing the batter's xwOBA on contact vs each pitch family.
# Reference dashed line = league average xwOBA against that family
# (computed from the qualified pool).
# ═══════════════════════════════════════════════════════════════════════════════
def pitch_type_chart(player, comparison_df, league_refs=None):
    """Vertical bar chart of player's xwOBA on contact vs each pitch family,
    with a PA-weighted league reference line.

    Why PA-weighted reference instead of unweighted player-mean:
    A simple mean of player-level xwOBAs weights a 30-PA hitter the same as
    a 300-PA hitter. The PA-weighted version (precomputed in league_refs)
    treats each plate appearance equally, which is the honest league avg."""
    categories = ['Fastball', 'Breaking', 'Offspeed']
    player_xwoba = []
    player_pas   = []
    league_xwoba = []
    for cat in categories:
        xcol = f'xwOBA_vs_{cat}'
        pcol = f'PAs_vs_{cat}'
        player_xwoba.append(player.get(xcol, np.nan))
        player_pas.append(player.get(pcol, np.nan))
        # Prefer the cached, PA-weighted reference; fall back to unweighted
        # if the cache isn't available for some reason.
        if league_refs and league_refs.get('pitch_xwoba', {}).get(cat) is not None:
            league_xwoba.append(league_refs['pitch_xwoba'][cat])
        else:
            pool = comparison_df
            if 'qualified' in comparison_df.columns:
                cand = comparison_df[comparison_df['qualified'] == 1]
                if len(cand) > 0: pool = cand
            league_xwoba.append(pool[xcol].mean() if xcol in pool.columns else np.nan)

    # If literally no data at all, bail out
    if all(pd.isna(v) for v in player_xwoba):
        return None

    # Color each bar by how the player compares to league avg for that family
    colors = []
    for p, l in zip(player_xwoba, league_xwoba):
        if pd.isna(p) or pd.isna(l):
            colors.append('#6b7280')
        elif p - l > 0.04:    colors.append('#dc2626')   # elite
        elif p - l > 0.015:   colors.append('#ea580c')   # above avg
        elif p - l > -0.015:  colors.append('#9ca3af')   # roughly avg
        elif p - l > -0.04:   colors.append('#3b82f6')   # below
        else:                 colors.append('#1e40af')   # poor

    fig = go.Figure()
    # Player bars with xwOBA values on top
    fig.add_trace(go.Bar(
        x=categories,
        y=player_xwoba,
        marker=dict(color=colors),
        text=[f"{v:.3f}<br><span style='font-size:10px;color:#888;'>{int(pa)} PAs</span>"
              if not pd.isna(v) else 'no data'
              for v, pa in zip(player_xwoba, player_pas)],
        textposition='outside',
        textfont=dict(color='#fff', size=13),
        name='Player',
        hovertemplate='Player: %{y:.3f}<extra></extra>',
    ))
    # League average overlay as small markers
    fig.add_trace(go.Scatter(
        x=categories,
        y=league_xwoba,
        mode='markers',
        marker=dict(symbol='line-ew', size=40, color='#888', line=dict(width=3)),
        name='League Avg',
        hovertemplate='League Avg: %{y:.3f}<extra></extra>',
    ))

    fig.update_layout(
        height=420,
        yaxis=dict(title='xwOBA on contact', color='#ddd', gridcolor='#333',
                   range=[0.2, max([v for v in player_xwoba if not pd.isna(v)] + [0.5]) + 0.05]),
        xaxis=dict(color='#ddd', gridcolor='#333'),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        showlegend=True,
        legend=dict(font=dict(color='#ddd')),
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# TIME-SERIES
# ═══════════════════════════════════════════════════════════════════════════════
TIME_SERIES_OPTIONS = {
    'xwOBA':'xwOBA', 'xBA':'xBA', 'xSLG':'xSLG', 'wOBA':'wOBA (park-neutral)',
    'BA':'AVG', 'OBP':'OBP', 'SLG':'SLG', 'BABIP':'BABIP',
    'K_rate':'K%', 'BB_rate':'BB%', 'HR_rate':'HR Rate',
    'barrel_rate':'Barrel%', 'hard_hit_rate':'HardHit%',
    'avg_exit_velo':'Avg Exit Velo (mph)', 'avg_bat_speed':'Avg Bat Speed (mph)',
    'avg_launch_angle':'Avg Launch Angle (°)',
}

def time_series_chart(player_id, stat_col, stat_label, history_df):
    player_data = history_df[history_df['batter'] == player_id].sort_values('season')
    # League average over qualified pool, falling back to full df where empty.
    league_pool = history_df
    if 'qualified' in history_df.columns:
        candidate = history_df[history_df['qualified'] == 1]
        if len(candidate) > 0: league_pool = candidate
    league_avg = league_pool.groupby('season')[stat_col].mean().sort_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=league_avg.index.astype(int), y=league_avg.values,
        mode='lines', name='League Avg',
        line=dict(color='#888', dash='dash', width=2),
    ))
    if len(player_data):
        fig.add_trace(go.Scatter(
            x=player_data['season'].astype(int), y=player_data[stat_col],
            mode='lines+markers+text', name=player_data.iloc[0]['name'].title(),
            line=dict(color='#dc2626', width=3),
            marker=dict(size=14, color='#dc2626', line=dict(width=2, color='white')),
            text=[f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}" for v in player_data[stat_col]],
            textposition='top center', textfont=dict(color='#fff', size=11),
        ))
    fig.update_layout(
        title=dict(text=stat_label, font=dict(color='#ddd', size=16)),
        xaxis=dict(title='Season', tickmode='array',
                   tickvals=sorted(history_df['season'].astype(int).unique()),
                   color='#ddd', gridcolor='#333'),
        yaxis=dict(color='#ddd', gridcolor='#333'),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        height=380, showlegend=True,
        legend=dict(font=dict(color='#ddd')),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════════
def render_glossary(league_refs=None):
    md = "### Stats — definitions + what counts as good\n\n"
    for key, d in STAT_DEFS.items():
        if key.startswith('baseline_') or key in ('xwOBA_deviation','projected_ros_xwOBA','signal_xwOBA'):
            continue
        md += f"**{d['name']}** — {d['defn']}  \n*{d['bench']}*\n\n"

    # If we have computed league references, surface them so viewers see
    # the actual benchmark numbers from THIS dataset (not approximations).
    if league_refs:
        md += "\n---\n### League reference values (computed from 2024-25 qualified pool)\n\n"
        if league_refs.get('hit_type'):
            md += "**Hit-type mix — what a typical qualified hitter looks like:**\n\n"
            ht_map = {'GB': 'Ground balls', 'LD': 'Line drives', 'FB': 'Fly balls', 'PU': 'Popups'}
            md += "| Bucket | Launch angle | League rate |\n|---|---|---|\n"
            for ht, label in ht_map.items():
                rate = league_refs['hit_type'].get(ht)
                if rate is not None:
                    angle = {'GB':'< 10°','LD':'10–25°','FB':'25–50°','PU':'≥ 50°'}[ht]
                    md += f"| **{label}** | {angle} | {rate*100:.1f}% |\n"
            md += "\n"
        if league_refs.get('pitch_xwoba'):
            md += "**xwOBA on contact vs each pitch family — league baseline:**\n\n"
            md += "| Pitch family | League xwOBA on contact |\n|---|---|\n"
            for cat, val in league_refs['pitch_xwoba'].items():
                md += f"| **{cat}** | {val:.3f} |\n"
            md += "\n*A bar that exceeds these values means the player is above league average against that family.*\n\n"
    md += """
---
### Model concepts
**Baseline** — Marcel-style projection from 2023-25 history (PA-weighted, regressed to league mean, age-adjusted).

**Deviation** — Current 2026 stat minus baseline. Positive = above expectation, negative = below.

**Sustain Score (0-100)** — Whether 4 peripheral skills (K%, BB%, barrel%, contact quality) back the trend. 75+ = strongly support; <25 = strongly oppose.

**Verdict tiers**
- 🔥🚀 *Hot streak is fully sustainable* (75+ hot)
- 🔥📈 *Hot streak is mostly sustainable* (60-74)
- 🤔 *Mixed signals — could go either way* (40-59)
- 👀 *Hot but sus — peripherals don't back it* (25-39 hot)
- 🎭 *Hot streak is mostly luck — mirage* (<25 hot)
- 🥶❄️ *Slump is fully real — ice cold* (75+ cold)
- 📉😞 *Slump is mostly real* (60-74)
- 🍀 *Slump is mostly bad luck* (25-39 cold)
- 💎 *Pure bad luck — buy low* (<25 cold)

**Projected RoS** — Rest-of-season xwOBA. Bayesian blend of 2026 current and aged baseline weighted by 2026 PA.

**Signal** — Below 0.10 = ignore the numbers, sample too small.

**Peripheral support spider** — Per axis: 0=peripheral strongly opposes current trend, 50=neutral, 100=strongly supports.
"""
    st.markdown(md)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — LEADERBOARDS
# ═══════════════════════════════════════════════════════════════════════════════
# Ordering for the tier axes (best → worst), used for the matrix + sorting.
TIER_ORDER = ['Elite', 'Above-Average', 'Average', 'Below-Average', 'Unproven']
PERF_ORDER = ['Unfathomably High', 'Unusually High', 'Notably Over', 'As Expected',
              'Notably Under', 'Unusually Low', 'Unfathomably Low']
MOM_ARROW = {'heating': '▲', 'cooling': '▼', 'steady': '▬'}

def _flags_emoji(row):
    """Compact emoji string of a player's active surface-stat flags."""
    e = []
    if row.get('is_breakout'):      e.append('🚀')
    if row.get('flag_buy_low'):     e.append('💎')
    if row.get('flag_sell_high'):   e.append('📉')
    if row.get('flag_power_outage'):e.append('🔋')
    if row.get('flag_power_surge'): e.append('💥')
    if row.get('flag_k_spike'):     e.append('⚡')
    if row.get('flag_k_improve'):   e.append('✅')
    return ''.join(e)

# Preset "views" — each is a filter answering a common discovery question.
# (name → function that filters the comparison df)
LEADERBOARD_PRESETS = {
    'All hitters':            lambda d: d,
    '🚀 Real Breakouts':      lambda d: d[d['is_breakout']],
    '🔥 Real Surges':         lambda d: d[d['current_state'] == 'Surging'],
    '💎 Buy-Low (unlucky)':   lambda d: d[d['flag_buy_low']],
    '📉 Sell-High (lucky)':   lambda d: d[d['flag_sell_high'] | (d['current_state'] == 'Hot but Lucky')],
    '❄️ Genuine Slumps':      lambda d: d[d['current_state'] == 'Declining'],
    '🎢 Riding Hot Luck':     lambda d: d[d['current_state'] == 'Hot but Lucky'],
    '🔋 Power Outages':       lambda d: d[d['flag_power_outage']],
    '⚡ K% Spiking':          lambda d: d[d['flag_k_spike']],
    '🌱 Breakout Building':   lambda d: d[d['trajectory_tag'] == 'Breakout Building'],
    '⚠️ Fading (warning)':    lambda d: d[d['trajectory_tag'] == 'Fading'],
    '📈 Bottoming Out':       lambda d: d[d['trajectory_tag'] == 'Bottoming Out'],
}

def build_leaderboard_table(d):
    """Turn a filtered comparison slice into a clean archetype-language table."""
    return pd.DataFrame({
        'Player':      d['name'].str.title(),
        'Team':        d.get('team', pd.Series(['—'] * len(d), index=d.index)),
        'Talent':      d['talent_tier'],
        'State':       d['current_state'],
        'Performance': d['performance_tier'],
        'Form':        d['momentum_label'].map(MOM_ARROW).fillna('▬'),
        'Trajectory':  d.get('trajectory_tag', pd.Series([''] * len(d), index=d.index)),
        'Tags':        d.apply(_flags_emoji, axis=1),
        'PA':          d['PA'].astype(int),
        'xwOBA':       d['xwOBA'].round(3),
        'Base':        d['baseline_xwOBA'].round(3),
        'Proj':        d['projected_ros_xwOBA'].round(3),
        'p(luck)%':    (d['p_luck_xwOBA'] * 100).round(0),
        'xwOBA+':      d['xwOBA_plus'].round(0).astype('Int64'),
    })

def render_leaderboards(df):
    st.title("⚾ MLB Sustainability — Explorer")
    st.markdown(
        "Browse and group the league by **who a player truly is** (Talent Tier) and "
        "**what's happening to him right now** (Performance + State). Use a Quick View "
        "for common questions, or open Custom Filters to build your own query "
        "(e.g. *Elite-talent guys who are Unusually Low with a buy-low flag*)."
    )

    # ── KPI strip (archetype language) ──
    nonrk = df[df['category'] != 'ROOKIE']
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Players", len(df))
    c2.metric("🚀 Breakouts", int(df['is_breakout'].sum()))
    c3.metric("💎 Buy-Low", int(df['flag_buy_low'].sum()))
    c4.metric("📉 Sell-High", int(df['flag_sell_high'].sum()))
    c5.metric("❄️ Real Slumps", int((nonrk['current_state'] == 'Declining').sum()))
    st.divider()

    # ── Quick View preset ──
    preset_name = st.selectbox("Quick View", list(LEADERBOARD_PRESETS.keys()), index=0)
    view = LEADERBOARD_PRESETS[preset_name](df).copy()

    # ── Custom filters (refine within the preset) ──
    with st.expander("🔧 Custom filters"):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_tiers = st.multiselect("Talent Tier", TIER_ORDER, default=[])
            min_pa = st.slider("Min PA", 0, int(df['PA'].max()), 0, step=10)
        with f2:
            sel_perf = st.multiselect("Performance", PERF_ORDER, default=[])
            sel_form = st.multiselect("Form", ['heating', 'steady', 'cooling'], default=[])
        with f3:
            sel_flags = st.multiselect(
                "Must have flag",
                ['is_breakout', 'flag_buy_low', 'flag_sell_high',
                 'flag_power_outage', 'flag_power_surge', 'flag_k_spike', 'flag_k_improve'],
                default=[],
                format_func=lambda c: {
                    'is_breakout':'🚀 Breakout','flag_buy_low':'💎 Buy-Low',
                    'flag_sell_high':'📉 Sell-High','flag_power_outage':'🔋 Power Outage',
                    'flag_power_surge':'💥 Power Surge','flag_k_spike':'⚡ K% Spike',
                    'flag_k_improve':'✅ K% Improve'}[c],
            )

    # Apply custom filters (empty multiselect = no constraint)
    if sel_tiers: view = view[view['talent_tier'].isin(sel_tiers)]
    if sel_perf:  view = view[view['performance_tier'].isin(sel_perf)]
    if sel_form:  view = view[view['momentum_label'].isin(sel_form)]
    view = view[view['PA'] >= min_pa]
    for fcol in sel_flags:
        view = view[view[fcol]]

    # ── Results table ──
    st.markdown(f"**{len(view)} players** — click any column header to sort")
    table = build_leaderboard_table(view.sort_values('xwOBA_plus', ascending=False))
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            'xwOBA':    st.column_config.NumberColumn(format="%.3f"),
            'Base':     st.column_config.NumberColumn(format="%.3f"),
            'Proj':     st.column_config.NumberColumn(format="%.3f"),
            'p(luck)%': st.column_config.NumberColumn(format="%.0f%%",
                          help="Probability the xwOBA deviation is pure chance. Low = real."),
            'xwOBA+':   st.column_config.NumberColumn(help="100 = league avg. League-normalized."),
            'Form':     st.column_config.TextColumn(help="Recent 50-BBE trend: ▲ up ▬ steady ▼ down"),
            'Tags':     st.column_config.TextColumn(help="🚀breakout 💎buy-low 📉sell-high 🔋power-outage 💥power-surge ⚡K-spike ✅K-improve"),
        },
    )

    # ── The landscape: Talent × Performance count matrix ──
    with st.expander("🗺️ Talent × Performance landscape (full league)"):
        st.caption("How many players sit in each Talent × Performance cell. The diagonal "
                   "(elite→performing-high, weak→performing-low) is 'as expected'; off-diagonal "
                   "cells are the interesting over/under-performers.")
        mat = pd.crosstab(df['talent_tier'], df['performance_tier'])
        mat = mat.reindex(
            index=[t for t in TIER_ORDER if t in mat.index],
            columns=[p for p in PERF_ORDER if p in mat.columns],
        ).fillna(0).astype(int)
        st.dataframe(mat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL DIAGNOSIS — heatmaps, spray, pitch-mix trend
# ═══════════════════════════════════════════════════════════════════════════════
_SWING_DESC = ['swinging_strike', 'foul', 'hit_into_play', 'swinging_strike_blocked',
               'foul_tip', 'foul_bunt', 'missed_bunt']
_WHIFF_DESC = ['swinging_strike', 'swinging_strike_blocked', 'foul_tip', 'missed_bunt']

# Heatmap grid (catcher's view). 0.2 ft cells → fine enough that Gaussian
# smoothing produces a continuous gradient, coarse enough that ~250 pitches/
# player still light up most relevant cells.
_HM_XBINS = np.linspace(-1.6, 1.6, 17)   # 16 cells across
_HM_YBINS = np.linspace(0.5,  4.5, 21)   # 20 cells tall
_HM_XCENTERS = (_HM_XBINS[:-1] + _HM_XBINS[1:]) / 2
_HM_YCENTERS = (_HM_YBINS[:-1] + _HM_YBINS[1:]) / 2
_HM_SIGMA = 1.4         # smoothing radius in cells (≈ 0.28 ft Gaussian σ)
_HM_MIN_COVERAGE = 0.25 # mask cells with less than this effective sample post-blur

def _smooth_density(x, y, sigma=_HM_SIGMA):
    """Bin (x, y) on the heatmap grid and return smoothed COUNTS (density)."""
    counts, _, _ = np.histogram2d(x, y, bins=[_HM_XBINS, _HM_YBINS])
    return _gaussian_filter(counts, sigma=sigma).T   # transpose: rows = y, cols = x

def _smooth_rate(x, y, weights, sigma=_HM_SIGMA):
    """Smooth a WEIGHTED MEAN over the heatmap grid.

    For a ratio (e.g. whiff rate, damage xwOBA), the right way to smooth is to
    blur the numerator and denominator SEPARATELY, then divide. Cells with too
    little effective coverage after blur are masked to NaN so we don't paint
    garbage from one stray sample.
    """
    counts, _, _ = np.histogram2d(x, y, bins=[_HM_XBINS, _HM_YBINS])
    nums,   _, _ = np.histogram2d(x, y, bins=[_HM_XBINS, _HM_YBINS], weights=weights)
    counts_smooth = _gaussian_filter(counts, sigma=sigma)
    nums_smooth   = _gaussian_filter(nums,   sigma=sigma)
    with np.errstate(invalid='ignore', divide='ignore'):
        rate = np.where(counts_smooth > _HM_MIN_COVERAGE, nums_smooth / counts_smooth, np.nan)
    return rate.T

# Heatmap mode config: title, colorscale, plain-English caption, and the
# data-prep functions. Adding a fourth mode = one new dict entry, nothing else.
#   filter_fn  → which subset of pitches to feed (e.g. swings only, BIP only)
#   weights_fn → which value to average per cell, or None for a density count
HEATMAP_MODES = {
    'faced': dict(
        title='Where pitchers attack him',
        colorscale='Hot',
        caption="Density of pitch locations. **Bright = pitchers attack here often.** "
                "Look for clusters at the edges of the zone — that's the pattern they've settled on.",
        filter_fn=lambda p: p,
        weights_fn=None,
    ),
    'whiff': dict(
        title='Whiff rate by location',
        colorscale='Reds',
        caption="On swings, what fraction missed at each spot. **Red = swing-and-miss zone.** "
                "These red blobs are the holes pitchers exploit.",
        filter_fn=lambda p: p[p['description'].isin(_SWING_DESC)],
        weights_fn=lambda d: d['description'].isin(_WHIFF_DESC).astype(float),
    ),
    'damage': dict(
        title='Damage by location',
        colorscale='RdBu_r',
        caption="On contact, average xwOBA at each spot. **Red = he does damage, blue = weak contact.** "
                "Compare this to the whiff map — they're often opposite sides of the plate.",
        filter_fn=lambda p: p[p['estimated_woba_using_speedangle'].notna()],
        weights_fn=lambda d: d['estimated_woba_using_speedangle'],
    ),
}

def zone_heatmap(pitches, mode):
    """Smooth catcher's-perspective heatmap. mode = 'faced' | 'whiff' | 'damage'."""
    cfg = HEATMAP_MODES[mode]
    p = pitches.dropna(subset=['plate_x', 'plate_z'])
    p = p[p['plate_x'].between(-2, 2) & p['plate_z'].between(0, 5)]
    if len(p) < 10:
        return None
    d = cfg['filter_fn'](p)
    if len(d) < 10:
        return None
    if cfg['weights_fn'] is None:
        z = _smooth_density(d['plate_x'], d['plate_z'])
    else:
        z = _smooth_rate(d['plate_x'], d['plate_z'], weights=cfg['weights_fn'](d))

    fig = go.Figure(go.Heatmap(
        z=z, x=_HM_XCENTERS, y=_HM_YCENTERS,
        colorscale=cfg['colorscale'], showscale=False,
        zsmooth='best',   # plotly's bilinear interp on top of the gaussian blur
        hoverinfo='skip',
    ))
    # Strike zone reference box (~17 in plate width, ~1.5–3.5 ft vertical).
    fig.add_shape(type='rect', x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                  line=dict(color='white', width=2), fillcolor='rgba(0,0,0,0)')
    fig.update_layout(
        title=dict(text=cfg['title'], font=dict(size=12, color='#ccc'), x=0.0),
        height=300, margin=dict(t=34, b=10, l=10, r=10),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        xaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[0.5, 4.5], showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# ── pybaseball built-in spraychart (matplotlib, on a real ballpark outline) ──
# Statcast team code → pybaseball stadium name (from mlbstadiums.csv). pybaseball
# still uses pre-2022 names: Cleveland is 'indians', Athletics is 'athletics'.
STATCAST_TO_STADIUM = {
    'AZ': 'diamondbacks', 'ATL': 'braves', 'BAL': 'orioles', 'BOS': 'red_sox',
    'CHC': 'cubs', 'CWS': 'white_sox', 'CIN': 'reds', 'CLE': 'indians',
    'COL': 'rockies', 'DET': 'tigers', 'HOU': 'astros', 'KC': 'royals',
    'LAA': 'angels', 'LAD': 'dodgers', 'MIA': 'marlins', 'MIL': 'brewers',
    'MIN': 'twins', 'NYM': 'mets', 'NYY': 'yankees', 'OAK': 'athletics',
    'ATH': 'athletics', 'PHI': 'phillies', 'PIT': 'pirates', 'SD': 'padres',
    'SEA': 'mariners', 'SF': 'giants', 'STL': 'cardinals', 'TB': 'rays',
    'TEX': 'rangers', 'TOR': 'blue_jays', 'WSH': 'nationals',
}

# Vivid, high-contrast outcome colors for the spray chart (pop on the dark bg).
# Keys are the title-cased `events` labels pybaseball assigns to each scatter.
# Anything not listed (the various out types) falls back to dim grey.
BRIGHT_SPRAY_COLORS = {
    'Home Run': '#ff2d55',   # vivid red
    'Triple':   '#00e676',   # vivid green
    'Double':   '#2979ff',   # vivid blue
    'Single':   '#ffd60a',   # vivid amber
}
_SPRAY_OUT_COLOR = '#5b6473'

def pybaseball_spray_fig(player_pitches, stadium='generic'):
    """Wrap pybaseball's built-in spraychart (matplotlib) for Streamlit.

    Returns a matplotlib Figure styled to the dashboard dark theme with vivid
    outcome colors, or None if there's nothing to plot, or the Exception if
    pybaseball/matplotlib misbehaves (caller shows a notice instead of crashing).
    We import inside the function so a missing/old pybaseball can't break the page.
    """
    bb = player_pitches.dropna(subset=['hc_x', 'hc_y'])
    bb = bb[bb['events'].notna()]
    if len(bb) < 3:
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')           # headless backend for Streamlit
        from pybaseball import spraychart
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # silence the reindex / plt.show noise
            ax = spraychart(bb, stadium, size=60)
        fig = ax.get_figure()

        # Recolor each outcome scatter with the vivid palette (hits) / grey (outs),
        # then rebuild the legend so its swatches match the new colors.
        for coll in ax.collections:
            label = coll.get_label()
            coll.set_color(BRIGHT_SPRAY_COLORS.get(label, _SPRAY_OUT_COLOR))
            coll.set_alpha(0.9)
        old_leg = ax.get_legend()
        leg_title = old_leg.get_title().get_text() if old_leg else 'Outcome'
        if old_leg:
            old_leg.remove()
        ax.legend(title=leg_title, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, framealpha=0)

        # Dark-theme the figure + the freshly-built legend.
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.title.set_color('#ddd')
        leg = ax.get_legend()
        if leg:
            leg.get_title().set_color('#ddd')
            for txt in leg.get_texts():
                txt.set_color('#ddd')
        return fig
    except Exception as exc:
        return exc

# Pitch family display colors. Single source of truth — any chart that needs
# to draw fastball/breaking/offspeed should pull from here.
PITCH_FAMILY_COLORS = {
    'Fastball': '#ef4444',
    'Breaking': '#3b82f6',
    'Offspeed': '#22c55e',
}

def pitch_mix_trend_chart(player_id, pitchmix_df):
    """% of each pitch family the batter SAW, per season — 'are pitchers adjusting?'"""
    d = pitchmix_df[pitchmix_df['batter'] == player_id].sort_values('season')
    if len(d) == 0:
        return None
    fig = go.Figure()
    for label, color in PITCH_FAMILY_COLORS.items():
        col = f'{label}_pct'
        if col in d.columns:
            fig.add_trace(go.Scatter(
                x=d['season'].astype(int), y=(d[col] * 100), mode='lines+markers+text',
                name=label, line=dict(color=color, width=3), marker=dict(size=9),
                text=[f"{v*100:.0f}%" for v in d[col]], textposition='top center',
                textfont=dict(size=10, color=color)))
    fig.update_layout(
        title=dict(text='Pitch mix FACED by season (are pitchers changing approach?)',
                   font=dict(size=12, color='#ccc'), x=0.0),
        height=320, margin=dict(t=40, b=30, l=40, r=20),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        xaxis=dict(title='Season', tickmode='array',
                   tickvals=sorted(d['season'].astype(int).unique()),
                   color='#ddd', gridcolor='#222'),
        yaxis=dict(title='% of pitches', color='#ddd', gridcolor='#222'),
        legend=dict(font=dict(color='#ddd'), orientation='h', y=-0.2),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — PLAYER DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
def render_player_detail(df, history_df, league_refs=None, rolling_df=None,
                         pitchmix_df=None):
    st.title("👤 Player Detail")
    st.markdown("""
Pick any qualified 2026 hitter below. This page answers one question: **"Is his hot start (or slump) real, or will it fade?"**

#### How to read the page
- **Header card (top right):** the player's **archetype** — *who he truly is* (Talent Tier: Elite / Above-Avg / Average / Below-Avg) and *what's happening right now* (a magnitude like **+14% better than projected level** plus a **Skill vs Luck** split that estimates how much of the move is real). Flag chips (🚀 Breakout · 💎 Buy-Low · 🔋 Power Outage · ⚡ K% Spiking · etc.) call out the obvious story.
- **Rolling xwOBA chart (under the portrait):** the within-season trajectory vs his projected baseline. Green when he's currently above, red when below.
- **Career Hitting Summary:** year-by-year traditional stats with **OPS+ / wOBA+ / xwOBA+** (league-normalized, 100 = avg) so seasons are comparable across different run environments.
- **Skill Profile:** pick a season — the **radar** shows year-over-year skill shape; the **percentile bars** show where he ranks vs the league (red = elite, blue = poor); the **peripheral spider** shows whether the underlying skills back the current trend.
- **Stat Trend Across Seasons:** any single stat plotted 2023→2026 with the league line.
- **🔬 Visual Diagnosis (the "why"):** zone **heatmaps** (where pitchers attack, where he whiffs, where he does damage), a **spray chart** (every batted ball, grey = out — hard outs in the outfield = bad luck), and **pitch-mix faced** by season (are pitchers throwing him more breaking now?).

**Hover** any stat, bar, or chart for a definition and benchmarks.
    """)

    names = df.sort_values('name')['name'].dropna().tolist()
    selected_name = st.selectbox(
        "Pick a player (type to search)",
        names,
        index=names.index('aaron judge') if 'aaron judge' in names else 0,
    )
    player = df[df['name'] == selected_name].iloc[0]
    info = get_player_info(player['batter'])

    render_header_row(player, info, history_df, rolling_df)
    # (Surface-stat flag chips render INSIDE the sustain card; the rolling xwOBA
    # trend renders inside the header's left column, directly under the portrait.)
    st.divider()

    st.subheader("Hitting Summary (last 4 seasons)")
    st.caption("Year-by-year traditional stats from 2023 to 2026.")
    render_season_table(player['batter'], history_df)

    st.subheader("Skill Profile")

    # ─── Season selector ────────────────────────────────────────────────
    # The percentile bars and the pitch-type chart are SEASON-SPECIFIC. We let
    # the user pick which season to view. Default = 2026 (the live season).
    # Only seasons this player actually appeared in are offered.
    player_seasons = sorted(
        history_df[history_df['batter'] == player['batter']]['season'].astype(int).unique(),
        reverse=True,   # most recent first
    )
    default_idx = player_seasons.index(2026) if 2026 in player_seasons else 0
    sel_season = st.selectbox(
        "Season to view (controls the percentile bars + pitch-type chart)",
        player_seasons,
        index=default_idx,
    )

    # Pull this player's row for the selected season, and the whole league's
    # rows for that season (used as the percentile/ranking pool).
    season_df = history_df[history_df['season'] == sel_season]
    player_season_match = season_df[season_df['batter'] == player['batter']]
    player_season_row = player_season_match.iloc[0] if len(player_season_match) else None

    col_radar, col_bars = st.columns([1, 1])
    with col_radar:
        st.caption("**Radar / breakdown** — choose a view from the tabs.")
        tab_multi, tab_vs, tab_spider, tab_hit, tab_pitch = st.tabs([
            "📅 Year-by-Year", "📊 vs Baseline", "🕸️ Peripheral",
            "🎾 Hit-Type Mix", "⚾ vs Pitch Type",
        ])
        with tab_multi:
            # Inherently multi-season — ignores the season dropdown by design.
            st.plotly_chart(radar_multi_season(player['batter'], history_df), use_container_width=True)
        with tab_vs:
            # Baseline only exists for 2026 (it's the projection target), so this
            # view is 2026-only regardless of the dropdown.
            st.caption("Baseline is built for 2026 — this view always compares the 2026 season vs that baseline.")
            st.plotly_chart(radar_current_vs_baseline(player, df), use_container_width=True)
        with tab_spider:
            # Deviation/direction are 2026 concepts → spider is 2026-only.
            spider = peripheral_spider(player)
            if spider is None:
                st.info("This player is performing at expected level (no trend) — peripheral support isn't applicable.")
            else:
                st.caption("2026 only. Each axis: **50 = neutral**, **100 = strongly supports** the current trend, **0 = strongly opposes**. Bigger shape = the underlying skills back the hot/cold streak.")
                st.plotly_chart(spider, use_container_width=True)
        with tab_hit:
            # Inherently multi-season (shows every year + league ref row).
            st.caption(
                "**Hit-type breakdown by season** (all years + ⚖️ league avg). "
                "Launch-angle buckets: GB <10°, LD 10–25°, FB 25–50°, PU 50°+. "
                "**Line drives are the best outcome.** Heavy GB% = slap hitter, "
                "heavy FB% = power tilt, rising PU% = swing-decay warning."
            )
            chart = hit_type_chart(player['batter'], history_df, league_refs)
            if chart is None:
                st.info("No hit-type data available for this player yet.")
            else:
                st.plotly_chart(chart, use_container_width=True)
        with tab_pitch:
            # Season-specific: uses the dropdown's selected season.
            st.caption(
                f"**{sel_season} — xwOBA on contact vs each pitch family.** "
                "Fastball (FF/SI/FC), Breaking (SL/CU/KC/SV/ST), Offspeed (CH/FS/FO). "
                "Gray line = league average. Bars: 🔴 elite (>.040 above league), "
                "🟠 above avg, ⚪ around avg, 🔵 below, 🟦 poor. PA count = sample size."
            )
            if player_season_row is None:
                st.info(f"No data for this player in {sel_season}.")
            else:
                chart = pitch_type_chart(player_season_row, season_df, league_refs)
                if chart is None:
                    st.info(f"No pitch-type data for this player in {sel_season}.")
                else:
                    st.plotly_chart(chart, use_container_width=True)
    with col_bars:
        st.caption(
            f"**Percentile bars — {sel_season}** (vs that season's qualified hitters). "
            "Red = elite, gray = average, blue = poor. Hover for definitions + benchmarks."
        )
        if player_season_row is None:
            st.info(f"No data for this player in {sel_season}.")
        else:
            render_percentile_section(player_season_row, season_df)

    st.subheader("Stat Trend Across Seasons")
    stat_choice = st.selectbox(
        "Pick a stat to plot over time",
        list(TIME_SERIES_OPTIONS.keys()),
        format_func=lambda k: TIME_SERIES_OPTIONS[k],
        index=0,
    )
    st.plotly_chart(
        time_series_chart(player['batter'], stat_choice, TIME_SERIES_OPTIONS[stat_choice], history_df),
        use_container_width=True,
    )

    # ─── VISUAL DIAGNOSIS — the "why" investigation ─────────────────────
    st.divider()
    st.subheader("🔬 Visual Diagnosis")
    st.markdown(
        "How to read this section: the three **heatmaps** show what's happening "
        "in the strike zone (catcher's view, white box = strike zone). The "
        "**spray chart** shows where every batted ball landed. The **pitch-mix** "
        "chart shows what pitchers are throwing him over time. Together they "
        "answer *why* his line looks the way it does."
    )

    pid = player['batter']

    # Season selector — heatmaps + spray are now available for every year the
    # player appears in (2023-2026). Only that year's slim pitch file is loaded.
    pitch_seasons = sorted(
        history_df[history_df['batter'] == pid]['season'].astype(int).unique(),
        reverse=True,
    )
    vd_default = pitch_seasons.index(2026) if 2026 in pitch_seasons else 0
    vd_season = st.selectbox(
        "Season for heatmaps + spray chart",
        pitch_seasons, index=vd_default, key='vd_season',
    )
    pitches_yr = load_pitches(vd_season, _file_mtime(pitches_path(vd_season)))
    player_pitches = pitches_yr[pitches_yr['batter'] == pid] if len(pitches_yr) else None

    if player_pitches is not None and len(player_pitches):
        # Heatmap row: same render pattern for each mode, driven by HEATMAP_MODES.
        for col, mode in zip(st.columns(len(HEATMAP_MODES)), HEATMAP_MODES):
            with col:
                f = zone_heatmap(player_pitches, mode)
                if f:
                    st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False})
                    st.caption(f"**{vd_season}** — " + HEATMAP_MODES[mode]['caption'])

        sp_col, pm_col = st.columns(2)
        with sp_col:
            # Spray on the player's home ballpark THAT season (from history),
            # falling back to the comparison team, then generic.
            hist_row = history_df[(history_df['batter'] == pid) &
                                  (history_df['season'] == vd_season)]
            season_team = hist_row['team'].iloc[0] if len(hist_row) and 'team' in hist_row.columns else None
            stadium = STATCAST_TO_STADIUM.get(season_team,
                        STATCAST_TO_STADIUM.get(player.get('team'), 'generic'))
            fig = pybaseball_spray_fig(player_pitches, stadium)
            if fig is None:
                st.info("Not enough batted balls to plot.")
            elif isinstance(fig, Exception):
                st.info(f"pybaseball spraychart unavailable: {fig}")
            else:
                st.pyplot(fig, clear_figure=True)
                st.caption(f"**{vd_season}** batted balls on the **{stadium.replace('_', ' ').title()}** outline. "
                           "🔴 HR · 🔵 Double · 🟢 Triple · 🟡 Single · ⚪ Out — "
                           "**hard-hit balls landing as outs deep in the field = bad luck.**")
        with pm_col:
            if pitchmix_df is not None:
                pm = pitch_mix_trend_chart(pid, pitchmix_df)
                if pm:
                    st.plotly_chart(pm, use_container_width=True, config={'displayModeBar': False})
                    st.caption("Share of pitches he saw each season, by family. "
                               "**Watch for big year-over-year shifts** — when pitchers go from 28% to "
                               "42% breaking balls, that's an adjustment that often drives a slump.")
    else:
        st.info(f"No pitch-level data available for this player in {vd_season}.")

    with st.expander("📖 Glossary — what every stat means and what counts as good"):
        render_glossary(league_refs)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR + DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — BEHIND THE SCENES (full methodology / math / workflow)
# ═══════════════════════════════════════════════════════════════════════════════
def render_methodology():
    st.title("🧠 Behind the Scenes — Full Methodology")
    st.caption("Everything the model does, why it does it that way, and the math behind every number on every other page. "
               "Click any section to expand.")

    st.markdown("---")
    st.markdown("### 📑 Table of Contents")
    st.markdown(
        "1. Model philosophy — why we built it this way  \n"
        "2. Architecture & data flow  \n"
        "3. Season-level aggregation  \n"
        "4. Marcel baseline math  \n"
        "5. Park factor neutralization (wOBA only)  \n"
        "6. Aging curves  \n"
        "7. Reliability & the Bayesian framework  \n"
        "8. Direction & deviation tolerance  \n"
        "9. Peripheral support (soft-cap tanh math)  \n"
        "10. Sustain score formula  \n"
        "11. Z-score engine & luck probability  \n"
        "12. Performance tier (bucketed |z|)  \n"
        "13. Talent tier (percentile of baseline)  \n"
        "14. Skill-adjusted luck (speed/contact fix)  \n"
        "15. Surface-stat flags  \n"
        "16. Rolling momentum & form arrow  \n"
        "17. Trajectory tags (State × Form)  \n"
        "18. \"+\" stats (xwOBA+ / wOBA+ / OPS+)  \n"
        "19. xwOBA-beater + contact foundation  \n"
        "20. Forward (rest-of-season) projection  \n"
        "21. Heatmap smoothing (Gaussian KDE)  \n"
        "22. Spray chart (pybaseball + real ballpark)  \n"
        "23. Pitch-mix-faced signal  \n"
        "24. Column glossary  \n"
        "25. Known limitations & honest caveats"
    )
    st.markdown("---")

    # ── 1. Philosophy ──────────────────────────────────────────────────────
    with st.expander("1. Model philosophy — why we built it this way", expanded=False):
        st.markdown("""
**The core question:** when a player has a hot or cold start, is it **real** (a true talent change) or **noise** (variance that will regress)?

Every projection system in baseball answers some version of this. Our angle:

- **A single composite score is lossy.** Talent, current performance, luck, sample size, and momentum all matter — collapsing them into one number erases information. Early in this project we built a single 0-100 "sustain score" and discovered it piled players at 50, hid important nuance, and felt arbitrary.
- **Multi-axis classification is honest.** Instead, we present **two orthogonal axes**: TALENT TIER (who he truly is, baseline-based) × PERFORMANCE TIER (how far his current level is from expected, z-score-based), then layer momentum, luck %, surface flags, and trajectory on top.
- **Every dimension uses public data only.** Statcast (via pybaseball) + MLB Stats API. No proprietary scouting grades, no batted-ball direction adjustments Savant keeps internal. If we can't compute it from open data, we don't claim it.
- **Casual-readable.** A number like "**+14% better than projected level**" with a "**60% skill / 40% luck**" split bar should be readable to any fan, while the underlying machinery (Marcel baseline, z-score, Bayesian projection) is fully exposed in this page for those who want it.

**The two-question hierarchy:**

1. *Is the deviation statistically real?* → z-score → p_luck → performance tier
2. *Do the underlying skills back it up?* → peripheral support → skill-support score
        """)

    # ── 2. Architecture ────────────────────────────────────────────────────
    with st.expander("2. Architecture & data flow"):
        st.markdown("""
Two layers, separated by parquet files:

**Pipeline (`data_pull_baseball.py`)** — runs once when data updates. Does all the math:
1. Pulls raw Statcast pitch-by-pitch data 2023-2026 via `pybaseball.statcast`
2. Aggregates per-season per-batter stats (`build_season_stats`)
3. Builds Marcel baselines, applies aging
4. Computes deviations, peripheral support, sustain score
5. Runs the z-score engine for p_luck and performance tier
6. Classifies talent tier, current state, archetype
7. Computes momentum, trajectory tag
8. Flags surface-stat events (power outage, K spike, buy-low, etc.)
9. Saves multiple parquets the dashboard reads.

**Dashboard (`mlb_model_sustain.py`)** — pure view layer, no math. Reads parquets cached by file modification time so regenerating the pipeline auto-refreshes the UI.

**Parquets emitted:**

| File | Contents | Used by |
|---|---|---|
| `season_stats_{2023,…,2026}.parquet` | Per-batter per-season aggregate stats | (internal to pipeline) |
| `player_history.parquet` | All four seasons stacked | Career table, radar, time-series, "+" stats |
| `comparison.parquet` | 2026 row per player + baselines + deviations + scores + flags | Everything in the dashboard |
| `rolling_xwoba_2026.parquet` | Trailing 50-BBE xwOBA per player per game-date | Rolling chart under portrait |
| `pitches_2026.parquet` | Slim per-pitch 2026 data (plate_x/z, hc_x/y, description, events, …) | Heatmaps + spray chart |
| `pitch_mix_by_season.parquet` | % of pitches a batter saw, by family, by year | Pitch-mix trend chart + narrative |
        """)

    # ── 3. Season-level aggregation ────────────────────────────────────────
    with st.expander("3. Season-level aggregation"):
        st.markdown("""
For each season, `build_season_stats` does:

**Filter PA-ending rows in the regular season:**
```
rs = raw_data[raw_data['game_type'] == 'R']
at_bats = rs[rs['events'].notna()]
```

`game_type == 'R'` excludes spring training (`S`), All-Star (`A`), and postseason. `events.notna()` keeps only plate-appearance-ending pitches.

**Tag each PA's event:** single / double / triple / HR / walk / IBB / HBP / SF / SAC / K / batted ball / barrel (`launch_speed_angle == 6`) / hard-hit (`launch_speed ≥ 95`).

**Park factor per PA:** maps `home_team` to a Baseball Savant 2024-26 rolling park factor (COL=112 inflates 12%, TEX=92 suppresses 8%, etc.). Saved as a column so we can take a **PA-weighted mean** per batter, giving each player's "effective park factor" across the venues he actually hit in.

**Derive each batter's primary team:** Statcast doesn't store batter team directly, so we derive it via `inning_topbot`:
```
Top of inning → away team is batting → batter_team = away_team
Bot of inning → home team is batting → batter_team = home_team
```
Then take the mode (most common team) per batter as their primary season team.

**Pitch-level chase % and whiff %** (a separate pass on ALL pitches, not just PA-ending):
- `whiff_rate` = swinging strikes / total swings
- `chase_rate` = out-of-zone swings / out-of-zone pitches

**Competitive-swing bat speed:** Statcast tracks `bat_speed` on every swing. Simple mean over PA-ending rows would bias toward final-pitch defensive swings. Instead: aggregate over **all** swings, drop bunts, then within each player keep only swings ≥ their personal 10th-percentile bat speed (Savant's "competitive swing" filter). Brings our numbers within ±0.2 mph of Savant.

**Hit-type from launch angle:**
```
GB: launch_angle < 10°
LD: 10° ≤ launch_angle < 25°
FB: 25° ≤ launch_angle < 50°
PU: launch_angle ≥ 50°
```

**Pitch-type performance:** xwOBA on contact vs each pitch family (Fastball / Breaking / Offspeed), pivoted wide as `xwOBA_vs_Fastball`, `PAs_vs_Fastball`, etc.

**Two-tier PA filter (Option C):**
- `min_pa=100` for inclusion in parquet (so Hyeseong Kim, injured Trout 2024 @ 152 PA still appear)
- `qualified=1` flag for PA ≥ 300 (the "league reference pool" used for league means and percentile rankings)
- 2026 forces `qualified=1` for everyone in the parquet, since mid-season nobody has 300 PA yet

**Derived rates:** BA, OBP, SLG, K%, BB%, HR_rate, barrel_rate, hard_hit_rate, xBA, xSLG.

**wOBA via FanGraphs linear weights:**
```
wOBA_raw = (0.689·uBB + 0.720·HBP + 0.882·1B + 1.254·2B + 1.586·3B + 2.050·HR)
           / (AB + uBB + SF + HBP)
```
Then park-neutralized: `wOBA = wOBA_raw / (weighted_pf / 100)`.

**xwOBA** uses the speedangle estimate directly: `mean(estimated_woba_using_speedangle)` — already park + sprint-speed adjusted by Savant at the source.

**BABIP:** `(H − HR) / (AB − K − HR + SF)`.

**Luck signal:** `xwOBA_diff = xwOBA − wOBA` (positive = unlucky on results).
        """)

    # ── 4. Marcel baseline ─────────────────────────────────────────────────
    with st.expander("4. Marcel baseline math"):
        st.markdown("""
Tom Tango's "Marcel the Monkey" projection — three ingredients:
1. Weighted average of historical seasons
2. Regression toward league mean
3. Age adjustment

For a player's history seasons 2023, 2024, 2025 with a given stat S:

**Step 1 — recency × PA combined weight per row:**
```
combined_w(y) = recency_w(y) × PA(y)
```
with recency weights `{2025: 5, 2024: 3, 2023: 2}` (steeper recency tilt than classic Marcel's 5/4/3).

A 2025 600-PA season contributes weight 5·600 = 3000; the same 600 PA in 2023 contributes only 2·600 = 1200. PA-weighting *and* recency-weighting in one shot.

**Step 2 — raw weighted average:**
```
raw_avg = Σ_y (combined_w(y) · stat(y)) / Σ_y combined_w(y)
```

**Step 3 — effective PA:** normalized by the max recency weight so a player with full 2025 data gets credit for their actual PA, not 5× it:
```
effective_PA = Σ_y combined_w(y) / max(recency_w) = Σ_y combined_w(y) / 5
```

**Step 4 — regression to league mean:**
```
baseline = (effective_PA · raw_avg + R · lg_mean) / (effective_PA + R)
```
Equivalently, this is **Bayesian posterior with a normal prior on the league mean weighted by R "ghost PAs"** — a player with effective_PA = R gets pulled exactly halfway to the league average.

**Regression amounts per stat** (Russell Carleton / FanGraphs stabilization):

| Stat | R | rationale |
|---|---|---|
| K_rate | 60 | Stabilizes fast — bat-to-ball is repeatable |
| BB_rate | 120 | Plate discipline ≈ as fast |
| HR_rate | 170 | Power takes longer |
| barrel_rate | 100 | Stabilizes ~50 BBE; we proxy in PA terms |
| xBA | 600 | BABIP-like variance |
| xwOBA | 500 | The umbrella stat |
| xwOBA_diff | 400 | Persistent skill gap (speed/contact players carry one) |

**League means** are computed PA-weighted over the QUALIFIED (300+ PA) pool in 2023-2025, NOT the inclusive pool. So "league average" anchors on full-time hitters even though baselines themselves draw from partial-season players too.

**Reliability** (used downstream for the noise band and projection blend):
```
reliability = effective_PA / (effective_PA + R)
```
0 = no history (rookies, gets league mean) — 1 = many seasons (full player signal).

**Rookies / no-history fill:** baseline = league mean, reliability = 0. The dashboard treats them as "we don't know" instead of dropping them.
        """)

    # ── 5. Park factor ─────────────────────────────────────────────────────
    with st.expander("5. Park factor neutralization (wOBA only — not xwOBA)"):
        st.markdown("""
xwOBA is built from exit velocity and launch angle, which don't change with venue — a 105 mph line drive is a 105 mph line drive in Coors or Oracle. So xwOBA needs no park adjustment.

wOBA, however, is **outcome-based** — that same line drive is a HR at Yankee Stadium and a flyout at Oracle. So we park-neutralize wOBA.

**Method:** for each PA, attach the home venue's park factor (Baseball Savant 2024-26 rolling). Take the **PA-weighted mean per batter** = the "effective park factor" across every venue he hit in.

```
wOBA = wOBA_raw / (effective_PF / 100)
```

**Concrete examples:**

| Player | Effective PF | wOBA_raw | wOBA_neutral |
|---|---|---|---|
| Brenton Doyle (COL) | ~105.5 | .310 | .294 (5% shrink) |
| Julio Rodríguez (SEA) | ~96 | .335 | .349 (4% bump) |
| Aaron Judge (NYY) | ~101 | .476 | .471 (tiny) |

After this fix, **`xwOBA_diff = xwOBA − wOBA`** is a clean luck signal — no longer contaminated by venue. A speed demon at Coors is no longer flagged as "overperforming contact" when it was actually just thin air.

**Honest caveat:** we apply the overall PF to the *whole* wOBA, including its walk/HBP/K portions. Those aren't really park-affected, so we slightly over-adjust ~10% of wOBA. Worst-case imprecision: ~0.002 of wOBA at the extremes (COL, TEX). Below the noise floor of everything else.
        """)

    # ── 6. Aging ───────────────────────────────────────────────────────────
    with st.expander("6. Aging curves"):
        st.markdown("""
Different stats age differently. Applying one multiplier to everything is wrong — K% actually goes **UP** with age (slower bat → more whiffs), so multiplying it by 0.95 for a 35-year-old predicts fewer Ks, which is backwards.

We define three aging "shapes":

**Offense** (xwOBA, xBA, HR_rate, barrel_rate — anything power/contact-quality):

| Age | Multiplier |
|---|---|
| < 25 | 1.02 |
| 25-26 | 1.01 |
| 27-29 | 1.00 (peak) |
| 30-31 | 0.99 |
| 32-33 | 0.97 |
| 34-35 | 0.95 |
| 36+ | 0.92 |

**Strikeout** (K_rate) — mirror of offense:

| Age | Multiplier |
|---|---|
| < 25 | 0.98 |
| 25-26 | 0.99 |
| 27-29 | 1.00 |
| 30-31 | 1.01 |
| 32-33 | 1.03 |
| 34-35 | 1.05 |
| 36+ | 1.08 |

**Discipline** (BB_rate, xwOBA_diff): flat 1.0 across all ages.

**Why xwOBA_diff is flat:** it's a signed difference stat. Multiplying a negative gap by 0.95 makes it less negative — the wrong direction.

We apply the factor to the **regressed baseline** so the regression target stays clean. The unaged version is kept as `baseline_<stat>_raw` for inspection.

**Magnitude:** factors are ±2-8%, gentle on purpose. Marcel-style projections shouldn't bet hard on age alone — there's too much player-to-player variance.
        """)

    # ── 7. Reliability & Bayesian framework ────────────────────────────────
    with st.expander("7. Reliability & the Bayesian framework"):
        st.markdown("""
The whole model is **Bayesian regression to the mean**, applied at multiple stages.

**Bayes in one sentence:**
```
estimate = (evidence × weight_evidence + prior × weight_prior)
         / (weight_evidence + weight_prior)
```
The weight is "how many plate appearances of trust" each side gets.

**Application 1 — building the baseline:**
- Evidence: the player's 2023-25 weighted history
- Prior: the league mean
- Weights: effective_PA vs R (regression PA)

→ output: `baseline_xwOBA`, with `reliability = effective_PA / (effective_PA + R)` = how much of the baseline is THE PLAYER vs the league average.

**Application 2 — projecting forward (rest of season):**
- Evidence: the player's 2026 current performance
- Prior: the player's aged baseline
- Weights: 2026 PA vs R

```
current_reliability = PA_2026 / (PA_2026 + R)
projected_ros = current_reliability · current + (1 − current_reliability) · baseline
```

A 235-PA Trout at xwOBA .423 with baseline .349:
- current_reliability = 235 / (235 + 500) = 0.32
- projected = 0.32·.423 + 0.68·.349 = **.373**

That .373 is our honest RoS estimate — not the season number (.423, with too much noise still) and not the baseline (.349, ignoring 2026 evidence).

**Combined signal** (used for the score's damping factor):
```
signal = baseline_reliability × current_reliability
```
- High signal (~0.4+) = long history AND lots of 2026 PA → trust the deviation
- Low signal (<0.10) = rookie OR tiny 2026 sample → ignore the deviation, damp to neutral

The sustain score uses signal_weight (signal scaled to saturate at 0.20) to damp the peripheral effect.
        """)

    # ── 8. Direction & deviation tolerance ─────────────────────────────────
    with st.expander("8. Direction & deviation tolerance"):
        st.markdown("""
`direction` ∈ {+1, 0, −1} based on xwOBA deviation:
```
direction = +1   if xwOBA_deviation > +0.010
direction = −1   if xwOBA_deviation < −0.010
direction =  0   otherwise (STABLE)
```

**Why ±0.010?** Even a true .320-xwOBA player has season-to-season natural variance of ~5-8 xwOBA points. Without a tolerance, we'd label half the league HOT or COLD on noise. ±.010 carves out a "performing at level" neutral zone.

`direction` drives:
- which peripheral movements count as "supportive" (K% down is supportive of hot, supportive of cold flips the direction)
- which verdict-tier wording applies (HOT/COLD/STABLE)
- whether momentum is interpreted (STABLE players don't get a "Fading" trajectory tag)
        """)

    # ── 9. Peripheral support ──────────────────────────────────────────────
    with st.expander("9. Peripheral support — the soft-cap math"):
        st.markdown("""
This is the "do the underlying skills back the trend?" question. We measure four peripherals:
- K_rate (negated — K% going DOWN supports offense going UP)
- BB_rate
- barrel_rate
- skill_adjusted_luck (= xwOBA_diff − baseline_xwOBA_diff)

**For each peripheral, the support score:**

```
z = peripheral_deviation / sigma                 # normalize to typical move size
support = 2 · tanh(z / 2) × direction            # soft-cap to [−2, +2], align to trend
```

**Per-stat sigmas** (typical season-to-season magnitude):

| Peripheral | sigma |
|---|---|
| K_rate | 0.040 (4 pp) |
| BB_rate | 0.020 (2 pp) |
| barrel_rate | 0.040 (4 pp) |
| xwOBA_diff (luck gap) | 0.020 (20 points) |

**Why `2·tanh(z/2)` instead of `clip(z, -1, 1)`?**

Hard-clipping treated "1σ supportive" and "3σ supportive" as identical — Devers' BB% collapse (z = −3.3) registered the same as a player whose BB% dropped a typical 1σ. We lost important magnitude information.

`tanh` is the smooth sigmoid alternative. Properties:

| z (sigmas) | hard clip | 2·tanh(z/2) |
|---|---|---|
| 1.0 | 1.00 (capped) | 0.92 |
| 2.0 | 1.00 (capped) | 1.52 |
| 3.0 | 1.00 (capped) | 1.81 |
| ∞ | 1.00 | 2.00 (asymptote) |

So a stat that's 3σ extreme contributes ~50% more than a typical 1σ move — but with diminishing returns so a single freakishly large move can't hijack the score.

**Direction multiply:** `support × direction` flips the sign so a positive support always means "supports the current trend" regardless of whether the player is HOT or COLD.

**Final alignment:**
```
peripheral_alignment = mean(K_support, BB_support, barrel_support, contact_support)
```
Lives in [-2, +2]. Positive = peripherals back the trend; negative = peripherals contradict.
        """)

    # ── 10. Sustain score ──────────────────────────────────────────────────
    with st.expander("10. Sustain score formula"):
        st.markdown("""
The 0-100 number now demoted to "Skill-Support" on the card.

```
signal_weight = clip(signal_xwOBA / 0.20, 0, 1)
raw_score = 50 + 30 × peripheral_alignment × signal_weight
sustain_score = clip(raw_score, 0, 100)
                  forced to 50 if direction == 0 (STABLE)
```

**Decomposition:**
- Starts at 50 (no signal → neutral)
- Peripherals can swing it ±60 points (30 × ±2 alignment, soft-cap max)
- `signal_weight` damps the swing when sample/history is too thin
- Clamped to [0, 100]

**Why multiplier 30 and not 40?** The original (hard-clip) formula used 40 × alignment with alignment ∈ [-1, +1] → max swing ±40. After moving to soft-cap (alignment ∈ [-2, +2] now), 40 × 2 would routinely max scores at 100 even for moderate cases. 30 × 2 keeps the score gradient meaningful — Devers-level extreme cases land around 85-90, typical 1σ-supportive players land in the 70s.

**Why a flat 50 for STABLE players?** They have no trend to grade. Forcing 50 honestly reports "no signal." The 30% of players landing at exactly 50 is by design — they're either STABLE or rookies (signal_weight = 0).

**The score's current role:** demoted to a supporting metric called "Skill-Support" on the card. The headline magnitude % (deviation as % of baseline) and skill/luck split bar (from p_luck) are the visual stars. Skill-Support adds the "do peripherals back it" dimension as supporting evidence, not the primary verdict.
        """)

    # ── 11. Z-score engine ────────────────────────────────────────────────
    with st.expander("11. Z-score engine — luck probability math (AP stats!)"):
        st.markdown("""
The classic statistical question: given his sample size, how likely is the observed deviation just chance?

**The noise band.** A rate stat's standard error over N plate appearances:
```
σ_noise(N) = σ_true × √(M / N)
```
where `M` = the stabilization point (= our regression_PA, 500 for xwOBA) and `σ_true` = the spread of TRUE talent across the league.

This formula falls out of the reliability identity:
```
reliability(N) = N / (N + M) = σ_true² / σ_obs²(N)
```

**Correcting σ_true from σ_obs.** We can only directly observe `σ_obs(N_qual)` = the standard deviation of xwOBA across qualified hitters. But even a 600-PA sample carries noise, so σ_obs overstates σ_true. Correction:
```
σ_true² = σ_obs² × N_qual / (N_qual + M)
σ_true = σ_obs × √(N_qual / (N_qual + M))
```

For our 2024-25 qualified pool: σ_obs ≈ .033, N_qual avg ≈ 600, M = 500 → **σ_true ≈ 0.0244**.

**The z-score:**
```
z = (current_xwOBA − baseline_xwOBA) / σ_noise(N)
```

For a 200-PA player: σ_noise = 0.0244 · √(500/200) = 0.0386. A +0.040 deviation gives z = 1.04.

**The luck probability** (two-sided p-value):
```
p_luck = 2 · (1 − Φ(|z|)) = 1 − erf(|z| / √2)
```
where Φ is the standard normal CDF. The math identity `2(1 − Φ(|z|)) = 1 − erf(|z|/√2)` lets us implement with Python's stdlib `math.erf` (no scipy needed).

**Interpretation:** p_luck is "the probability of observing a deviation this extreme by pure chance if the player's true talent equals his baseline." Low = the hot/cold is almost certainly real. High = it's within the noise.

**Concrete worked examples:**

| Player situation | σ_noise | z | p_luck | Read |
|---|---|---|---|---|
| +.040 at 200 PA | .039 | 1.04 | **30%** | mostly noise |
| +.080 at 200 PA | .039 | 2.05 | **4%** | almost certainly real |
| +.040 at 600 PA | .022 | 1.79 | **7%** | leaning real (more PA = tighter band) |

**On the dashboard, this becomes the SKILL vs LUCK split bar:** the skill share is `100 − p_luck` and the luck share is `p_luck`. Both halves of the same horizontal bar, labeled above. The standard probability-split visual.
        """)

    # ── 12. Performance tier ──────────────────────────────────────────────
    with st.expander("12. Performance tier (bucketed |z|)"):
        st.markdown("""
The z-score doubles as a magnitude classifier, bucketed:

| `abs(z)` | Performance tier | Meaning |
|---|---|---|
| < 1 | As Expected — *"Performing at Projected Level"* | within normal noise |
| 1-2 | Notably Over/Under | leaning real |
| 2-3 | Unusually High/Low | < 5% chance noise |
| ≥ 3 | Unfathomably High/Low | < 0.3% chance noise — historic |

**Same |z|, different headline message:** the card shows these in plain language ("Better Than Expected", "Much Worse Than Expected", "Far Better Than Expected") plus the magnitude % and the skill/luck split bar.

**This is the SECOND axis** in the Talent × Performance Explorer grid on the Leaderboards page — combining "who he is" with "how he's doing right now" gives 5 × 7 = 35 archetype cells.
        """)

    # ── 13. Talent tier ──────────────────────────────────────────────────
    with st.expander("13. Talent tier (percentile of baseline_xwOBA)"):
        st.markdown("""
The first axis: who the player TRULY is, independent of right-now performance. Built from the aged-regressed `baseline_xwOBA` percentile within the non-rookie population.

| Percentile | Tier |
|---|---|
| ≥ 90 | 🌟 Elite |
| 65-90 | ✅ Above-Average |
| 35-65 | ➖ Average |
| < 35 | 🔻 Below-Average |
| (rookie / no history) | 🆕 Unproven |

**Why percentile-based?** Auto-adjusts to the run-scoring environment. If 2026 is a juiced year and league xwOBA rises, percentile cuts stay meaningful — absolute xwOBA cuts (e.g. "Elite if baseline > .370") would silently drift.

**Why exclude rookies from the percentile ranking?** Their baselines are league-mean fillers (no history). Including them would compress the real-player distribution and skew the cuts. Rookies always get tier `Unproven`.
        """)

    # ── 14. Skill-adjusted luck ──────────────────────────────────────────
    with st.expander("14. Skill-adjusted luck (the speed/contact fix)"):
        st.markdown("""
**Problem.** `xwOBA_diff = xwOBA − wOBA` was being treated as pure luck. But speed/contact players (Yelich, Arráez, Chandler Simpson) carry a **persistent** negative gap — their wOBA reliably beats their xwOBA via legs, contact placement, shift-beating.

Treating that persistent skill as luck broke the model in BOTH directions:
- HOT speed demon: persistent negative gap dragged sustain score DOWN → wrongly faded
- COLD speed demon: persistent negative gap supported the cold call → wrongly buried

**Fix.** Build a **personal baseline for `xwOBA_diff`** via the same Marcel machinery (regression_PA = 400 — noisy stat, but heavy regression credits enough persistent skill for established beaters without flagging noise). Then:

```
skill_adjusted_luck = current_xwOBA_diff − baseline_xwOBA_diff
```

What's left is the **abnormal part** — true luck relative to the player's own norm.

A speed demon at his usual −.030 gap → skill_adjusted_luck ≈ 0 → contact_support stays neutral. Only an abnormal gap (vs his own history) counts as luck.

**Worked example — Christian Yelich 2026:**

| Variable | Value |
|---|---|
| xwOBA (2026) | .287 |
| wOBA (2026, park-neutral) | .359 |
| xwOBA_diff (current) | −.072 (HUGE positive luck normally would say) |
| baseline_xwOBA_diff (his persistent norm) | −.011 |
| **skill_adjusted_luck** | **−.061** (the abnormal part) |

So we credit his −.011 persistent skill gap, and only the remaining −.061 enters the luck calculation. That's strictly more correct than treating the whole −.072 as luck.

**Beater flag:** if `baseline_xwOBA_diff < −0.010`, set `xwoba_beater = True`. ~32 players league-wide. These get a special caveat on the card combining their beater status with their `contact_foundation` (whether their xwOBA is eroding/stable/improving vs their own norm).
        """)

    # ── 15. Surface-stat flags ───────────────────────────────────────────
    with st.expander("15. Surface-stat flags"):
        st.markdown("""
xwOBA-centric models miss obvious surface-stat stories (a power hitter with 0 HR is a story even if his xwOBA is fine). These boolean flags surface the obvious; the dashboard renders them as colored chips next to the verdict.

| Flag | Trigger | Catches |
|---|---|---|
| 🔋 Power Outage | HR_rate_deviation < −0.020 AND baseline_HR_rate ≥ 0.030 (was a real power threat) | Tatís 0 HR |
| 💥 Power Surge | HR_rate_deviation > +0.020 | breakout power |
| ⚡ K% Spiking | K_rate_deviation > +0.040 | contact decay |
| ✅ K% Improving | K_rate_deviation < −0.040 | real improvement |
| 💎 Buy-Low (unlucky) | skill_adjusted_luck > +0.030 | Semien, Hayes, Tatís |
| 📉 Sell-High (lucky) | skill_adjusted_luck < −0.030 | regression bait |
| 🚀 Breakout | current_state = Surging AND (sophomore OR experience ≤ 4 yrs) | James Wood, Caminero |

Multiple flags can fire on one player — a Cal Raleigh reads "Worse Than Expected · 🔋 Power Outage · ⚡ K% Spiking · 💎 Buy-Low" all at once.
        """)

    # ── 16. Momentum & form arrow ────────────────────────────────────────
    with st.expander("16. Rolling momentum & form arrow"):
        st.markdown("""
Season xwOBA is a single average that flattens the within-season arc. Two players at .350 on the season can have opposite stories (one climbing, one fading). Momentum captures that.

**Computation:**
```
recent_xwOBA = mean(estimated_woba) over the player's LAST 50 batted balls
momentum     = recent_xwOBA − season_xwOBA
```

**Thresholds:**
- `momentum > +0.020` → 'heating'   → **▲ Trending Up** (green arrow on the card)
- `momentum < −0.020` → 'cooling'   → **▼ Trending Down** (red)
- otherwise           → 'steady'    → **▬ Steady** (gray)

Players with < 60 batted balls get 'steady' (insufficient sample).

**Why 50 BBE?** Sweet spot:
- 25 BBE (1 week) — too noisy
- 100 BBE (4-6 weeks) — too laggy
- 50 BBE (~2-3 weeks) — responsive enough to catch a real shift, stable enough not to whipsaw on a 3-game heater

**Honest caveat:** momentum is the **noisiest signal in the system** and the trajectory tags built on it (Fading, Bottoming Out, etc.) firm up as season sample grows. Soft notes, not hard verdicts.

**The rolling chart under the portrait** shows the FULL rolling series (every game-date), not just the recent vs season number — letting you see the within-season trajectory at a glance. Green line if currently above baseline, red if below; dashed horizontal line at baseline.
        """)

    # ── 17. Trajectory tags ────────────────────────────────────────────
    with st.expander("17. Trajectory tags (State × Form edge cases)"):
        st.markdown("""
Crossing current state with momentum direction surfaces edge cases the snapshot alone misses:

| Current state | Form | Tier modifier | Trajectory tag |
|---|---|---|---|
| Declining / Cooling Off / Unlucky | ▲ heating | tier == Below-Average | **Improving but Capped** |
| Declining / Cooling Off / Unlucky | ▲ heating | other tiers | **Bottoming Out** |
| True to Level / Heating Up | ▲ heating | experience ≤ 5 yrs | **Breakout Building** |
| Hot but Lucky | ▼ cooling | any | **Bubble Bursting** |
| Surging / Heating Up / True to Level | ▼ cooling | any | **Fading** |
| Declining / Cooling Off | ▼ cooling | any | **Free Fall** |

Rendered on the card as an amber/green/red caveat block depending on whether it's a warning or an opportunity.
        """)

    # ── 18. "+" stats ────────────────────────────────────────────────────
    with st.expander('18. "+" stats (xwOBA+ / wOBA+ / OPS+)'):
        st.markdown("""
Raw stats can't be compared across seasons because the run-scoring environment shifts. A "+" stat divides by that season's league average and scales to 100, so each point = 1% above/below league. Comparable across 2023→2026.

```
xwOBA+ = 100 × player_xwOBA / lg_xwOBA
wOBA+  = 100 × player_wOBA  / lg_wOBA              (already park-neutral → park+league adjusted)
OPS+   = 100 × (OBP/lg_OBP + SLG/lg_SLG − 1)       (Baseball-Reference formula)
```

League means are PA-weighted over that season's qualified pool.

**Worked example — Juan Soto:**

| Season | Raw OPS | OPS+ |
|---|---|---|
| 2024 | .987 | 168 |
| 2026 | .986 | 172 |

Nearly identical raw OPS, but OPS+ says **2026 is better** because the run environment is slightly lower — same raw production is worth more relative to league.

(Our OPS+ is league-adjusted only; Baseball-Reference also park-adjusts, so our Judge 2024 OPS+ ≈ 212 vs BBRef's 218 — close, but not identical.)
        """)

    # ── 19. xwOBA-beater + foundation ───────────────────────────────────
    with st.expander("19. xwOBA-beater + contact foundation (the speed/contact 2-part read)"):
        st.markdown("""
Two columns work together for the speed/contact edge case:

**`xwoba_beater`** (boolean): True if `baseline_xwOBA_diff < -0.010` — the player's regressed personal luck gap is persistently negative, i.e., he reliably out-hits his contact via speed/placement/shift-beating.

**`contact_foundation`** (string): how his xwOBA (= his contact quality) is moving vs his own baseline:
- `eroding` if xwOBA_deviation < −0.020 (foundation slipping)
- `improving` if xwOBA_deviation > +0.020
- `stable` otherwise

**The card combines them as a caveat block:**

| beater | foundation | Caveat shown |
|---|---|---|
| True | eroding | ⚠️ *"Beats xwOBA via speed/contact — wOBA propped up, but contact is slipping. Holds up for now; sustainability at risk."* |
| True | stable/improving | 🦵 *"wOBA is skill-driven and should hold even though raw contact looks modest."* |
| False | any | no caveat |

This is the "looks OK now but worrying" case the user asked for. Yelich 2026 is the textbook example: still flagged as a beater, but his contact foundation is eroding — value propped up by skill, but the underlying is cracking.
        """)

    # ── 20. Projection ──────────────────────────────────────────────────
    with st.expander("20. Forward (rest-of-season) projection"):
        st.markdown("""
The rest-of-season xwOBA estimate, Bayesian:
```
current_reliability = PA_2026 / (PA_2026 + R)
projected_ros_xwOBA = current_reliability · current + (1 − current_reliability) · baseline
```
where the baseline is the **aged** baseline.

**Why current_reliability, not signal_weight?** Using `signal_weight = baseline_reliability × current_reliability` would double-discount uncertainty — once via baseline_reliability (which already drove regression INTO the baseline) and again via the blend. That over-pulls projections toward league average, especially for rookies (whose baseline_reliability = 0 would freeze them at league mean no matter how many 2026 PAs they accumulate). `current_reliability` alone is the honest weight for "how much do we trust the 2026 sample on its own."

**Examples (xwOBA, R = 500):**

| Player | 2026 PA | current_rel | composition |
|---|---|---|---|
| Trout | 235 | 0.32 | 32% him, 68% aged baseline |
| Bleday | 101 | 0.17 | 17% him, 83% baseline (heavily regressed) |
| Rookie | 200 | 0.29 | 29% him, 71% league mean (= his "baseline" fill) |

The card shows a "PROJECTED REST-OF-SEASON" table with **current → projected** arrows for AVG, OBP, SLG, OPS, HR%. For BA/OBP/SLG/OPS we project on the fly in the dashboard using each player's career averages from history (PA-weighted across 2023-25) as the prior, blended with their 2026 current value by current_reliability.

**`expected_regression` columns:** `projected − current`. Positive = projection expects improvement (buy-low candidates); negative = expects decline.
        """)

    # ── 21. Heatmap smoothing ──────────────────────────────────────────
    with st.expander("21. Heatmap smoothing (Gaussian KDE)"):
        st.markdown("""
The Savant-style smooth gradient look isn't a mosaic of cells — it's a **density estimate** with continuity between neighbors.

**Algorithm:**
1. Bin pitches on a fine 0.2-ft grid: 16 cells wide (plate_x ∈ [−1.6, 1.6]) × 20 cells tall (plate_z ∈ [0.5, 4.5])
2. Apply a 2D **Gaussian kernel** with σ = 1.4 cells (~0.28 ft) via `scipy.ndimage.gaussian_filter`
3. For **weighted metrics** (whiff rate, damage xwOBA), smooth the numerator AND denominator separately, then divide:

```
smoothed_rate = blur(Σ values) / blur(count)
                masked to NaN where blur(count) < 0.25 (too sparse)
```

This is the correct way to smooth a ratio — smoothing the mean directly would over-amplify cells with thin samples.

**Math.** A 2D Gaussian convolution:
```
smoothed[i,j] = Σ_{di,dj} original[i+di, j+dj] · G(di, dj)
where G(di, dj) = (1 / 2πσ²) · exp(-(di² + dj²) / 2σ²)
```

Think of each pitch as a fuzzy bell-shaped circle of influence rather than a hard cell — the bell extends to neighbors with diminishing weight.

**Coverage masking.** Cells where the smoothed denominator < 0.25 are masked to NaN so we don't paint garbage from one stray sample at the edge of the zone.

**Fallback.** If scipy isn't available, the code falls back to a no-op filter + Plotly's built-in `zsmooth='best'` (bilinear interpolation between cells). Less Gaussian-correct but still smoother than a raw histogram.
        """)

    # ── 22. Spray ──────────────────────────────────────────────────────
    with st.expander("22. Spray chart (pybaseball + real ballpark outline)"):
        st.markdown("""
The spray chart is rendered with **pybaseball's built-in `spraychart()`** (matplotlib) overlaid on the player's actual home-ballpark outline — not a hand-rolled scatter. We landed here after trying a manual plotly version first; the built-in handles the coordinate transform, draws a real stadium shape, and looks far cleaner.

**Why pybaseball's built-in:**
- It ships with stadium outline coordinates (`mlbstadiums.csv`) for all 30 parks plus a `generic` fallback, so balls land on a recognizable field shape.
- It handles the `hc_x` / `hc_y` → field transform internally (the common manual mistake is forgetting to flip the y-axis — `hc_y` increases *downward* in the raw Statcast image grid).
- This version returns a **matplotlib `Axes`** (older pybaseball returned bokeh), so it renders in Streamlit via `st.pyplot()` — no bokeh version headaches.

**The transform it does internally** (for reference — home plate at the image grid center, scaled to feet):
```
field_x =  (hc_x − 125.42)
field_y = −(hc_y − 198.27)     # flip: hc_y grows downward
```

**What we customize after it builds the figure:**
- Map each Statcast `home_team` code → pybaseball stadium name (`STATCAST_TO_STADIUM`); Cleveland is still `indians`, Athletics is `athletics`. Falls back to `generic` if unmapped.
- Plot **all** batted balls with `hc_x`/`hc_y` (not hits-only) so outs are visible — the whole analytic point.
- Recolor each outcome scatter with a vivid palette, then rebuild the legend to match:
  - 🔴 Home Run `#ff2d55`  ·  🔵 Double `#2979ff`  ·  🟢 Triple `#00e676`  ·  🟡 Single `#ffd60a`  ·  ⚪ all outs `#5b6473`
- Dark-theme the figure + legend to match the dashboard.

**The read:** hard-hit balls landing as **grey** dots deep in the outfield = bad luck (well-struck contact that found gloves). The function is wrapped in try/except and imports pybaseball lazily, so if it ever fails the tab shows a notice instead of crashing the page.
        """)

    # ── 23. Pitch-mix shift ────────────────────────────────────────────
    with st.expander("23. Pitch-mix-faced signal"):
        st.markdown("""
"Pitchers adjusted" is a real and underrated cause of slumps. We compute pitch mix faced by season per batter, then surface the shift.

**Pitch family mapping:**

| Family | Statcast codes |
|---|---|
| Fastball | FF (4-seam), SI (sinker), FC (cutter) |
| Breaking | SL, CU, KC, SV, ST, CS |
| Offspeed | CH, FS, FO |

For each player-season:
```
Family_pct = (pitches in family) / (total pitches with a recognized family)
```

The pitch-mix-trend chart on the player detail page plots Fastball% / Breaking% / Offspeed% as three lines over 2023-2026, with point labels — instant "are pitchers throwing him more breaking this year" read.

**Computed shift column** (in `comparison.parquet`):
```
breaking_pct_shift = Breaking_pct_2026 − mean(Breaking_pct over 2023-25)
```
Positive = pitchers throwing him more breaking than his historical norm. Currently used internally for diagnostic narration; could feed a "🐍 Slider Diet" flag in the future.
        """)

    # ── 24. Column glossary ──────────────────────────────────────────
    with st.expander("24. Column glossary — what's in `comparison.parquet`"):
        st.markdown("""
Every column the dashboard reads from `comparison.parquet`. Most have appeared in sections above; this is a single-stop reference.

**Identity:** `batter`, `name`, `name_first`, `name_last`, `key_mlbam`, `mlb_played_first`, `team`, `experience_yrs`, `category` (ROOKIE / SOPHOMORE / COMEBACK / VETERAN / ESTABLISHED)

**Raw 2026 counts & rates:** `PA`, `AB`, `is_hit`, `is_hr`, `is_k`, `is_bb`, `is_ubb`, `is_ibb`, `is_hbp`, `is_sf`, `is_sac`, `is_ci`, `is_batted_ball`, `is_barrel`, `is_hard_hit`, `is_1b`, `is_2b`, `is_3b`, `total_bases`, `BA`, `OBP`, `SLG`, `K_rate`, `BB_rate`, `HR_rate`, `barrel_rate`, `hard_hit_rate`, `xBA`, `xSLG`, `xwOBA`, `wOBA`, `wOBA_raw` (pre-park), `BABIP`

**Statcast advanced:** `avg_exit_velo`, `avg_launch_angle`, `avg_bat_speed` (competitive-swing filtered), `whiff_rate`, `chase_rate`, `weighted_pf`

**Hit-type mix:** `GB_rate`, `LD_rate`, `FB_rate`, `PU_rate`

**Pitch-type performance:** `xwOBA_vs_Fastball`, `xwOBA_vs_Breaking`, `xwOBA_vs_Offspeed`, `PAs_vs_*`

**"+" stats:** `xwOBA_plus`, `wOBA_plus`, `OPS_plus`

**Baselines (aged):** `baseline_xwOBA`, `baseline_xBA`, `baseline_K_rate`, `baseline_BB_rate`, `baseline_HR_rate`, `baseline_barrel_rate`, `baseline_xwOBA_diff` (+ `_raw` variants pre-aging)

**Reliability:** `reliability_<stat>`, `current_reliability_<stat>`, `signal_<stat>`, `weighted_pa_<stat>`, `signal_weight` (rescaled signal_xwOBA)

**Deviations:** `xwOBA_deviation`, `xBA_deviation`, `K_rate_deviation`, `BB_rate_deviation`, `HR_rate_deviation`, `barrel_rate_deviation`

**Luck signals:** `xwOBA_diff` (= xwOBA − wOBA), `skill_adjusted_luck`, `xBA_diff`, `xSLG_diff`

**Score & verdict:** `direction`, `trend`, `peripheral_alignment`, `sustain_score`, `verdict`

**Archetype:** `talent_pctile`, `talent_tier`, `current_state`, `is_breakout`, `archetype` (= "Tier — State")

**Z-score engine:** `sigma_noise_xwOBA`, `z_xwOBA`, `p_luck_xwOBA`, `performance_tier`

**Trajectory:** `recent_xwOBA`, `momentum`, `momentum_label`, `trajectory_tag`

**Flags:** `flag_power_outage`, `flag_power_surge`, `flag_k_spike`, `flag_k_improve`, `flag_buy_low`, `flag_sell_high`, `xwoba_beater`, `contact_foundation`

**Pitch-mix shift:** `breaking_pct_shift`

**Projections:** `projected_ros_<stat>` for every baseline stat, plus `expected_regression_<stat>`

**Other:** `age`, `age_factor_<stat>`, `historical_seasons`, `pa_2025`, `qualified`
        """)

    # ── 25. Limitations ────────────────────────────────────────────────
    with st.expander("25. Known limitations & honest caveats"):
        st.markdown("""
Things the model can't or doesn't do:

**Sample-driven limitations**
- Mid-season trajectory signals (momentum, form arrow, trajectory tags) are the noisiest — they firm up as season sample grows. "Fading" can fire on cold-weather April cooling that isn't real decline.
- Rookies have no baseline → tier `Unproven`, reliability = 0. Deviations are nominally computed against league mean but should be ignored.
- The 50-BBE momentum window misses true very-recent shifts (last 3 games).

**Methodology trade-offs**
- xwOBA_diff personal baseline regression at 400 PA: too high erases the signal (Yelich beater status), too low credits noise (random fluky seasons). 400 is the calibrated sweet spot but it's a judgment call.
- Park factor application to whole wOBA slightly over-adjusts the walk/HBP portions (~0.002 of wOBA imprecision at extremes).
- Aging curves are gentle (±2-8%) and population-average. A specific player whose individual aging arc differs from the league average will be slightly mis-projected.
- Soft-cap tanh in peripheral support: extreme moves (z > 3) still get diminishing returns at asymptote +2. Some real outliers are slightly capped.

**Data gaps**
- Sprint speed (a key driver of beater status) isn't directly modeled — we infer it through the personal xwOBA_diff baseline. So a rookie speed demon won't be flagged as a beater until he accumulates 2-3 seasons.
- bat_speed only exists from 2023 onward in our Statcast pulls. Earlier seasons' bat-speed column is empty.
- No L/R splits in the current model. A player who's a totally different hitter vs LHP could be misclassified.
- No injury adjustments. A guy returning from injury at 60% gets the same baseline as if he were healthy.

**Comparisons to other sources**
- OPS+ doesn't perfectly match Baseball-Reference because BBRef also park-adjusts; we're league-only. Our Judge 2024 OPS+ = 212 vs BBRef 218.
- xwOBA exactly matches Savant (we use their estimated_woba_using_speedangle).
- wOBA matches FanGraphs to ~±.003 after park adjustment.
- Bat speed matches Savant to within ~0.2 mph (the competitive-swing filter pulls us within rounding distance).

**Conceptual edge cases not modeled**
- Players whose true talent is genuinely RISING year-over-year (improving Marcel) — the regression-to-mean assumption baked into the baseline pulls them down toward league. The Breakout flag partially addresses but only for the heating-up state.
- League-wide environmental shifts (juiced ball, rule changes) aren't separately accounted for beyond the per-season league means that drive regression targets.
        """)

st.sidebar.title("⚾ Navigation")
page = st.sidebar.radio(
    "Page",
    ["📊 Leaderboards", "👤 Player Detail", "🧠 Behind the Scenes"],
    label_visibility="collapsed",
)

df = load_comparison(_file_mtime(COMPARISON_PATH))
history_df = load_history(_file_mtime(HISTORY_PATH))
league_refs = compute_league_refs(_file_mtime(HISTORY_PATH))
rolling_df = load_rolling(_file_mtime(ROLLING_PATH))
pitchmix_df = load_pitchmix(_file_mtime(PITCHMIX_PATH))
# pitch-level data is loaded lazily inside the page (per the season dropdown).

if page == "📊 Leaderboards":
    render_leaderboards(df)
elif page == "🧠 Behind the Scenes":
    render_methodology()
else:
    render_player_detail(df, history_df, league_refs, rolling_df, pitchmix_df)
