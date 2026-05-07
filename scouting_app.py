import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from jsonschema.exceptions import best_match

CSV_PATH = 'PERFECT_scouting_data_2026.csv'

st.set_page_config(
    page_title="Player Evaluation Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ─────────────────────────────────────────────────────────────
       DESIGN TOKENS — Football-Manager-Datahub colour & spacing system.
       Every other rule pulls from these, so palette tweaks are one-line.
       ───────────────────────────────────────────────────────────── */
    :root {
        /* Surface levels — darker → lighter as you elevate */
        --bg-deep:        #07090d;
        --bg-canvas:      #0a0e14;
        --bg-surface:     #11161e;
        --bg-elevated:    #181f2a;
        --bg-hover:       #222a37;

        /* Borders */
        --border-faint:   #161b24;
        --border-subtle:  #1f2630;
        --border-default: #2a3140;
        --border-strong:  #3a4252;

        /* Text */
        --text-primary:   #ecf0f5;
        --text-secondary: #9ba3af;
        --text-muted:     #6b7380;
        --text-faint:     #424a55;

        /* Accents */
        --accent-blue:    #5DA5E8;
        --accent-blue-d:  #4a8cd8;
        --accent-cyan:    #4ECDC4;
        --accent-success: #4CAF50;
        --accent-warning: #FFC107;
        --accent-danger:  #FF5252;
        --accent-purple:  #9B7FE8;
        --accent-amber:   #FF9F43;

        /* Tonal radius scale — FM uses small radii, never pill-shaped */
        --radius-xs: 3px;
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
    }

    /* ─────────────────────────────────────────────────────────────
       TYPOGRAPHY
       ───────────────────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        /* Inter stylistic alternates — tighter zero, single-storey a, etc. */
        font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11', 'ss01';
    }

    /* Numerals — tabular figures so columns of stats line up cleanly */
    [data-testid="stDataFrame"], [data-testid="stMetricValue"],
    .stat-bar-fill, [class*="numeric"] {
        font-variant-numeric: tabular-nums;
    }

    /* Headings — FM-style sharp, slightly negative tracking */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1 {
        font-weight: 800 !important;
        font-size: 2rem !important;
        letter-spacing: -0.035em !important;
        margin: 0.25rem 0 0.5rem !important;
    }
    h2 {
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        letter-spacing: -0.025em !important;
    }
    h3 {
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        letter-spacing: -0.015em !important;
    }
    /* h4 = the blue-accented section headers used throughout pages */
    [data-testid="stMarkdownContainer"] h4 {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.005em !important;
        margin: 1.25rem 0 0.5rem !important;
        padding-left: 12px;
        position: relative;
    }
    [data-testid="stMarkdownContainer"] h4::before {
        content: "";
        position: absolute;
        left: 0; top: 7px; bottom: 7px;
        width: 3px;
        background: var(--accent-blue);
        border-radius: 2px;
    }

    /* Captions — secondary explainers everywhere */
    [data-testid="stCaptionContainer"], .stCaption {
        color: var(--text-secondary) !important;
        font-size: 12px !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
    }

    /* ─────────────────────────────────────────────────────────────
       PAGE BACKGROUND — subtle dual-radial wash + linear underlay
       ───────────────────────────────────────────────────────────── */
    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at top right,
                            rgba(93, 165, 232, 0.05), transparent 60%),
            radial-gradient(ellipse 60% 50% at bottom left,
                            rgba(155, 127, 232, 0.04), transparent 60%),
            linear-gradient(180deg, var(--bg-canvas), var(--bg-deep));
        background-attachment: fixed;
    }

    /* ─────────────────────────────────────────────────────────────
       SIDEBAR
       ───────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg,
                            var(--bg-surface) 0%,
                            var(--bg-canvas) 100%) !important;
        border-right: 1px solid var(--border-subtle);
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: var(--text-muted) !important;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }
    /* Sidebar radio nav — hover + selected states */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 4px !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: var(--radius-md);
        padding: 6px 10px !important;
        margin: 0 !important;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: var(--bg-elevated);
        border-color: var(--border-subtle);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
        color: var(--text-secondary);
        font-weight: 500 !important;
    }

    /* ─────────────────────────────────────────────────────────────
       STREAMLIT CHROME — kill defaults
       ───────────────────────────────────────────────────────────── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        height: 0 !important;
    }

    /* ─────────────────────────────────────────────────────────────
       EXPANDERS — info / methodology cards
       ───────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-surface) !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--accent-blue) !important;
    }

    /* ─────────────────────────────────────────────────────────────
       METRICS — head-to-head tally row on Comparison page
       ───────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 14px;
        transition: border-color 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-default);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        font-variant-numeric: tabular-nums;
    }

    /* ─────────────────────────────────────────────────────────────
       TABS — used on Stat Leaderboards
       ───────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none !important;
        color: var(--text-secondary);
        padding: 8px 14px !important;
        font-weight: 500;
        font-size: 13px;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(255,255,255,0.02);
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-blue) !important;
        background: rgba(93, 165, 232, 0.06) !important;
        border-bottom: 2px solid var(--accent-blue) !important;
    }

    /* ─────────────────────────────────────────────────────────────
       SLIDERS — the 0–5 role-fit sliders
       ───────────────────────────────────────────────────────────── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent-blue) !important;
        border: 2px solid var(--bg-canvas) !important;
        box-shadow: 0 0 0 1px var(--accent-blue), 0 2px 6px rgba(0,0,0,0.5) !important;
    }
    .stSlider [data-testid="stSliderTickBarMin"],
    .stSlider [data-testid="stSliderTickBarMax"] {
        color: var(--text-muted) !important;
        font-size: 10px !important;
    }

    /* ─────────────────────────────────────────────────────────────
       SELECTBOXES, MULTISELECT — input chrome
       ───────────────────────────────────────────────────────────── */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] {
        background: var(--bg-surface) !important;
        border-color: var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        transition: border-color 0.15s ease;
    }
    [data-baseweb="select"] > div:hover,
    [data-baseweb="input"]:hover {
        border-color: var(--border-strong) !important;
    }
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"]:focus-within {
        border-color: var(--accent-blue) !important;
    }

    /* ─────────────────────────────────────────────────────────────
       DIVIDERS
       ───────────────────────────────────────────────────────────── */
    hr, [data-testid="stHorizontalBlock"] hr {
        border: none !important;
        border-top: 1px solid var(--border-subtle) !important;
        margin: 1.5rem 0 !important;
    }

    /* ─────────────────────────────────────────────────────────────
       DATAFRAMES — leaderboard tables
       ───────────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        overflow: hidden;
        background: var(--bg-surface);
    }

    /* ─────────────────────────────────────────────────────────────
       INFO / WARNING boxes
       ───────────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-subtle) !important;
        background: var(--bg-surface) !important;
    }

    /* ─────────────────────────────────────────────────────────────
       HOVERABLE STAT LABELS — discoverable tooltip cue
       ───────────────────────────────────────────────────────────── */
    [title]:not([title=""]) {
        cursor: help;
        text-decoration: underline dotted rgba(255,255,255,0.18);
        text-underline-offset: 3px;
        transition: text-decoration-color 0.15s ease;
    }
    [title]:not([title=""]):hover {
        text-decoration-color: var(--accent-blue);
    }

    /* ─────────────────────────────────────────────────────────────
       LEGACY CLASSES — used inside HTML-rendered components.
       Keep names stable; refine the visuals only.
       ───────────────────────────────────────────────────────────── */
    .stat-bar-bg {
        background-color: rgba(255,255,255,0.04);
        border-radius: 3px;
        width: 100%;
        height: 8px;
        margin-bottom: 10px;
    }
    .stat-bar-fill {
        height: 8px;
        border-radius: 3px;
        transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .sub-grade-box {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 6px 4px;
        text-align: center;
        border-bottom: 3px solid var(--accent-success);
        transition: transform 0.15s ease, background 0.15s ease;
    }
    .sub-grade-box:hover {
        background: var(--bg-elevated);
        transform: translateY(-1px);
    }

    /* ─────────────────────────────────────────────────────────────
       BUTTONS
       ───────────────────────────────────────────────────────────── */
    .stButton button {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-primary);
        font-weight: 500;
        font-size: 13px;
        border-radius: var(--radius-md);
        padding: 6px 14px;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        background: var(--bg-hover);
        border-color: var(--border-strong);
    }
    .stButton button:active {
        transform: scale(0.98);
    }

    /* ─────────────────────────────────────────────────────────────
       SIDEBAR WORDMARK CONTAINER — matches the SVG injected below
       ───────────────────────────────────────────────────────────── */
    .ct-wordmark {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 4px 14px;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .ct-wordmark-text {
        display: flex;
        flex-direction: column;
        line-height: 1.05;
    }
    .ct-wordmark-name {
        font-weight: 800;
        font-size: 16px;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    .ct-wordmark-tag {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--accent-blue);
        font-weight: 700;
        margin-top: 2px;
    }

    /* Page-section header style (FM-style microcaps) used as alternative
       to ####. Available via class for one-off polish. */
    .ct-section-head {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: var(--text-secondary);
        margin: 18px 0 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border-subtle);
    }
    </style>
    """, unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_scouts(csv_mtime: float):
    """csv_mtime is unused inside the function — its only job is to feed
    @st.cache_data a fresh cache key whenever the CSV file is modified.
    Without it, edits to the source data (added players, name fixes, etc.)
    would never be reflected because the cached result has no inputs to
    compare against. Pass `os.path.getmtime(CSV_PATH)` at call site."""
    scouts = pd.read_csv(CSV_PATH)
    scouts = scouts[scouts['pos_'].isin(['MF', 'MF,DF'])]
    #scouts = scouts[scouts['pos_'].str.contains('MF', na=False)].copy()
    scouts = scouts[scouts['Playing Time_Min'] >= 900].copy()

    missing_understat = scouts[scouts['xg_under'].isna()]
    missing_sofascore = scouts[scouts['accurateLongBalls'].isna()]

    # 4. Display results
    if missing_understat.empty:
        print("✅ SUCCESS: All midfielders with 900+ mins have Understat data!")
    else:
        print(f"⚠️ Found {len(missing_understat)} players missing Understat data:")
        # Display Player, Team, and Minutes to help you find them on Understat manually if needed
        print(missing_understat[['player', 'team', 'Playing Time_Min']].to_string(index=False))

    if missing_sofascore.empty:
        print("✅ SUCCESS: All players have Sofascore data!")
    else:
        print(f"⚠️ Found {len(missing_sofascore)} players missing Sofascore data:")
        print(missing_sofascore[['player', 'team', 'Playing Time_Min']].to_string(index=False))

    s_90s = scouts['90s_'].replace(0, np.nan)

    # ── DEFENSIVE ─────────────────────────────────────────────────────────────
    scouts['TacklesWon_90'] = scouts['tacklesWon'] / s_90s
    scouts['Tackle_Win_Pct'] = scouts['tacklesWonPercentage']  # already a rate
    scouts['Interception_90'] = scouts['interceptions'] / s_90s
    scouts['Clearance_90'] = scouts['clearances'] / s_90s
    scouts['Aerial_Won_Pct'] = scouts['aerialDuelsWonPercentage']  # already a rate
    scouts['Ground_Duel_Pct'] = scouts['groundDuelsWonPercentage']  # already a rate
    scouts['Recovery_90'] = scouts['ballRecovery'] / s_90s
    scouts['PressWon_90'] = scouts['possessionWonAttThird'] / s_90s # pressing proxy
    scouts['Total_Duel'] = scouts['totalDuelsWon'] / s_90s
    scouts['Block_90'] = scouts['outfielderBlocks'] / s_90s

    # ── PASSING ───────────────────────────────────────────────────────────────
    scouts['Pass%'] = scouts['accuratePassesPercentage']  # already a rate
    scouts['Touch_90'] = scouts['touches'] / s_90s
    scouts['LongBall_90'] = scouts['accurateLongBalls'] / s_90s
    scouts['LongBall_Acc%'] = scouts['accurateLongBallsPercentage']  # already a rate
    scouts['OppHalf_90'] = scouts['accurateOppositionHalfPasses'] / s_90s
    scouts['Cross_90'] = scouts['accurateCrosses'] / s_90s
    scouts['xA_90'] = scouts['xa_under'] / s_90s  # back in passing per design

    # ── INVOLVEMENT ───────────────────────────────────────────────────────────
    # xG_Buildup: touched ball in attacking chain but NOT assister/shooter → pure deep involvement
    # xG_Chain:   involved at ANY point → volume/presence proxy
    # Final_Third: how often you move the ball into dangerous areas
    scouts['xG_Buildup_90'] = scouts['xg_buildup_under'] / s_90s
    scouts['xG_Chain_90'] = scouts['xg_chain_under'] / s_90s
    scouts['Final_Third_90'] = scouts['accurateFinalThirdPasses'] / s_90s

    # ── FINAL PRODUCT ─────────────────────────────────────────────────────────
    # Direct_Creation: Chain − Buildup → player made the decisive action (shot or key pass)
    # xGxA: combined attacking output — the "bottom line" stat
    scouts['Direct_Creation_90'] = (scouts['xg_chain_under'] - scouts['xg_buildup_under']) / s_90s
    scouts['npxGxA_90'] = (scouts['np_xg_under'] + scouts['xa_under']) / s_90s
    scouts['KeyPass_90'] = scouts['keyPasses'] / s_90s
    scouts['BigChance_90'] = scouts['bigChancesCreated'] / s_90s

    # ── DRIBBLING ─────────────────────────────────────────────────────────────
    # Fouls drawn lives here — you draw fouls by carrying into contact, not by defending
    scouts['Dribbles_90'] = scouts['successfulDribbles'] / s_90s
    scouts['Dribble_Succ%'] = scouts['successfulDribblesPercentage']  # already a rate


    # ── SHOOTING ──────────────────────────────────────────────────────────────
    scouts['Shots_90'] = scouts['Standard_Sh'] / s_90s
    scouts['SoT_90'] = scouts['Standard_SoT'] / s_90s
    scouts['SoT%'] = scouts['Standard_SoT%']
    scouts['npxG_90'] = scouts['np_xg_under'] / s_90s
    scouts['G/Sh'] = scouts['Standard_G/Sh']
    scouts['G/SoT'] = scouts['Standard_G/SoT']
    scouts['ShotsInBox_90'] = scouts['shotsFromInsideTheBox'] / s_90s

    # ── EFFICIENCY ────────────────────────────────────────────────────────────
    # Ratio stats — volume cancels out. Rewards doing more with less.
    kp_safe = scouts['keyPasses'].replace(0, np.nan)
    sh_safe = scouts['shots_under'].replace(0, np.nan)
    touch_safe = scouts['touches'].replace(0, np.nan)

    scouts['xA_per_KP'] = (scouts['xa_under'] / kp_safe).fillna(0)
    scouts['npxG_per_Sh'] = (scouts['np_xg_under'] / sh_safe).fillna(0)
    scouts['DC_per_Touch'] = (scouts['Direct_Creation_90'] / touch_safe * s_90s).fillna(0)
    scouts['FT_per_Touch'] = (scouts['accurateFinalThirdPasses'] / touch_safe).fillna(0)

   # ── BALL RETENTION(inverted — lower raw = better percentile) ────────────
    touch_safe = scouts['touches'].replace(0, np.nan)

    # Dispossessed per 100 touches — normalised by involvement
    # A Regista touching 90 balls will naturally lose it more in raw numbers
    # This ratio makes it fair across roles
    scouts['Disp_per100T'] = (scouts['dispossessed'] / touch_safe * 100).fillna(0)

    # Possession lost per 100 touches — broader carelessness metric
    scouts['Loss_per100T'] = (scouts['possessionLost'] / touch_safe * 100).fillna(0)

    # Times dribbled past per 90 — defensive shape / solidity
    scouts['DribbledPast_90'] = scouts['dribbledPast'] / s_90s


    # Shots from outside box per 90 — range, audacity, #8 profile indicator
    scouts['ShotsOB_90'] = scouts['shotsFromOutsideTheBox'] / s_90s

    scouts = scouts.fillna(0)
    return scouts


scouts = load_scouts(os.path.getmtime(CSV_PATH))

STATS = [
    # ── DEFENSIVE ─────────────────────────────────────────────────────────────
    ("TacklesWon_90",    "Tackles Won /90",              "🛡️ Defensive"),
    ("Tackle_Win_Pct",   "Tackle Win %",                 "🛡️ Defensive"),
    ("Interception_90",  "Interceptions /90",            "🛡️ Defensive"),
    ("Clearance_90",     "Clearances /90",               "🛡️ Defensive"),
    ("Total_Duel",       "Total Duels Won /90",        "🛡️ Defensive"),
    ("Block_90",        "Blocks /90",                     "🛡️ Defensive"),
    ("Aerial_Won_Pct",   "Aerial Duel Won %",            "🛡️ Defensive"),
    ("Ground_Duel_Pct",  "Ground Duel Won %",            "🛡️ Defensive"),
    ("Recovery_90",      "Ball Recoveries /90",          "🛡️ Defensive"),
    ("PressWon_90",      "Possession Won Att Third /90", "🛡️ Defensive"),
    ("DribbledPast_90",  "Dribbled Past /90 ↓",         "🛡️ Defensive"),

    # ── PASSING ───────────────────────────────────────────────────────────────
    ("Pass%",            "Pass Accuracy %",              "🎯 Passing"),
    ("LongBall_90",      "Accurate Long Balls /90",      "🎯 Passing"),
    ("LongBall_Acc%",    "Long Ball Accuracy %",         "🎯 Passing"),
    ("OppHalf_90",       "Opp Half Passes /90",          "🎯 Passing"),
    ("Final_Third_90",   "Final Third Passes /90",         "🎯 Passing"),
    ("xA_90",            "xA /90",                      "🎯 Passing"),


    # ── INVOLVEMENT ───────────────────────────────────────────────────────────
    ("xG_Buildup_90",    "xG Buildup /90",               "⚙️ Involvement"),
    ("xG_Chain_90",      "xG Chain /90",                 "⚙️ Involvement"),
    #("Final_Third_90",   "Final Third Passes /90",       "⚙️ Involvement"),
    ("Touch_90",         "Touches /90",                  "⚙️ Involvement"),

    # ── FINAL PRODUCT ─────────────────────────────────────────────────────────
    ("Direct_Creation_90", "Direct Creation /90",        "🔑 Final Product"),
    ("npxGxA_90",          "npxG+xA /90",                   "🔑 Final Product"),
    ("KeyPass_90",       "Key Passes /90",               "🔑 Final Product"),
    ("BigChance_90",     "Big Chances Created /90",      "🔑 Final Product"),

    # ── DRIBBLING ─────────────────────────────────────────────────────────────
    ("Dribbles_90",      "Successful Dribbles /90",      "🏃 Dribbling"),
    ("Dribble_Succ%",    "Dribble Success %",            "🏃 Dribbling"),
    ("Disp_per100T",     "Dispossessed /100 Touches ↓", "🏃 Dribbling"),


    # ── SHOOTING ──────────────────────────────────────────────────────────────
    ("Shots_90",         "Shots /90",                    "⚽ Shooting"),
    ("SoT_90",           "Shots on Target /90",          "⚽ Shooting"),
    ("SoT%",             "Shot Accuracy %",              "⚽ Shooting"),
    ("npxG_90",          "npxG /90",                     "⚽ Shooting"),
    ("G/Sh",             "Goals per Shot",               "⚽ Shooting"),
    ("G/SoT",            "Goals per SoT",                "⚽ Shooting"),
    #("ShotsInBox_90",    "Shots Inside Box /90",         "⚽ Shooting"),
    #("ShotsOB_90",       "Shots Outside Box /90",        "⚽ Shooting"),

    # ── EFFICIENCY ────────────────────────────────────────────────────────────
    ("xA_per_KP",        "xA per Key Pass",              "📊 Efficiency"),
    ("npxG_per_Sh",      "npxG per Shot",                "📊 Efficiency"),
    ("DC_per_Touch",     "Direct Creation per Touch",    "📊 Efficiency"),
    ("FT_per_Touch",     "Final Third Pass Rate",        "📊 Efficiency"),
    ("Loss_per100T",     "Possession Lost /100T ↓",      "📊 Efficiency"),
]

INVERTED_STATS = {
    "Disp_per100T",    # rarely dispossessed relative to touches
    "Loss_per100T",    # rarely loses possession relative to touches
    "DribbledPast_90", # rarely beaten by dribblers
}

# Defensive *volume* per-90 stats whose raw counts inflate in pressing-heavy
# leagues (Bundesliga / EPL) and deflate in possession-heavy leagues (La Liga).
# Efficiency rates (Tackle_Win_Pct, Aerial_Won_Pct, Ground_Duel_Pct) are NOT
# in this list — they're already percentages and don't need adjustment.
LEAGUE_ADJ_COLS = [
    "TacklesWon_90", "Interception_90", "Clearance_90",
    "Recovery_90",   "PressWon_90",     "Total_Duel",
    "Block_90",      "DribbledPast_90",
]

ALL_COLS   = [s[0] for s in STATS]
STAT_LABEL = {s[0]: s[1] for s in STATS}

# Plain-English glossary surfaced as hover tooltips throughout the app.
# Keep these single-line, jargon-light, ~15 words. The ↓ marker on inverted
# stats means "lower raw value is better" — the percentile is auto-flipped.
STAT_HELP = {
    "TacklesWon_90":     "Successful tackles per 90 minutes — volume of clean defensive engagement.",
    "Tackle_Win_Pct":    "Of tackles attempted, share won cleanly — reading and timing.",
    "Interception_90":   "Passes cut off per 90 — anticipation, not engagement.",
    "Clearance_90":      "Defensive clearances per 90 — ball cleared from danger.",
    "Total_Duel":        "Total duels won (ground + aerial) per 90 — overall physical battles.",
    "Block_90":          "Shots and crosses blocked per 90 — last-line defending.",
    "Aerial_Won_Pct":    "Share of aerial duels won — heading dominance.",
    "Ground_Duel_Pct":   "Share of ground duels won — physical 1v1 contests.",
    "Recovery_90":       "Loose-ball recoveries per 90 — being in the right place.",
    "PressWon_90":       "Possessions won in the attacking third per 90 — high-press output.",
    "DribbledPast_90":   "↓ Times beaten by an opponent's dribble per 90 — lower is better.",

    "Pass%":             "Pass completion percentage — overall ball security.",
    "LongBall_90":       "Successful long balls per 90 — switching play, releasing runners.",
    "LongBall_Acc%":     "Accuracy on long balls — quality of long passing.",
    "OppHalf_90":        "Successful passes in the opposition half per 90 — territorial influence.",
    "Cross_90":          "Successful crosses per 90 — wide service into the box.",
    "xA_90":             "Expected assists per 90 — quality of chances created, regardless of finish.",

    "xG_Buildup_90":     "xG of chains the player joined, EXCLUDING the shot/key pass — pure deep involvement.",
    "xG_Chain_90":       "xG of every chain the player touched — overall presence in attacking moves.",
    "Final_Third_90":    "Successful passes into the final third per 90 — ball progression.",
    "Touch_90":          "Total touches per 90 — sheer involvement in play.",

    "Direct_Creation_90":"Chain minus Buildup — value created by the player's own decisive action (shot or key pass).",
    "npxGxA_90":         "Non-penalty xG plus xA per 90 — bottom-line attacking output.",
    "KeyPass_90":        "Passes leading directly to a shot per 90.",
    "BigChance_90":      "Clear-cut chances created per 90 (Opta definition).",

    "Dribbles_90":       "Successful dribbles past an opponent per 90.",
    "Dribble_Succ%":     "Of dribbles attempted, share successful — efficiency.",
    "Disp_per100T":      "↓ Times dispossessed per 100 touches — lower is better. Normalised so high-touch players aren't unfairly punished.",

    "Shots_90":          "Total shots per 90 — shooting volume.",
    "SoT_90":            "Shots on target per 90.",
    "SoT%":              "Share of shots on target — accuracy.",
    "npxG_90":           "Non-penalty expected goals per 90 — quality of chances taken.",
    "G/Sh":              "Goals per shot — finishing efficiency.",
    "G/SoT":             "Goals per shot on target — finishing among efforts that hit.",

    "xA_per_KP":         "xA per key pass — quality of chances created, not just volume.",
    "npxG_per_Sh":       "npxG per shot — average shot quality (proxy for shot selection).",
    "DC_per_Touch":      "Direct creation per touch — how often touches lead to dangerous actions.",
    "FT_per_Touch":      "Final-third passes per touch — territorial efficiency.",
    "Loss_per100T":      "↓ Possessions lost per 100 touches — careless turnovers, lower is better.",
}

# ─────────────────────────────────────────────────────────────────────────────
# I18N — language packs and helpers. Keep all user-facing strings in TXT/STAT_*
# so swapping languages doesn't require touching any rendering code. Internal
# identifiers (column names, preset keys, page keys) stay English so dict
# lookups never break.
# ─────────────────────────────────────────────────────────────────────────────

LANG_OPTIONS = {"English": "en", "Tiếng Việt": "vi"}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def _lang() -> str:
    return st.session_state.get("lang", "en")

# Vietnamese stat labels
STAT_LABEL_VI = {
    "TacklesWon_90":     "Tắc bóng thắng /90",
    "Tackle_Win_Pct":    "Tỷ lệ thắng tắc bóng %",
    "Interception_90":   "Cắt bóng /90",
    "Clearance_90":      "Phá bóng /90",
    "Total_Duel":        "Tổng tranh chấp thắng /90",
    "Block_90":          "Cản phá /90",
    "Aerial_Won_Pct":    "Thắng không chiến %",
    "Ground_Duel_Pct":   "Thắng tranh chấp đất %",
    "Recovery_90":       "Thu hồi bóng /90",
    "PressWon_90":       "Đoạt bóng 1/3 sân ĐP /90",
    "DribbledPast_90":   "Bị qua người /90 ↓",
    "Pass%":             "Độ chính xác chuyền %",
    "LongBall_90":       "Bóng dài chính xác /90",
    "LongBall_Acc%":     "Độ chính xác bóng dài %",
    "OppHalf_90":        "Chuyền sân ĐP /90",
    "Cross_90":          "Tạt bóng chính xác /90",
    "xA_90":             "xA /90",
    "xG_Buildup_90":     "xG Buildup /90",
    "xG_Chain_90":       "xG Chain /90",
    "Final_Third_90":    "Chuyền 1/3 cuối /90",
    "Touch_90":          "Chạm bóng /90",
    "Direct_Creation_90":"Tạo cơ hội trực tiếp /90",
    "npxGxA_90":         "npxG+xA /90",
    "KeyPass_90":        "Chuyền quyết định /90",
    "BigChance_90":      "Cơ hội rõ tạo ra /90",
    "Dribbles_90":       "Rê bóng thành công /90",
    "Dribble_Succ%":     "Tỷ lệ rê bóng %",
    "Disp_per100T":      "Mất bóng /100 chạm ↓",
    "Shots_90":          "Sút /90",
    "SoT_90":            "Sút trúng đích /90",
    "SoT%":              "Độ chính xác sút %",
    "npxG_90":           "npxG /90",
    "G/Sh":              "Bàn/cú sút",
    "G/SoT":             "Bàn/sút trúng đích",
    "xA_per_KP":         "xA/chuyền QĐ",
    "npxG_per_Sh":       "npxG/cú sút",
    "DC_per_Touch":      "Tạo cơ hội/chạm bóng",
    "FT_per_Touch":      "Tỷ lệ chuyền 1/3 cuối",
    "Loss_per100T":      "Mất kiểm soát /100 chạm ↓",
}

STAT_HELP_VI = {
    "TacklesWon_90":     "Tắc bóng thành công mỗi 90 phút — số lượng can thiệp phòng ngự.",
    "Tackle_Win_Pct":    "Trong các pha tắc bóng đã thử, tỷ lệ thắng — đọc trận và canh giờ.",
    "Interception_90":   "Đường chuyền bị cắt mỗi 90 — phán đoán, không phải va chạm.",
    "Clearance_90":      "Phá bóng giải vây mỗi 90 — đẩy bóng khỏi vùng nguy hiểm.",
    "Total_Duel":        "Tổng tranh chấp thắng (đất + bổng) mỗi 90 — sức mạnh tranh chấp tổng thể.",
    "Block_90":          "Cú sút và đường tạt bị chặn mỗi 90 — phòng ngự cuối cùng.",
    "Aerial_Won_Pct":    "Tỷ lệ tranh chấp bổng thắng — sức mạnh đánh đầu.",
    "Ground_Duel_Pct":   "Tỷ lệ tranh chấp dưới đất thắng — đối đầu 1v1 thể chất.",
    "Recovery_90":       "Thu hồi bóng lỏng mỗi 90 — luôn đúng vị trí.",
    "PressWon_90":       "Đoạt bóng ở 1/3 sân đối phương mỗi 90 — pressing tầm cao.",
    "DribbledPast_90":   "↓ Bị đối phương qua mặt mỗi 90 — càng thấp càng tốt.",
    "Pass%":             "Tỷ lệ chuyền chính xác — độ an toàn của bóng.",
    "LongBall_90":       "Bóng dài chính xác mỗi 90 — chuyển hướng, giải phóng đồng đội.",
    "LongBall_Acc%":     "Độ chính xác bóng dài — chất lượng chuyền dài.",
    "OppHalf_90":        "Chuyền chính xác ở sân đối phương mỗi 90 — ảnh hưởng lãnh thổ.",
    "Cross_90":          "Tạt bóng chính xác mỗi 90 — phục vụ từ cánh vào vòng cấm.",
    "xA_90":             "Kiến tạo kỳ vọng mỗi 90 — chất lượng cơ hội tạo, không phụ thuộc người dứt điểm.",
    "xG_Buildup_90":     "xG của các pha bóng cầu thủ tham gia, KHÔNG TÍNH cú sút/chuyền QĐ — tham gia sâu thuần túy.",
    "xG_Chain_90":       "xG của mọi pha bóng cầu thủ chạm — sự hiện diện trong các pha tấn công.",
    "Final_Third_90":    "Chuyền chính xác vào 1/3 cuối sân mỗi 90 — đẩy bóng lên.",
    "Touch_90":          "Tổng số lần chạm bóng mỗi 90 — mức độ tham gia.",
    "Direct_Creation_90":"Chain trừ Buildup — giá trị tạo bởi hành động quyết định (sút hoặc chuyền QĐ).",
    "npxGxA_90":         "npxG cộng xA mỗi 90 — sản phẩm tấn công cuối cùng.",
    "KeyPass_90":        "Đường chuyền dẫn trực tiếp đến cú sút mỗi 90.",
    "BigChance_90":      "Cơ hội rõ ràng tạo ra mỗi 90 (theo định nghĩa Opta).",
    "Dribbles_90":       "Lần rê bóng vượt qua đối phương thành công mỗi 90.",
    "Dribble_Succ%":     "Tỷ lệ rê bóng thành công — hiệu quả.",
    "Disp_per100T":      "↓ Số lần mất bóng mỗi 100 lần chạm — càng thấp càng tốt. Chuẩn hóa để không trừng phạt cầu thủ chạm bóng nhiều.",
    "Shots_90":          "Tổng cú sút mỗi 90 — số lượng dứt điểm.",
    "SoT_90":            "Sút trúng đích mỗi 90.",
    "SoT%":              "Tỷ lệ sút trúng đích — độ chính xác.",
    "npxG_90":           "Bàn thắng kỳ vọng không tính phạt đền mỗi 90 — chất lượng cơ hội tạo.",
    "G/Sh":              "Bàn thắng trên mỗi cú sút — hiệu quả dứt điểm.",
    "G/SoT":             "Bàn thắng trên mỗi cú sút trúng đích.",
    "xA_per_KP":         "xA mỗi chuyền quyết định — chất lượng cơ hội, không chỉ số lượng.",
    "npxG_per_Sh":       "npxG mỗi cú sút — chất lượng dứt điểm trung bình.",
    "DC_per_Touch":      "Tạo cơ hội trực tiếp mỗi lần chạm — tần suất chạm bóng dẫn đến nguy hiểm.",
    "FT_per_Touch":      "Chuyền 1/3 cuối mỗi lần chạm — hiệu quả lãnh thổ.",
    "Loss_per100T":      "↓ Mất quyền kiểm soát mỗi 100 lần chạm — sai lầm cẩu thả, càng thấp càng tốt.",
}

CATEGORY_VI = {
    "🛡️ Defensive":      "🛡️ Phòng Ngự",
    "🎯 Passing":         "🎯 Chuyền Bóng",
    "⚙️ Involvement":    "⚙️ Tham Gia",
    "🔑 Final Product":   "🔑 Sản Phẩm Cuối",
    "🏃 Dribbling":       "🏃 Rê Bóng",
    "⚽ Shooting":        "⚽ Dứt Điểm",
    "📊 Efficiency":      "📊 Hiệu Quả",
}

# Italian/Spanish role names stay as-is (international tactical terms)
PRESET_LABEL_VI = {
    "Custom (Manual)":         "Tùy chỉnh (Thủ công)",
    "Anchor Man":              "Anchor Man (Tiền vệ Trụ)",
    "Ball-Winning Midfielder": "Tiền vệ Đánh chặn (BWM)",
    "Half-Back":               "Half-Back",
    "Deep-Lying Playmaker":    "Nhạc trưởng Lùi sâu (DLP)",
    "Regista":                 "Regista",
    "Carrilero":               "Carrilero",
    "Central Midfielder":      "Tiền vệ Trung tâm (CM)",
    "Box-to-Box":              "Box-to-Box",
    "Segundo Volante":         "Segundo Volante",
    "Mezzala":                 "Mezzala",
    "Wide Midfielder":         "Tiền vệ Cánh (WM)",
    "Advanced Playmaker":      "Nhạc trưởng Tấn công (AP)",
    "Trequartista":            "Trequartista",
}

# Page routing keys — internal identifiers, never displayed
PAGE_KEYS = ["scouting", "leaderboards", "compare"]

# All UI strings. Use named placeholders ({name}) so word order can vary.
TXT = {
    "en": {
        # Sidebar / nav
        "lang_label":      "🌐 Language / Ngôn ngữ",
        "app_title":       "⚽ CT's Scouting Dashboard",
        "app_caption":     "Big 5 Leagues · 2025–26 · Midfielders (900+ min)",
        "nav_label":       "Navigate",
        "nav_scouting":    "🔍 Scouting Report",
        "nav_leaderboards":"📊 Stat Leaderboards",
        "nav_compare":     "⚖️ Player Comparison",

        # Methodology
        "howto_header":      "ℹ️ How to read this page",
        "howto_percentile":  "**Percentile rank.** Every bar in this app is a percentile (0–100) of how this player compares to all Big-5 midfielders with 900+ minutes. 90th percentile means they're better than 90% of their peers at that stat. Stats marked ↓ are inverted — the percentile is flipped so a high score still means 'good'.",
        "howto_rolefit":     "**Role-fit grade (0–100, shown as ★ stars).** Tell the app what you want from the role using the 0–5 sliders (0 = don't want, 2.5 = neutral, 5 = strongly want). For each stat that isn't neutral, we take the player's percentile and weight it by your importance. Categories are equalised so the 11 defensive stats don't drown the 4 final-product stats. The result is the importance-weighted average — if everything you care about is at the 90th percentile, the grade is 90.",
        "howto_similarity":  "**Similarity %.** Mean absolute percentile gap between two players, flipped to 0–100%. Categories are equalised so a creative #10 isn't matched on his tackling profile. The role-bias slider blends in your active preset's importances — at 1.0, only stats relevant to that role count toward the match.",

        # Leaderboards
        "lb_title":         "📊 Stat Leaderboards",
        "lb_caption":       "Top 10 midfielders per individual metric. Each stat is independent — no aggregation bias.",
        "lb_howto_extra":   "**Pct (filtered)** in each table is the percentile rank *within your current filter pool*, not the league-wide percentile — useful when narrowing to one league or a U21 cohort. Hover any **stat heading** for a plain-English definition of that metric.",
        "filters_header":   "🔽 Filters",
        "filter_league":    "League",
        "filter_age_range": "Age range",
        "filter_min_mins":  "Min. minutes played",
        "lb_showing":       "Showing **{n}** players after filters.",
        "lb_no_results":    "No players match the current filters.",
        "col_player":       "Player",
        "col_team":         "Team",
        "col_league":       "League",
        "col_pct_filtered": "Pct (filtered)",

        # Scouting
        "sc_title":           "🔍 Scouting Report",
        "sc_workflow":        "**Workflow:** pick a player → choose a role preset (or set sliders manually) → read the stars + radar + bars to see *how well this player fits the brief*. Hover any **stat label** for a definition.",
        "sc_select_player":   "Select a player:",
        "sc_select_player_h": "The pool is Big-5 league midfielders with at least 900 minutes played this season.",
        "sc_tactical":        "#### ⚖️ Tactical Profile",
        "sc_slider_intro":    "Each slider asks how much you want that stat in your role. **0** = actively don't want · **2.5** = neutral (ignored) · **5** = strongly want. Pick a preset to auto-fill, then tweak.",
        "sc_preset":          "🎯 Profile preset:",
        "sc_preset_help":     "Loads a Football Manager-style role template into the sliders. Pick the closest archetype to your brief, then fine-tune.",
        "sc_role_fit":        "Role Fit",
        "sc_attr_breakdown":  "**Attribute Breakdown**",
        "sc_attr_caption":    "Grouped by category · bars show rank vs all Big 5 midfielders",
        "sc_similar_header":  "#### 🧬 Similar Statistical Profiles",
        "sc_similar_caption": "Category-equalised percentile match. Crank the role-bias slider to weight stats you actually care about for this profile.",
        "sc_role_bias":       "Role bias  ·  0 = pure shape match · 1 = match within active preset",
        "sc_role_bias_help":  "At 0 we compare across all stats equally — pure stylistic lookalikes. At 1 we only count stats that matter for the active role preset, so 'similar as a Regista' will surface different names than 'similar as an Anchor'.",
        "sc_age_window":      "Age window (± years)",
        "sc_age_window_help": "Only show players within this many years of the target's age. Useful for finding younger/cheaper alternatives or aged-down replacements.",
        "sc_no_similar":      "No comparable players inside the current age window.",
        "sc_match":           "match",
        "sc_role":            "role fit",
        "sc_age_short":       "Age",
        "sc_min_short":       "min",
        "sc_sofascore":       "SofaScore",

        # Comparison
        "cmp_title":            "⚖️ Player Comparison",
        "cmp_caption":          "Side-by-side percentile breakdown. Winner stays bright, loser fades. Equal stats keep their colour.",
        "cmp_howto_similarity": "**Style Similarity %** — mean absolute percentile gap across every stat, flipped to 0–100. Two players who score nearly identically across every metric will hit ~95%+; two players with totally opposite profiles end up in the 40–60% band.",
        "cmp_howto_diff":       "**Biggest Differentiators** lists the 5 stats where each player most outperforms the other in percentile terms — the quickest answer to *'where does Player A actually beat Player B'*.",
        "cmp_player_a":         "Player A",
        "cmp_player_a_help":    "Shown in BLUE throughout this page.",
        "cmp_player_b":         "Player B",
        "cmp_player_b_help":    "Shown in ORANGE throughout this page.",
        "cmp_lens":             "Role-fit lens",
        "cmp_lens_help":        "Compare both players' fit for this role. Switch the lens to ask 'who's the better Regista?' vs 'who's the better Anchor?' — same two players, very different verdicts.",
        "cmp_pick_diff":        "Pick two different players to compare.",
        "cmp_role_fit_label":   "Role fit ({preset}) · {grade}",
        "cmp_neutral_lens":     "neutral",
        "cmp_ahead":            "{name} ahead",
        "cmp_ahead_val":        "{w} / {n}",
        "cmp_ties_gap":         "Ties · Avg gap",
        "cmp_ties_gap_val":     "{ties} · {gap}",
        "cmp_style_sim":        "Style similarity",
        "cmp_diff_header":      "#### 🔥 Biggest Differentiators",
        "cmp_diff_caption":     "Where each player most clearly outperforms the other.",
        "cmp_diff_gap":         "+{gap} pct ({pa} vs {pb})",
        "cmp_radar_header":     "#### 🎯 Category Radar — Overlapped",
        "cmp_breakdown_header": "#### 📈 Full Breakdown",
        "cmp_breakdown_caption":"Bars grow inward toward the stat label · numbers are percentile ranks.",

        # Card glyphs
        "card_goals":   "G",
        "card_assists": "A",

        # Comparison cohorts (Scouting Report — what pool to rank against)
        "cohort_label":       "Comparison pool",
        "cohort_help":        "What pool to rank this player against. The smaller the pool, the more demanding the percentiles. Default is the full Big-5 midfielder pool.",
        "cohort_status":      "Comparing **{player}** against **{n}** {cohort}.",
        "cohort_status_solo": "Cohort has only this player — try a wider pool.",
        "cohort_all":         "All Big-5 midfielders",
        "cohort_same_league": "Same league only",
        "cohort_u21":         "U21 (wonderkid pool)",
        "cohort_u23":         "U23 (prospect pool)",
        "cohort_u25":         "U25 (development pool)",
        "cohort_vets30":      "30+ veterans",
        "cohort_regulars":    "Regular starters (1500+ min)",
        "cohort_age_bracket": "Same age bracket (±2 yrs)",

        # League pressing-strength adjustment toggle
        "league_adj_label":  "League pressing adjustment",
        "league_adj_help":   "Scales defensive volume stats (tackles, interceptions, recoveries, presses, blocks, total duels) by each league's average so pressing-heavy leagues like Bundesliga don't get an unfair boost vs possession-heavy ones like La Liga. Efficiency rates (Tackle Win %, etc.) are unaffected. Off by default; flip on for cross-league fairness.",
        "league_adj_active": "League-adjusted view",
    },
    "vi": {
        "lang_label":      "🌐 Language / Ngôn ngữ",
        "app_title":       "⚽ Bảng Tuyển Trạch của CT",
        "app_caption":     "5 Giải Hàng Đầu · 2025–26 · Tiền vệ (900+ phút)",
        "nav_label":       "Điều hướng",
        "nav_scouting":    "🔍 Báo Cáo Tuyển Trạch",
        "nav_leaderboards":"📊 Bảng Xếp Hạng Chỉ Số",
        "nav_compare":     "⚖️ So Sánh Cầu Thủ",

        "howto_header":      "ℹ️ Hướng dẫn đọc trang này",
        "howto_percentile":  "**Thứ hạng phân vị.** Mọi thanh trong ứng dụng là phân vị (0–100) so với toàn bộ tiền vệ 5 giải hàng đầu chơi 900+ phút. Phân vị 90 nghĩa là cầu thủ giỏi hơn 90% đồng nghiệp ở chỉ số đó. Chỉ số có ↓ được đảo ngược — phân vị đã được lật để điểm cao luôn nghĩa là 'tốt'.",
        "howto_rolefit":     "**Độ phù hợp vai trò (0–100, hiển thị bằng ★ sao).** Cho ứng dụng biết bạn muốn gì ở vai trò bằng các thanh trượt 0–5 (0 = không muốn, 2.5 = trung lập, 5 = rất muốn). Với mỗi chỉ số không trung lập, ta lấy phân vị của cầu thủ và nhân với độ quan trọng. Các thể loại được cân bằng để 11 chỉ số phòng ngự không lấn át 4 chỉ số sản phẩm cuối. Kết quả là trung bình có trọng số — nếu mọi thứ bạn quan tâm ở phân vị 90, điểm là 90.",
        "howto_similarity":  "**Độ tương tự %.** Chênh lệch phân vị tuyệt đối trung bình giữa hai cầu thủ, đảo về thang 0–100%. Các thể loại được cân bằng để một số 10 sáng tạo không bị khớp theo hồ sơ tắc bóng. Thanh thiên vai trò pha trộn độ quan trọng của vai trò đang chọn — ở 1.0, chỉ các chỉ số liên quan đến vai trò mới được tính.",

        "lb_title":         "📊 Bảng Xếp Hạng Chỉ Số",
        "lb_caption":       "Top 10 tiền vệ theo từng chỉ số. Mỗi chỉ số độc lập — không tổng hợp.",
        "lb_howto_extra":   "**Phân vị (lọc)** trong mỗi bảng là thứ hạng phân vị *trong nhóm đã lọc*, không phải toàn giải — hữu ích khi thu hẹp về một giải hoặc nhóm U21. Di chuột qua **tên chỉ số** để xem định nghĩa.",
        "filters_header":   "🔽 Bộ lọc",
        "filter_league":    "Giải đấu",
        "filter_age_range": "Khoảng tuổi",
        "filter_min_mins":  "Số phút tối thiểu",
        "lb_showing":       "Hiển thị **{n}** cầu thủ sau bộ lọc.",
        "lb_no_results":    "Không có cầu thủ nào khớp bộ lọc.",
        "col_player":       "Cầu thủ",
        "col_team":         "CLB",
        "col_league":       "Giải",
        "col_pct_filtered": "Phân vị (lọc)",

        "sc_title":           "🔍 Báo Cáo Tuyển Trạch",
        "sc_workflow":        "**Quy trình:** chọn cầu thủ → chọn vai trò có sẵn (hoặc kéo thanh trượt thủ công) → đọc sao + radar + thanh chỉ số để xem *cầu thủ phù hợp vai trò ra sao*. Di chuột qua **tên chỉ số** để xem định nghĩa.",
        "sc_select_player":   "Chọn cầu thủ:",
        "sc_select_player_h": "Nhóm dữ liệu là tiền vệ ở 5 giải hàng đầu chơi tối thiểu 900 phút mùa này.",
        "sc_tactical":        "#### ⚖️ Hồ Sơ Chiến Thuật",
        "sc_slider_intro":    "Mỗi thanh trượt hỏi bạn muốn chỉ số đó nhiều như thế nào trong vai trò. **0** = không muốn · **2.5** = trung lập (bỏ qua) · **5** = rất muốn. Chọn vai trò có sẵn để tự điền, sau đó tinh chỉnh.",
        "sc_preset":          "🎯 Vai trò có sẵn:",
        "sc_preset_help":     "Tải mẫu vai trò kiểu Football Manager vào thanh trượt. Chọn mẫu gần nhất rồi tinh chỉnh.",
        "sc_role_fit":        "Độ Phù Hợp",
        "sc_attr_breakdown":  "**Phân Tích Chỉ Số**",
        "sc_attr_caption":    "Nhóm theo thể loại · thanh hiển thị thứ hạng so với mọi tiền vệ 5 giải hàng đầu",
        "sc_similar_header":  "#### 🧬 Hồ Sơ Chỉ Số Tương Tự",
        "sc_similar_caption": "Khớp phân vị cân bằng theo thể loại. Tăng độ thiên vai trò để ưu tiên các chỉ số thực sự quan trọng cho hồ sơ này.",
        "sc_role_bias":       "Thiên vai trò  ·  0 = khớp thuần theo số liệu · 1 = khớp trong vai trò đang chọn",
        "sc_role_bias_help":  "Ở mức 0, so sánh đồng đều mọi chỉ số — kiểu phong cách thuần túy. Ở mức 1, chỉ tính các chỉ số quan trọng cho vai trò đang chọn, nên 'giống Regista' sẽ ra danh sách khác 'giống Anchor'.",
        "sc_age_window":      "Khoảng tuổi (± năm)",
        "sc_age_window_help": "Chỉ hiển thị cầu thủ trong khoảng tuổi này so với mục tiêu. Hữu ích để tìm phương án trẻ/rẻ hơn hoặc người thay thế.",
        "sc_no_similar":      "Không có cầu thủ tương tự trong khoảng tuổi hiện tại.",
        "sc_match":           "khớp",
        "sc_role":            "phù hợp",
        "sc_age_short":       "Tuổi",
        "sc_min_short":       "phút",
        "sc_sofascore":       "SofaScore",

        "cmp_title":            "⚖️ So Sánh Cầu Thủ",
        "cmp_caption":          "Phân tích phân vị cạnh nhau. Người thắng giữ màu sáng, người thua mờ đi. Bằng nhau giữ nguyên màu.",
        "cmp_howto_similarity": "**Độ tương tự phong cách %** — chênh lệch phân vị tuyệt đối trung bình trên mọi chỉ số, đảo về thang 0–100. Hai cầu thủ có điểm gần như giống hệt sẽ đạt ~95%+; hai hồ sơ trái ngược thường rơi vào khoảng 40–60%.",
        "cmp_howto_diff":       "**Khác Biệt Lớn Nhất** liệt kê 5 chỉ số mà mỗi cầu thủ vượt trội rõ nhất — câu trả lời nhanh nhất cho *'Cầu thủ A thắng B ở đâu thực sự'*.",
        "cmp_player_a":         "Cầu thủ A",
        "cmp_player_a_help":    "Hiển thị màu XANH xuyên suốt trang.",
        "cmp_player_b":         "Cầu thủ B",
        "cmp_player_b_help":    "Hiển thị màu CAM xuyên suốt trang.",
        "cmp_lens":             "Lăng kính vai trò",
        "cmp_lens_help":        "So sánh độ phù hợp của cả hai cho vai trò này. Đổi lăng kính để hỏi 'ai là Regista tốt hơn?' so với 'ai là Anchor tốt hơn?' — cùng hai cầu thủ, kết luận rất khác.",
        "cmp_pick_diff":        "Chọn hai cầu thủ khác nhau để so sánh.",
        "cmp_role_fit_label":   "Độ phù hợp ({preset}) · {grade}",
        "cmp_neutral_lens":     "trung lập",
        "cmp_ahead":            "{name} dẫn",
        "cmp_ahead_val":        "{w} / {n}",
        "cmp_ties_gap":         "Hòa · Chênh trung bình",
        "cmp_ties_gap_val":     "{ties} · {gap}",
        "cmp_style_sim":        "Độ tương tự phong cách",
        "cmp_diff_header":      "#### 🔥 Khác Biệt Lớn Nhất",
        "cmp_diff_caption":     "Nơi mỗi cầu thủ vượt trội rõ rệt nhất so với người kia.",
        "cmp_diff_gap":         "+{gap} phân vị ({pa} vs {pb})",
        "cmp_radar_header":     "#### 🎯 Radar Thể Loại — Chồng Lên Nhau",
        "cmp_breakdown_header": "#### 📈 Phân Tích Toàn Diện",
        "cmp_breakdown_caption":"Thanh phát triển vào trong về phía nhãn chỉ số · số là thứ hạng phân vị.",

        "card_goals":   "B",
        "card_assists": "K",

        "cohort_label":       "Nhóm so sánh",
        "cohort_help":        "Nhóm cầu thủ dùng để xếp hạng. Nhóm càng nhỏ thì phân vị càng khắt khe. Mặc định là toàn bộ tiền vệ 5 giải hàng đầu.",
        "cohort_status":      "So sánh **{player}** với **{n}** {cohort}.",
        "cohort_status_solo": "Nhóm chỉ có mỗi cầu thủ này — hãy chọn nhóm rộng hơn.",
        "cohort_all":         "Toàn bộ tiền vệ 5 giải",
        "cohort_same_league": "Chỉ cùng giải đấu",
        "cohort_u21":         "U21 (mầm non)",
        "cohort_u23":         "U23 (triển vọng)",
        "cohort_u25":         "U25 (phát triển)",
        "cohort_vets30":      "30+ (cựu binh)",
        "cohort_regulars":    "Đá chính thường xuyên (1500+ phút)",
        "cohort_age_bracket": "Cùng nhóm tuổi (±2 năm)",

        "league_adj_label":  "Điều chỉnh pressing theo giải",
        "league_adj_help":   "Cân chỉnh các chỉ số phòng ngự *về số lượng* (tắc bóng, cắt bóng, thu hồi, đoạt bóng, cản phá, tranh chấp) theo trung bình của từng giải để giải pressing mạnh như Bundesliga không được lợi thế bất công so với giải kiểm soát bóng như La Liga. Các tỷ lệ hiệu quả (Tỷ lệ thắng tắc bóng, v.v.) không bị ảnh hưởng. Mặc định tắt; bật khi cần so sánh công bằng giữa các giải.",
        "league_adj_active": "Đã điều chỉnh giải",
    },
}

def t(key: str, **fmt) -> str:
    """Translate UI string. Falls back to English then to the key itself."""
    s = TXT.get(_lang(), {}).get(key) or TXT["en"].get(key, key)
    return s.format(**fmt) if fmt else s

def stat_label(col: str) -> str:
    if _lang() == "vi":
        return STAT_LABEL_VI.get(col, STAT_LABEL.get(col, col))
    return STAT_LABEL.get(col, col)

def stat_help(col: str) -> str:
    if _lang() == "vi":
        return STAT_HELP_VI.get(col, STAT_HELP.get(col, ""))
    return STAT_HELP.get(col, "")

def cat_label(c: str) -> str:
    if _lang() == "vi":
        return CATEGORY_VI.get(c, c)
    return c

def preset_label(p: str) -> str:
    if _lang() == "vi":
        return PRESET_LABEL_VI.get(p, p)
    return p

# One-paragraph plain-English explanation of each tactical preset, FM-style.
PRESET_DESC_EN = {
    "Custom (Manual)":         "All sliders neutral — set them yourself to define a custom role.",
    "Anchor Man":              "A pure defensive midfielder who sits in front of the back line. Wins the ball, keeps it simple, rarely ventures forward. The destroyer.",
    "Ball-Winning Midfielder": "An aggressive ball-recoverer focused on breaking up play. Tackles, intercepts, presses, then hands it off to teammates.",
    "Half-Back":               "A deep-lying defender-cum-midfielder who drops between the centre-backs to start the build-up. Long passing range, defensive solidity, aerial presence.",
    "Deep-Lying Playmaker":    "A creative #6 who dictates tempo from deep. Touches the ball constantly, picks out long passes, threads it into the final third without venturing forward.",
    "Regista":                 "An advanced creator stationed deep — Pirlo, Xabi Alonso. Does everything a DLP does plus more creative output: assists, key passes, big chances. Rarely tackles.",
    "Carrilero":               "A balanced shuttler covering box-to-box ground. Solid in defence, tidy in possession, contributes some creation. Bridges between the lines.",
    "Central Midfielder":      "An all-rounder with no extreme leanings — roughly average across all categories. Useful as a baseline profile.",
    "Box-to-Box":              "A high-energy two-way midfielder who covers ground from one penalty area to the other. Strong defensively, scores and assists from late runs.",
    "Segundo Volante":         "A defensive midfielder who breaks forward into the box. Strong defensive base with timing-runs and shooting from late attacking surges.",
    "Mezzala":                 "A wide #8 in a 4-3-3 who attacks the half-spaces. Heavy on creation, dribbling, goals. Light on defence.",
    "Wide Midfielder":         "An orthodox wide midfielder in a 4-4-2. Crosses, runs at full-backs, defends the flank. Old-school winger duties.",
    "Advanced Playmaker":      "A creative #10 who orchestrates attacks. Maximum xA, key passes, big chances. Defends little but carries the team's creativity.",
    "Trequartista":            "Pure attacking creator with zero defensive duties. Free roam in the final third — dribbles, shots, assists. The luxury #10.",
}

PRESET_DESC_VI = {
    "Custom (Manual)":         "Tất cả thanh trượt trung lập — tự đặt để định nghĩa vai trò tùy chỉnh.",
    "Anchor Man":              "Tiền vệ phòng ngự thuần túy đứng trước hàng thủ. Đoạt bóng, chơi đơn giản, hiếm khi dâng cao. Mẫu 'kẻ phá hoại'.",
    "Ball-Winning Midfielder": "Người đoạt bóng quyết liệt, tập trung phá lối chơi đối phương. Tắc bóng, cắt bóng, pressing, rồi giao bóng cho đồng đội.",
    "Half-Back":               "Mẫu lùi sâu giữa hai trung vệ để khởi xướng tấn công. Chuyền bóng tầm xa tốt, vững chắc phòng ngự, mạnh không chiến.",
    "Deep-Lying Playmaker":    "Số 6 sáng tạo, điều nhịp từ sâu. Chạm bóng liên tục, tung những đường chuyền dài, đưa bóng vào 1/3 cuối mà không cần dâng cao.",
    "Regista":                 "Nhạc trưởng đứng sâu kiểu Pirlo, Xabi Alonso. Làm mọi thứ DLP làm cộng thêm sáng tạo: kiến tạo, chuyền QĐ, cơ hội lớn. Ít tắc bóng.",
    "Carrilero":               "Tiền vệ con thoi cân bằng box-to-box. Phòng ngự chắc, giữ bóng gọn gàng, có chút sáng tạo. Cầu nối giữa các tuyến.",
    "Central Midfielder":      "Tiền vệ toàn diện không thiên hướng cực đoan — trung bình ở mọi thể loại. Dùng làm hồ sơ chuẩn cơ sở.",
    "Box-to-Box":              "Tiền vệ hai chiều giàu năng lượng, di chuyển từ vòng cấm này sang vòng cấm kia. Phòng ngự khỏe, ghi bàn và kiến tạo từ những pha băng lên muộn.",
    "Segundo Volante":         "Tiền vệ phòng ngự dâng lên tấn công. Nền tảng phòng ngự vững với những pha băng lên đúng thời điểm và sút từ tuyến hai.",
    "Mezzala":                 "Số 8 lệch cánh trong sơ đồ 4-3-3, tấn công các half-space. Nặng về sáng tạo, rê bóng, ghi bàn. Nhẹ phòng ngự.",
    "Wide Midfielder":         "Tiền vệ cánh chính thống trong 4-4-2. Tạt bóng, đi bóng qua hậu vệ biên, phòng ngự cánh. Nhiệm vụ tiền đạo cánh kiểu cũ.",
    "Advanced Playmaker":      "Số 10 sáng tạo điều phối tấn công. Tối đa xA, chuyền QĐ, cơ hội lớn. Phòng ngự ít nhưng gánh sáng tạo cho cả đội.",
    "Trequartista":            "Người sáng tạo tấn công thuần, không có nhiệm vụ phòng ngự. Tự do ở 1/3 cuối — rê bóng, sút, kiến tạo. Số 10 xa xỉ.",
}

def preset_desc(p: str) -> str:
    if _lang() == "vi":
        return PRESET_DESC_VI.get(p, PRESET_DESC_EN.get(p, ""))
    return PRESET_DESC_EN.get(p, "")

def render_preset_card(p: str, accent: str = "#64B4FF") -> None:
    """Styled FM-like description card shown beneath a preset selector."""
    desc = preset_desc(p)
    if not desc:
        return
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #30363d;'
        f'border-left:3px solid {accent};border-radius:6px;'
        f'padding:10px 14px;margin-top:-6px;margin-bottom:10px;'
        f'font-size:12px;line-height:1.5;color:#bbb;">'
        f'<span style="color:{accent};font-weight:700;font-size:11px;'
        f'text-transform:uppercase;letter-spacing:1px;">'
        f'{preset_label(p)}</span><br>{desc}</div>',
        unsafe_allow_html=True,
    )

def page_label(k: str) -> str:
    return t({"scouting": "nav_scouting", "leaderboards": "nav_leaderboards", "compare": "nav_compare"}[k])

def _tip(stat_col: str) -> str:
    """HTML title attribute for a stat — empty string if no help defined."""
    txt = stat_help(stat_col).replace('"', "'")
    return f' title="{txt}"' if txt else ""

PRESETS = {
    "Custom (Manual)": None,

    # ── DEEP / DEFENSIVE ──────────────────────────────────────────────────────

    "Anchor Man": {
        # Defensive
        "TacklesWon_90": 5.0, "Tackle_Win_Pct": 4.5, "Interception_90": 5.0,
        "Clearance_90": 4.5,  "Total_Duel": 5.0,    "Block_90": 4.5,
        "Aerial_Won_Pct": 4.0, "Ground_Duel_Pct": 4.5,
        "Recovery_90": 5.0,    "PressWon_90": 3.0,  "DribbledPast_90": 5.0,
        # Passing
        "Pass%": 4.5,          "LongBall_90": 3.5,  "LongBall_Acc%": 4.0,
        "OppHalf_90": 1.0,     "Cross_90": 0.0,     "xA_90": 0.5,
        # Involvement
        "xG_Buildup_90": 3.5,  "xG_Chain_90": 1.5,  "Final_Third_90": 0.5,
        "Touch_90": 3.5,
        # Final Product
        "Direct_Creation_90": 0.0, "npxGxA_90": 0.0, "KeyPass_90": 0.5,
        "BigChance_90": 0.0,
        # Dribbling
        "Dribbles_90": 1.0,    "Dribble_Succ%": 1.5, "Disp_per100T": 5.0,
        # Shooting
        "Shots_90": 0.0,       "SoT_90": 0.0,       "SoT%": 0.5,
        "npxG_90": 0.0,        "G/Sh": 0.0,         "G/SoT": 0.0,
        # Efficiency
        "xA_per_KP": 0.5,      "npxG_per_Sh": 0.0,  "DC_per_Touch": 0.0,
        "FT_per_Touch": 1.0,   "Loss_per100T": 5.0,
    },

    "Ball-Winning Midfielder": {
        "TacklesWon_90": 5.0, "Tackle_Win_Pct": 5.0, "Interception_90": 5.0,
        "Clearance_90": 3.0,  "Total_Duel": 5.0,    "Block_90": 3.5,
        "Aerial_Won_Pct": 4.0, "Ground_Duel_Pct": 5.0,
        "Recovery_90": 5.0,    "PressWon_90": 5.0,  "DribbledPast_90": 4.5,
        "Pass%": 3.0,          "LongBall_90": 1.5,  "LongBall_Acc%": 2.0,
        "OppHalf_90": 1.5,     "Cross_90": 0.0,     "xA_90": 0.5,
        "xG_Buildup_90": 2.0,  "xG_Chain_90": 1.5,  "Final_Third_90": 1.0,
        "Touch_90": 2.5,
        "Direct_Creation_90": 0.5, "npxGxA_90": 0.5, "KeyPass_90": 1.0,
        "BigChance_90": 0.0,
        "Dribbles_90": 2.0,    "Dribble_Succ%": 2.0, "Disp_per100T": 3.5,
        "Shots_90": 1.0,       "SoT_90": 1.0,       "SoT%": 1.0,
        "npxG_90": 0.5,        "G/Sh": 0.5,         "G/SoT": 0.5,
        "xA_per_KP": 0.5,      "npxG_per_Sh": 0.5,  "DC_per_Touch": 0.5,
        "FT_per_Touch": 1.0,   "Loss_per100T": 3.5,
    },

    "Half-Back": {
        "TacklesWon_90": 4.5, "Tackle_Win_Pct": 4.0, "Interception_90": 4.5,
        "Clearance_90": 5.0,  "Total_Duel": 4.0,    "Block_90": 4.0,
        "Aerial_Won_Pct": 4.5, "Ground_Duel_Pct": 4.0,
        "Recovery_90": 4.0,    "PressWon_90": 1.5,  "DribbledPast_90": 4.5,
        "Pass%": 5.0,          "LongBall_90": 5.0,  "LongBall_Acc%": 5.0,
        "OppHalf_90": 1.0,     "Cross_90": 0.0,     "xA_90": 1.0,
        "xG_Buildup_90": 5.0,  "xG_Chain_90": 1.5,  "Final_Third_90": 1.5,
        "Touch_90": 5.0,
        "Direct_Creation_90": 0.0, "npxGxA_90": 0.5, "KeyPass_90": 1.5,
        "BigChance_90": 0.0,
        "Dribbles_90": 0.5,    "Dribble_Succ%": 1.5, "Disp_per100T": 4.5,
        "Shots_90": 0.0,       "SoT_90": 0.0,       "SoT%": 0.5,
        "npxG_90": 0.0,        "G/Sh": 0.0,         "G/SoT": 0.0,
        "xA_per_KP": 1.0,      "npxG_per_Sh": 0.0,  "DC_per_Touch": 0.0,
        "FT_per_Touch": 1.5,   "Loss_per100T": 5.0,
    },

    "Deep-Lying Playmaker": {
        "TacklesWon_90": 3.5, "Tackle_Win_Pct": 3.0, "Interception_90": 4.5,
        "Clearance_90": 2.0,  "Total_Duel": 3.0,    "Block_90": 1.5,
        "Aerial_Won_Pct": 2.0, "Ground_Duel_Pct": 3.0,
        "Recovery_90": 4.0,    "PressWon_90": 2.0,  "DribbledPast_90": 3.5,
        "Pass%": 5.0,          "LongBall_90": 5.0,  "LongBall_Acc%": 5.0,
        "OppHalf_90": 2.5,     "Cross_90": 1.0,     "xA_90": 2.5,
        "xG_Buildup_90": 5.0,  "xG_Chain_90": 3.5,  "Final_Third_90": 4.0,
        "Touch_90": 5.0,
        "Direct_Creation_90": 1.5, "npxGxA_90": 1.5, "KeyPass_90": 3.0,
        "BigChance_90": 1.5,
        "Dribbles_90": 1.5,    "Dribble_Succ%": 2.0, "Disp_per100T": 4.0,
        "Shots_90": 0.5,       "SoT_90": 0.5,       "SoT%": 1.0,
        "npxG_90": 0.5,        "G/Sh": 0.5,         "G/SoT": 0.5,
        "xA_per_KP": 3.0,      "npxG_per_Sh": 1.0,  "DC_per_Touch": 1.5,
        "FT_per_Touch": 4.0,   "Loss_per100T": 5.0,
    },

    "Regista": {
        "TacklesWon_90": 1.5, "Tackle_Win_Pct": 1.5, "Interception_90": 2.5,
        "Clearance_90": 1.0,  "Total_Duel": 2.0,    "Block_90": 1.0,
        "Aerial_Won_Pct": 1.0, "Ground_Duel_Pct": 2.0,
        "Recovery_90": 2.5,    "PressWon_90": 1.0,  "DribbledPast_90": 3.0,
        "Pass%": 5.0,          "LongBall_90": 5.0,  "LongBall_Acc%": 5.0,
        "OppHalf_90": 3.5,     "Cross_90": 1.0,     "xA_90": 3.5,
        "xG_Buildup_90": 5.0,  "xG_Chain_90": 4.0,  "Final_Third_90": 5.0,
        "Touch_90": 5.0,
        "Direct_Creation_90": 2.5, "npxGxA_90": 2.5, "KeyPass_90": 3.5,
        "BigChance_90": 2.5,
        "Dribbles_90": 2.0,    "Dribble_Succ%": 2.5, "Disp_per100T": 4.5,
        "Shots_90": 0.5,       "SoT_90": 0.5,       "SoT%": 1.0,
        "npxG_90": 0.5,        "G/Sh": 0.5,         "G/SoT": 0.5,
        "xA_per_KP": 4.5,      "npxG_per_Sh": 1.0,  "DC_per_Touch": 2.5,
        "FT_per_Touch": 5.0,   "Loss_per100T": 5.0,
    },

    # ── BALANCED / DYNAMIC ────────────────────────────────────────────────────

    "Carrilero": {
        "TacklesWon_90": 4.5, "Tackle_Win_Pct": 4.0, "Interception_90": 4.0,
        "Clearance_90": 2.5,  "Total_Duel": 4.0,    "Block_90": 2.5,
        "Aerial_Won_Pct": 2.5, "Ground_Duel_Pct": 4.0,
        "Recovery_90": 4.5,    "PressWon_90": 4.0,  "DribbledPast_90": 3.5,
        "Pass%": 4.5,          "LongBall_90": 2.5,  "LongBall_Acc%": 3.0,
        "OppHalf_90": 3.0,     "Cross_90": 2.0,     "xA_90": 2.0,
        "xG_Buildup_90": 4.0,  "xG_Chain_90": 3.5,  "Final_Third_90": 3.0,
        "Touch_90": 3.5,
        "Direct_Creation_90": 2.0, "npxGxA_90": 2.0, "KeyPass_90": 2.0,
        "BigChance_90": 1.0,
        "Dribbles_90": 2.0,    "Dribble_Succ%": 2.5, "Disp_per100T": 3.5,
        "Shots_90": 1.5,       "SoT_90": 1.5,       "SoT%": 1.5,
        "npxG_90": 1.0,        "G/Sh": 1.0,         "G/SoT": 1.0,
        "xA_per_KP": 2.0,      "npxG_per_Sh": 1.0,  "DC_per_Touch": 1.5,
        "FT_per_Touch": 3.0,   "Loss_per100T": 3.0,
    },

    "Central Midfielder": {
        "TacklesWon_90": 3.0, "Tackle_Win_Pct": 3.0, "Interception_90": 3.0,
        "Clearance_90": 2.5,  "Total_Duel": 3.0,    "Block_90": 2.5,
        "Aerial_Won_Pct": 2.5, "Ground_Duel_Pct": 3.0,
        "Recovery_90": 3.0,    "PressWon_90": 3.0,  "DribbledPast_90": 3.0,
        "Pass%": 3.5,          "LongBall_90": 2.5,  "LongBall_Acc%": 2.5,
        "OppHalf_90": 3.0,     "Cross_90": 2.0,     "xA_90": 3.0,
        "xG_Buildup_90": 3.0,  "xG_Chain_90": 3.0,  "Final_Third_90": 3.0,
        "Touch_90": 3.0,
        "Direct_Creation_90": 2.5, "npxGxA_90": 2.5, "KeyPass_90": 2.5,
        "BigChance_90": 2.5,
        "Dribbles_90": 2.5,    "Dribble_Succ%": 2.5, "Disp_per100T": 3.0,
        "Shots_90": 2.5,       "SoT_90": 2.5,       "SoT%": 2.5,
        "npxG_90": 2.5,        "G/Sh": 2.5,         "G/SoT": 2.5,
        "xA_per_KP": 2.5,      "npxG_per_Sh": 2.5,  "DC_per_Touch": 2.5,
        "FT_per_Touch": 2.5,   "Loss_per100T": 3.0,
    },

    "Box-to-Box": {
        "TacklesWon_90": 4.0, "Tackle_Win_Pct": 3.5, "Interception_90": 3.5,
        "Clearance_90": 2.0,  "Total_Duel": 4.0,    "Block_90": 2.5,
        "Aerial_Won_Pct": 3.5, "Ground_Duel_Pct": 3.5,
        "Recovery_90": 4.5,    "PressWon_90": 4.0,  "DribbledPast_90": 3.0,
        "Pass%": 3.0,          "LongBall_90": 2.0,  "LongBall_Acc%": 2.0,
        "OppHalf_90": 3.5,     "Cross_90": 2.0,     "xA_90": 3.5,
        "xG_Buildup_90": 3.0,  "xG_Chain_90": 4.0,  "Final_Third_90": 3.5,
        "Touch_90": 3.5,
        "Direct_Creation_90": 4.0, "npxGxA_90": 4.0, "KeyPass_90": 3.0,
        "BigChance_90": 2.5,
        "Dribbles_90": 3.5,    "Dribble_Succ%": 3.0, "Disp_per100T": 3.0,
        "Shots_90": 4.5,       "SoT_90": 4.0,       "SoT%": 3.5,
        "npxG_90": 4.5,        "G/Sh": 3.5,         "G/SoT": 3.5,
        "xA_per_KP": 3.0,      "npxG_per_Sh": 3.5,  "DC_per_Touch": 3.5,
        "FT_per_Touch": 3.5,   "Loss_per100T": 2.5,
    },

    "Segundo Volante": {
        # Defensive base: a Segundo Volante IS a #6 first; late runs are bonus
        "TacklesWon_90": 4.5, "Tackle_Win_Pct": 4.0, "Interception_90": 4.0,
        "Clearance_90": 2.5,  "Total_Duel": 4.5,    "Block_90": 3.0,
        "Aerial_Won_Pct": 4.0, "Ground_Duel_Pct": 4.5,
        "Recovery_90": 4.5,    "PressWon_90": 4.0,  "DribbledPast_90": 4.0,
        # Passing: simple, accurate, progressive
        "Pass%": 3.5,          "LongBall_90": 2.5,  "LongBall_Acc%": 3.0,
        "OppHalf_90": 3.5,     "Cross_90": 1.5,     "xA_90": 2.5,
        # Involvement: deep + Final third presence (they get there late)
        "xG_Buildup_90": 3.0,  "xG_Chain_90": 3.5,  "Final_Third_90": 3.0,
        "Touch_90": 3.0,
        # Final product: late-runner xG bump > pure creation
        "Direct_Creation_90": 3.0, "npxGxA_90": 4.0, "KeyPass_90": 2.5,
        "BigChance_90": 3.0,
        # Dribbling: not their primary tool
        "Dribbles_90": 2.5,    "Dribble_Succ%": 2.5, "Disp_per100T": 3.5,
        # Shooting: real but not striker-level — keeps Salah-type scorers out
        "Shots_90": 4.0,       "SoT_90": 4.0,       "SoT%": 3.5,
        "npxG_90": 4.5,        "G/Sh": 3.5,         "G/SoT": 3.5,
        # Efficiency: ball security + box arrival
        "xA_per_KP": 2.5,      "npxG_per_Sh": 3.5,  "DC_per_Touch": 3.0,
        "FT_per_Touch": 3.0,   "Loss_per100T": 3.5,
    },

    # ── CREATIVE / ADVANCED ───────────────────────────────────────────────────

    "Mezzala": {
        "TacklesWon_90": 2.0, "Tackle_Win_Pct": 2.0, "Interception_90": 2.0,
        "Clearance_90": 1.0,  "Total_Duel": 2.5,    "Block_90": 1.0,
        "Aerial_Won_Pct": 1.5, "Ground_Duel_Pct": 2.5,
        "Recovery_90": 2.5,    "PressWon_90": 3.0,  "DribbledPast_90": 2.0,
        "Pass%": 3.5,          "LongBall_90": 1.5,  "LongBall_Acc%": 1.5,
        "OppHalf_90": 4.5,     "Cross_90": 3.5,     "xA_90": 5.0,
        "xG_Buildup_90": 2.0,  "xG_Chain_90": 4.5,  "Final_Third_90": 5.0,
        "Touch_90": 3.5,
        "Direct_Creation_90": 5.0, "npxGxA_90": 5.0, "KeyPass_90": 4.5,
        "BigChance_90": 4.5,
        "Dribbles_90": 4.5,    "Dribble_Succ%": 4.0, "Disp_per100T": 2.0,
        "Shots_90": 4.5,       "SoT_90": 4.0,       "SoT%": 3.5,
        "npxG_90": 4.5,        "G/Sh": 3.5,         "G/SoT": 3.5,
        "xA_per_KP": 4.5,      "npxG_per_Sh": 4.0,  "DC_per_Touch": 5.0,
        "FT_per_Touch": 5.0,   "Loss_per100T": 2.0,
    },

    "Wide Midfielder": {
        "TacklesWon_90": 3.0, "Tackle_Win_Pct": 2.5, "Interception_90": 2.5,
        "Clearance_90": 1.5,  "Total_Duel": 3.0,    "Block_90": 1.5,
        "Aerial_Won_Pct": 2.0, "Ground_Duel_Pct": 3.0,
        "Recovery_90": 2.5,    "PressWon_90": 3.5,  "DribbledPast_90": 2.5,
        "Pass%": 3.0,          "LongBall_90": 2.5,  "LongBall_Acc%": 2.5,
        "OppHalf_90": 4.5,     "Cross_90": 5.0,     "xA_90": 5.0,
        "xG_Buildup_90": 2.5,  "xG_Chain_90": 4.5,  "Final_Third_90": 5.0,
        "Touch_90": 3.5,
        "Direct_Creation_90": 4.5, "npxGxA_90": 4.0, "KeyPass_90": 4.5,
        "BigChance_90": 4.0,
        "Dribbles_90": 4.5,    "Dribble_Succ%": 4.0, "Disp_per100T": 2.0,
        "Shots_90": 3.0,       "SoT_90": 3.0,       "SoT%": 2.5,
        "npxG_90": 3.0,        "G/Sh": 2.5,         "G/SoT": 2.5,
        "xA_per_KP": 4.5,      "npxG_per_Sh": 2.5,  "DC_per_Touch": 4.5,
        "FT_per_Touch": 5.0,   "Loss_per100T": 2.0,
    },

    "Advanced Playmaker": {
        "TacklesWon_90": 0.5, "Tackle_Win_Pct": 0.5, "Interception_90": 1.0,
        "Clearance_90": 0.5,  "Total_Duel": 1.0,    "Block_90": 0.5,
        "Aerial_Won_Pct": 1.0, "Ground_Duel_Pct": 1.5,
        "Recovery_90": 1.5,    "PressWon_90": 1.5,  "DribbledPast_90": 1.5,
        "Pass%": 4.0,          "LongBall_90": 1.5,  "LongBall_Acc%": 1.5,
        "OppHalf_90": 5.0,     "Cross_90": 3.0,     "xA_90": 5.0,
        "xG_Buildup_90": 2.0,  "xG_Chain_90": 5.0,  "Final_Third_90": 5.0,
        "Touch_90": 4.0,
        "Direct_Creation_90": 5.0, "npxGxA_90": 5.0, "KeyPass_90": 5.0,
        "BigChance_90": 5.0,
        "Dribbles_90": 4.0,    "Dribble_Succ%": 4.0, "Disp_per100T": 2.0,
        "Shots_90": 4.0,       "SoT_90": 4.0,       "SoT%": 3.5,
        "npxG_90": 4.0,        "G/Sh": 3.5,         "G/SoT": 3.5,
        "xA_per_KP": 5.0,      "npxG_per_Sh": 4.0,  "DC_per_Touch": 5.0,
        "FT_per_Touch": 5.0,   "Loss_per100T": 2.0,
    },

    "Trequartista": {
        "TacklesWon_90": 0.0, "Tackle_Win_Pct": 0.0, "Interception_90": 0.0,
        "Clearance_90": 0.0,  "Total_Duel": 0.0,    "Block_90": 0.0,
        "Aerial_Won_Pct": 0.5, "Ground_Duel_Pct": 0.5,
        "Recovery_90": 0.5,    "PressWon_90": 0.5,  "DribbledPast_90": 0.5,
        "Pass%": 3.5,          "LongBall_90": 1.0,  "LongBall_Acc%": 1.0,
        "OppHalf_90": 5.0,     "Cross_90": 2.0,     "xA_90": 5.0,
        "xG_Buildup_90": 1.5,  "xG_Chain_90": 5.0,  "Final_Third_90": 5.0,
        "Touch_90": 4.0,
        "Direct_Creation_90": 5.0, "npxGxA_90": 5.0, "KeyPass_90": 5.0,
        "BigChance_90": 5.0,
        "Dribbles_90": 5.0,    "Dribble_Succ%": 5.0, "Disp_per100T": 1.5,
        "Shots_90": 4.5,       "SoT_90": 4.5,       "SoT%": 4.0,
        "npxG_90": 5.0,        "G/Sh": 4.5,         "G/SoT": 4.5,
        "xA_per_KP": 5.0,      "npxG_per_Sh": 5.0,  "DC_per_Touch": 5.0,
        "FT_per_Touch": 5.0,   "Loss_per100T": 1.5,
    },
}

@st.cache_data
def apply_league_adjustment(scouts: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Per-league mean normalisation for defensive *volume* stats only.

    Pressing structure varies hugely across the Big 5 — Bundesliga produces
    more raw tackles/interceptions per match than La Liga because both teams
    press more. Without correction, that league effect leaks into every
    defensive percentile.

    Method: for each defensive volume stat, scale every player's value by
    `global_mean / league_mean` so that all 5 leagues now average to the
    same baseline. Individual deviation from the new league baseline is
    preserved — a Bundesliga DM who's elite at recoveries vs his peers
    will still rank highly, just not boosted by league inflation.

    Efficiency rates (Tackle_Win_Pct etc.) are intentionally left alone —
    your share of duels won doesn't depend on how often duels happen.
    Same for passing/shooting, which can leak smaller league effects but
    are higher-stakes for scouting and not worth correcting in a v1.

    `enabled=False` returns the input untouched, so toggling off restores
    the league-naive view byte-for-byte.
    """
    if not enabled:
        return scouts
    out = scouts.copy()
    for col in LEAGUE_ADJ_COLS:
        if col not in out.columns:
            continue
        league_mean = out.groupby("league")[col].transform("mean")
        global_mean = float(out[col].mean())
        # replace 0 → NaN avoids div-by-zero, then fillna(1) keeps any
        # league with a degenerate mean from blowing the column away
        out[col] = out[col] * (global_mean / league_mean.replace(0, np.nan)).fillna(1.0)
    return out

@st.cache_data
def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ALL_COLS:
        if col in INVERTED_STATS:
            # Invert: high raw value → low percentile
            out[f'pct_{col}'] = (1 - out[col].rank(pct=True)) * 100
        else:
            out[f'pct_{col}'] = out[col].rank(pct=True) * 100
    return out


# ─────────────────────────────────────────────────────────────────────────────
# COHORT FILTERS — let users re-rank the selected player against a tighter
# peer group (own league, U21, prospects, vets, regulars, same-age bracket).
# The target player is ALWAYS kept in the pool so bars/radar always render
# even when the player technically falls outside the cohort criterion.
# ─────────────────────────────────────────────────────────────────────────────
COHORT_KEYS = [
    "all", "same_league", "u21", "u23", "u25",
    "vets30", "regulars", "age_bracket",
]

def _age_int(x) -> int:
    """Parse '23-307' style strings to int(23). 99 / 0 fallbacks below
    keep U-cutoffs and 30+ cutoffs from accidentally including missing data."""
    try:
        return int(str(x).split("-")[0]) if pd.notna(x) else None
    except Exception:
        return None

def build_cohort_df(scouts: pd.DataFrame, player_row: pd.Series, cohort_key: str) -> pd.DataFrame:
    """Filter `scouts` down to the requested comparison cohort.

    The selected player is always kept in the pool — they may rank as an
    outlier (e.g. 100th pct as the only 28-year-old in a U21 cohort) but
    will never disappear from their own scouting report.
    """
    if cohort_key == "all":
        return scouts.copy()

    df = scouts.copy()

    if cohort_key == "same_league":
        league = player_row.get("league")
        df = df[df["league"] == league]

    elif cohort_key in ("u21", "u23", "u25", "vets30", "age_bracket"):
        df["_age"] = df["age_"].apply(_age_int)
        if   cohort_key == "u21":     df = df[df["_age"].fillna(99) <= 21]
        elif cohort_key == "u23":     df = df[df["_age"].fillna(99) <= 23]
        elif cohort_key == "u25":     df = df[df["_age"].fillna(99) <= 25]
        elif cohort_key == "vets30":  df = df[df["_age"].fillna(0)  >= 30]
        elif cohort_key == "age_bracket":
            target = _age_int(player_row.get("age_")) or 0
            df = df[df["_age"].between(target - 2, target + 2)]
        df = df.drop(columns=["_age"])

    elif cohort_key == "regulars":
        df = df[df["Playing Time_Min"] >= 1500]

    # Always re-include the focal player (avoid empty / missing target)
    target_name = player_row["player"]
    if target_name not in df["player"].values:
        df = pd.concat([df, scouts[scouts["player"] == target_name]], ignore_index=True)

    return df

def cohort_percentiles(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute pct_X columns within the filtered cohort.

    Same logic as `compute_percentiles` but applied to a subset, so the
    target player's percentiles reflect their rank inside that pool only.
    Not @st.cache_data — input df isn't hashable and the rank op is fast
    even on the full pool (~600 players × 39 stats < 30 ms).
    """
    out = cohort_df.copy()
    for col in ALL_COLS:
        if col in INVERTED_STATS:
            out[f'pct_{col}'] = (1 - out[col].rank(pct=True)) * 100
        else:
            out[f'pct_{col}'] = out[col].rank(pct=True) * 100
    return out

def get_letter_grade(score):
    if score >= 93: return "S+", "#00CFFF"
    if score >= 85: return "S",  "#87CEEB"
    if score >= 75: return "A",  "#4CAF50"
    if score >= 62: return "B",  "#8BC34A"
    if score >= 48: return "C",  "#FFC107"
    if score >= 35: return "D",  "#FF9800"
    return "F", "#FF5252"


def render_stars(grade, size=22, letter_spacing=1):
    """
    Football Manager-style 5-star rating with smooth fractional fill.
    Uses two layered <div>s: muted grey base + colour-tinted overlay clipped
    by width = grade%. No half-star unicode quirks, works in any browser.
    """
    pct = max(0.0, min(100.0, float(grade)))
    color = (
        "#00CFFF" if grade >= 90 else
        "#FFD54A" if grade >= 75 else
        "#FFC107" if grade >= 60 else
        "#FF9800" if grade >= 45 else
        "#FF5252"
    )
    return (
        f'<div style="position:relative;display:inline-block;font-size:{size}px;'
        f'line-height:1;letter-spacing:{letter_spacing}px;font-family:Arial,sans-serif;">'
        f'<div style="color:#2a2a2a;">★★★★★</div>'
        f'<div style="position:absolute;top:0;left:0;color:{color};'
        f'width:{pct}%;overflow:hidden;white-space:nowrap;'
        f'text-shadow:0 0 8px {color}66;">★★★★★</div>'
        f'</div>'
    )


def compute_role_grade(row, weights):
    """
    Direction-aware role grading — stat-level resolution.

    Slider 0  → "I actively don't want this" — elite values HURT the grade
    Slider 2.5 → neutral — stat is ignored entirely
    Slider 5  → "I want this" — elite values HELP the grade

    Per stat:
        signed_w   = (slider − 2.5) / 2.5    ∈ [−1, +1]
        importance = |signed_w|              ∈ [ 0, +1]
        oriented   = pct  if signed_w ≥ 0  else  100 − pct

    Per category fit:
        cat_fit = Σ(oriented · importance) / Σ(importance)
                  — only stats you actually care about contribute,
                    and they contribute proportional to how much.
                  — neutral sliders do NOT drag the score toward 50.

    Final grade:
        Importance-weighted mean of cat_fit, equalised across categories
        (each category's pull = mean importance of its stats, so smaller
        categories aren't drowned by larger ones).

    cat_scores is kept as raw category percentile means (used by the radar,
    which shows the player's strengths regardless of role bias).
    """
    categories  = list(dict.fromkeys(s[2] for s in STATS))
    cat_scores  = {}    # raw percentile means → for the radar
    cat_weights = {}    # mean slider value    → for display
    cat_fit     = {}    # direction-aware fit  → drives the grade
    cat_pull    = {}    # how much this category influences the grade

    for cat in categories:
        cat_stats = [s[0] for s in STATS if s[2] == cat]
        pcts = np.array([float(row[f'pct_{c}']) for c in cat_stats])
        ws   = np.array([float(weights.get(c, 2.5)) for c in cat_stats])

        cat_scores[cat]  = float(pcts.mean())
        cat_weights[cat] = float(ws.mean())

        signed   = (ws - 2.5) / 2.5
        imp      = np.abs(signed)
        oriented = np.where(signed >= 0, pcts, 100 - pcts)

        if imp.sum() > 1e-3:
            cat_fit[cat]  = float((oriented * imp).sum() / imp.sum())
            cat_pull[cat] = float(imp.mean())   # equalises across categories
        else:
            cat_fit[cat]  = float(pcts.mean())  # no opinion → fall back
            cat_pull[cat] = 0.0

    total_pull = sum(cat_pull.values())
    if total_pull > 1e-3:
        grade = sum(cat_fit[c] * cat_pull[c] for c in categories) / total_pull
    else:
        # all sliders neutral → no role signal → raw average of strengths
        grade = float(np.mean(list(cat_scores.values())))

    return float(np.clip(grade, 0, 100)), cat_scores, cat_weights


def make_radar(cat_scores):


    # Translate then strip the leading emoji so radar reads "Defensive" or
    # "Phòng Ngự" depending on the active language.
    labels = [cat_label(c).split(" ", 1)[-1] for c in cat_scores]
    values = list(cat_scores.values())
    # close the polygon
    labels += [labels[0]]
    values += [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels,
        fill="toself",
        fillcolor="rgba(100, 180, 255, 0.12)",
        line=dict(color="#64B4FF", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=8, color="#666"),
                gridcolor="#2a2a2a", linecolor="#2a2a2a",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#ccc"),
                gridcolor="#2a2a2a", linecolor="#333",
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=50, b=50),
        height=360,
    )
    return fig


from scipy.spatial.distance import cdist


def get_similar_players(df_data, target_player_name, base_features, n=8,
                        user_weights=None, role_blend=0.0):
    """
    Similarity in percentile space, with two football-aware tweaks:

      1. Category-equalised weights — each of the 7 categories contributes the
         same total weight regardless of how many stats it contains. Without
         this, categories with more stats (Defensive=11, Shooting=7) dominate
         the distance and a creative #10 gets matched on his tackling profile.

      2. Optional role bias (role_blend ∈ [0,1]) — blends in the user's slider
         importances |w-2.5|/2.5 so "similar as a Regista" differs from
         "similar as an Anchor". 0 = pure shape, 1 = pure role-relevant shape.
    """
    pct_cols = [f'pct_{c}' for c in base_features if f'pct_{c}' in df_data.columns]
    feats    = [c for c in base_features if f'pct_{c}' in df_data.columns]

    target_row = df_data[df_data['player'] == target_player_name]
    if target_row.empty:
        return pd.DataFrame()

    # ── weights ───────────────────────────────────────────────────────────
    cat_of     = {s[0]: s[2] for s in STATS}
    cat_counts = {}
    for c in feats:
        cat_counts[cat_of[c]] = cat_counts.get(cat_of[c], 0) + 1
    cat_eq = np.array([1.0 / cat_counts[cat_of[c]] for c in feats])
    cat_eq = cat_eq * len(feats) / cat_eq.sum()

    if user_weights is not None and role_blend > 0:
        imp = np.array([abs(user_weights.get(c, 2.5) - 2.5) / 2.5 for c in feats])
        if imp.sum() > 0.01:
            imp = imp * len(feats) / imp.sum()
            w   = (1 - role_blend) * cat_eq + role_blend * imp
        else:
            w = cat_eq
    else:
        w = cat_eq

    # ── distance ──────────────────────────────────────────────────────────
    data       = df_data[pct_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    target_vec = data[df_data.index.get_loc(target_row.index[0])]

    # weighted mean absolute percentile difference → similarity in 0..100
    mean_diff  = (np.abs(data - target_vec) * w).sum(axis=1) / w.sum()
    similarity = 100 - mean_diff

    out = df_data.copy()
    out['similarity_score'] = similarity
    out = out[out['player'] != target_player_name].sort_values(
        'similarity_score', ascending=False
    )
    return out.head(n)

# League pressing-strength adjustment — toggled from the sidebar. Default
# OFF so the canonical raw view matches what casual users will share.
# We read from session_state here (not the widget) because the sidebar is
# rendered later in the script — Streamlit reruns on toggle so this stays
# in sync after the user flips it.
if "league_adj" not in st.session_state:
    st.session_state.league_adj = False
scouts = apply_league_adjustment(scouts, st.session_state.league_adj)

df_pct = compute_percentiles(scouts)



# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────

# Language selector — sits at the very top so users see it before anything else
lang_choice = st.sidebar.selectbox(
    t("lang_label"),
    list(LANG_OPTIONS.keys()),
    index=list(LANG_OPTIONS.values()).index(_lang()),
    key="lang_selector",
)
_new_lang = LANG_OPTIONS[lang_choice]
if _new_lang != st.session_state.lang:
    st.session_state.lang = _new_lang
    st.rerun()

# Custom inline-SVG wordmark — stylised "scouting target on a pitch" mark.
# Replaces the plain emoji+text title for a more pro/FM-Datahub feel.
_WORDMARK_NAME = "CT's Scouting" if _lang() == "en" else "Chỗ Tuyển Trạch của CT"
_WORDMARK_TAG  = "Big-5 Midfielder Hub" if _lang() == "en" else "Tiền Vệ 5 Giải Đấu Lớn"
st.sidebar.markdown(
    f'''
    <div class="ct-wordmark">
        <svg width="34" height="34" viewBox="0 0 34 34" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <!-- Pitch outline -->
            <rect x="2" y="6" width="30" height="22" rx="2"
                  stroke="#5DA5E8" stroke-width="1.4" fill="rgba(93,165,232,0.06)"/>
            <!-- Centre line -->
            <line x1="17" y1="6" x2="17" y2="28"
                  stroke="#5DA5E8" stroke-width="1.2" opacity="0.7"/>
            <!-- Centre circle -->
            <circle cx="17" cy="17" r="3.5"
                    stroke="#5DA5E8" stroke-width="1.2" fill="none" opacity="0.7"/>
            <!-- Penalty boxes -->
            <rect x="2" y="11" width="4.5" height="12"
                  stroke="#5DA5E8" stroke-width="1.2" fill="none" opacity="0.7"/>
            <rect x="27.5" y="11" width="4.5" height="12"
                  stroke="#5DA5E8" stroke-width="1.2" fill="none" opacity="0.7"/>
            <!-- Scouting target dot -->
            <circle cx="17" cy="17" r="1.6" fill="#5DA5E8"/>
        </svg>
        <div class="ct-wordmark-text">
            <span class="ct-wordmark-name">{_WORDMARK_NAME}</span>
            <span class="ct-wordmark-tag">{_WORDMARK_TAG}</span>
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)
st.sidebar.caption(t("app_caption"))
st.sidebar.divider()

page = st.sidebar.radio(
    t("nav_label"),
    PAGE_KEYS,
    format_func=page_label,
    label_visibility="collapsed",
)

# ── ADVANCED CONTROLS ─────────────────────────────────────────────────────────
# League pressing-strength adjustment. Off by default — most casual users
# won't think to flip it, and the canonical (raw) view matches default
# screenshots people share. Power users who care can opt in.
st.sidebar.divider()
st.sidebar.toggle(
    t("league_adj_label"),
    key="league_adj",
    help=t("league_adj_help"),
)
if st.session_state.league_adj:
    st.sidebar.markdown(
        f"<div style='color:#5DA5E8;font-size:10px;text-transform:uppercase;"
        f"letter-spacing:1.5px;font-weight:700;margin-top:-8px;'>● {t('league_adj_active')}</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STAT LEADERBOARDS
# ═════════════════════════════════════════════════════════════════════════════

if page == "leaderboards":
    st.title(t("lb_title"))
    st.caption(t("lb_caption"))

    with st.expander(t("howto_header"), expanded=False):
        st.markdown(t("howto_percentile"))
        st.markdown(t("lb_howto_extra"))

    # ── FILTERS ───────────────────────────────────────────────────────────────
    with st.expander(t("filters_header"), expanded=True):
        f1, f2, f3 = st.columns(3)

        with f1:
            all_leagues = sorted(df_pct['league'].dropna().unique().tolist())
            selected_leagues = st.multiselect(
                t("filter_league"), all_leagues, default=all_leagues, key="lb_leagues"
            )

        with f2:
            min_age = int(df_pct['age_'].apply(lambda x: int(str(x).split('-')[0]) if pd.notna(x) else 0).min())
            max_age = int(df_pct['age_'].apply(lambda x: int(str(x).split('-')[0]) if pd.notna(x) else 0).max())
            age_range = st.slider(t("filter_age_range"), min_age, max_age, (min_age, max_age), key="lb_age")

        with f3:
            min_min  = int(df_pct['Playing Time_Min'].min())
            max_min  = int(df_pct['Playing Time_Min'].max())
            min_mins = st.slider(t("filter_min_mins"), min_min, max_min, 900, 90, key="lb_mins")

    # ── APPLY FILTERS ─────────────────────────────────────────────────────────
    df_lb = df_pct.copy()

    if selected_leagues:
        df_lb = df_lb[df_lb['league'].isin(selected_leagues)]

    df_lb['_age_int'] = df_lb['age_'].apply(
        lambda x: int(str(x).split('-')[0]) if pd.notna(x) else 0
    )
    df_lb = df_lb[
        (df_lb['_age_int'] >= age_range[0]) &
        (df_lb['_age_int'] <= age_range[1]) &
        (df_lb['Playing Time_Min'] >= min_mins)
    ]

    st.caption(t("lb_showing", n=len(df_lb)))

    if df_lb.empty:
        st.warning(t("lb_no_results"))
    else:
        # ── TABS BY SECTION ───────────────────────────────────────────────────
        sections      = list(dict.fromkeys(s[2] for s in STATS))
        section_tabs  = st.tabs([cat_label(s) for s in sections])

        for tab, section in zip(section_tabs, sections):
            with tab:
                section_stats = [col for col, _, sec in STATS if sec == section]
                cols = st.columns(min(len(section_stats), 2))

                for i, col in enumerate(section_stats):
                    label = stat_label(col)
                    with cols[i % 2]:
                        st.markdown(
                            f'<span style="font-weight:700;font-size:14px;color:#fff;"'
                            f'{_tip(col)}>{label}</span>',
                            unsafe_allow_html=True,
                        )

                        top10 = (
                            df_lb[['player', 'team', 'league', col]]
                            .copy()
                            .sort_values(col, ascending=False)
                            .head(10)
                            .reset_index(drop=True)
                        )
                        top10.index += 1

                        top10.columns = [t("col_player"), t("col_team"), t("col_league"), label]
                        st.dataframe(
                            top10.style.format({label: '{:.3f}'}),
                            use_container_width=True
                        )
# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SCOUTING REPORT
# ═════════════════════════════════════════════════════════════════════════════

elif page == "scouting":
    st.title(t("sc_title"))

    with st.expander(t("howto_header"), expanded=False):
        st.markdown(t("howto_percentile"))
        st.markdown(t("howto_rolefit"))
        st.markdown(t("howto_similarity"))
        st.markdown(t("sc_workflow"))

    # ── PLAYER SELECTOR + COMPARISON COHORT ───────────────────────────────────
    player_names = df_pct.sort_values('player')['player'].tolist()
    pick_col, pool_col = st.columns([2, 1])
    with pick_col:
        selected = st.selectbox(
            t("sc_select_player"), player_names,
            help=t("sc_select_player_h"),
        )
    with pool_col:
        cohort_key = st.selectbox(
            t("cohort_label"),
            COHORT_KEYS,
            format_func=lambda k: t(f"cohort_{k}"),
            index=0,
            key="sc_cohort",
            help=t("cohort_help"),
        )

    # Build the cohort + recompute pct_X cols against just that pool. The
    # full df_pct is left untouched and continues to feed the Similar
    # Profiles widget so lookalikes are searched across the wider Big-5
    # population — only the selected player's bars/radar/role-fit shift.
    _selected_row_global = scouts[scouts['player'] == selected].iloc[0]
    cohort_df = build_cohort_df(scouts, _selected_row_global, cohort_key)
    df_for_page = cohort_percentiles(cohort_df)
    n_pool = len(cohort_df)

    if n_pool <= 1:
        st.warning(t("cohort_status_solo"))
    else:
        cohort_text = t(f"cohort_{cohort_key}").lower()
        st.caption(t("cohort_status", player=selected, n=n_pool, cohort=cohort_text))

    st.divider()

    # ── TACTICAL PROFILE + SLIDERS ────────────────────────────────────────────
    st.markdown(t("sc_tactical"))
    st.caption(t("sc_slider_intro"))

    if 'last_preset' not in st.session_state:
        st.session_state.last_preset = "Custom (Manual)"
    if 'sliders' not in st.session_state:
        st.session_state.sliders = {col: 2.5 for col in ALL_COLS}

    preset_choice = st.selectbox(
        t("sc_preset"), list(PRESETS.keys()),
        format_func=preset_label,
        key="preset_select",
        help=t("sc_preset_help"),
    )
    render_preset_card(preset_choice, accent="#64B4FF")

    if preset_choice != st.session_state.last_preset:
        st.session_state.last_preset = preset_choice

        if preset_choice == "Custom (Manual)":
            for col in ALL_COLS:
                st.session_state.sliders[col] = 2.5
                st.session_state[f"w_{col}"] = 2.5
        elif PRESETS.get(preset_choice):
            for col, val in PRESETS[preset_choice].items():
                st.session_state.sliders[col] = float(val)
                st.session_state[f"w_{col}"] = float(val)

    sections = list(dict.fromkeys(s[2] for s in STATS))
    slider_cols = st.columns(len(sections))
    weights = {}

    for i, section in enumerate(sections):
        with slider_cols[i]:
            with st.expander(cat_label(section), expanded=False):
                for col, _, sec in STATS:
                    if sec != section:
                        continue
                    weights[col] = st.slider(
                        stat_label(col), 0.0, 5.0,
                        float(st.session_state.sliders.get(col, 2.0)),
                        0.5, key=f"w_{col}",
                        help=stat_help(col),
                    )

    st.divider()

    # ── SCORING (needed for percentile breakdown) ─────────────────────────────
    # Use the cohort-recomputed percentiles for the focal player so the
    # radar / pills / attribute bars / role-fit reflect "vs cohort" ranking.
    df_scores = df_for_page

    # ── PLAYER CARD ───────────────────────────────────────────────────────────
    if selected:
        row = df_scores[df_scores['player'] == selected].iloc[0]
        grade, cat_scores, cat_weights = compute_role_grade(row, weights)
        letter, grade_color = get_letter_grade(grade)

        st.markdown("""
                <style>
                .stat-bar-bg  { background-color:#222; border-radius:4px; width:100%; height:9px; margin-bottom:10px; }
                .stat-bar-fill{ height:9px; border-radius:4px; }
                .sub-grade-box{ background-color:#111; border-radius:6px; padding:6px 4px;
                                text-align:center; border-bottom:3px solid #4CAF50; }
                </style>
            """, unsafe_allow_html=True)

    # ── HEADER ────────────────────────────────────────────────────────────
    h1, h2, h3 = st.columns([3, 1, 1])
    with h1:
        st.subheader(selected)
        gls = int(row.get('goals', 0))
        ast = int(row.get('assists', 0))
        mins = int(row.get('minutesPlayed', 0))
        age = int(str(row.get('age_', 0)).split('-')[0])
        rating = row.get('rating', 0)
        rat_color = (
            "#87CEEB" if rating >= 7.5 else
            "#4CAF50" if rating >= 7.0 else
            "#FFC107" if rating >= 6.5 else
            "#FF5252"
        )
        team_name = row.get('team', None)
        team_name = team_name if pd.notna(team_name) and team_name else 'Unknown'
        st.caption(
            f"{team_name} · {row.get('pos_', 'MF')} · "
            f"{t('sc_age_short')} {age} · {mins} {t('sc_min_short')} · "
            f"⚽ {gls} {t('card_goals')} · 🅰️ {ast} {t('card_assists')}"
        )
    with h2:
        st.markdown(f"""
            <div style="text-align:center; padding-top:6px;">
                <div style="font-size:11px;color:#888;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:8px;">{t('sc_role_fit')}</div>
                {render_stars(grade, size=30, letter_spacing=2)}
                <div style="font-size:11px;color:#666;margin-top:6px;
                            text-transform:uppercase;letter-spacing:1px;">
                    {letter} · {grade:.0f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(
            f"<div style='text-align:center;padding-top:4px;'>"
            f"<span style='font-size:10px;color:#666;'>{t('sc_sofascore')}</span><br>"
            f"<span style='font-size:22px;font-weight:bold;color:{rat_color};'>"
            f"{rating:.2f}</span></div>",
            unsafe_allow_html=True
        )

    st.write("")

    # ── RADAR  |  ATTRIBUTE BARS ──────────────────────────────────────────
    radar_col, bars_col = st.columns([1, 1])

    with radar_col:
        st.plotly_chart(make_radar(cat_scores), use_container_width=True, config={"displayModeBar": False})

        # Category grade pills below radar
        pill_cols = st.columns(len(cat_scores))
        for i, (cat, val) in enumerate(cat_scores.items()):
            ltr, col = get_letter_grade(val)
            pill_cols[i].markdown(f"""
                        <div class="sub-grade-box" style="border-color:{col};">
                            <p style="margin:0;font-size:9px;color:#666;">{cat_label(cat).split()[-1]}</p>
                            <p style="margin:0;font-size:15px;font-weight:bold;color:white;">{ltr}</p>
                            <p style="margin:0;font-size:10px;color:#888;">{val:.0f}</p>
                        </div>
                    """, unsafe_allow_html=True)


        st.divider()
        st.markdown(t("sc_similar_header"))
        st.caption(t("sc_similar_caption"))

        c1, c2 = st.columns([2, 1])
        with c1:
            role_blend = st.slider(
                t("sc_role_bias"),
                0.0, 1.0, 0.35, 0.05, key="sim_role_blend",
                help=t("sc_role_bias_help"),
            )
        with c2:
            target_age = int(str(row.get('age_', 0)).split('-')[0])
            age_window = st.slider(
                t("sc_age_window"), 0, 15, 5, 1, key="sim_age_window",
                help=t("sc_age_window_help"),
            )

        # Pre-filter pool by age window, then run similarity
        pool = df_pct.copy()
        pool['_age'] = pool['age_'].apply(
            lambda x: int(str(x).split('-')[0]) if pd.notna(x) else 0
        )
        pool = pool[
            (pool['_age'] >= target_age - age_window) &
            (pool['_age'] <= target_age + age_window)
        ]

        similar_df = get_similar_players(
            pool, selected, ALL_COLS, n=8,
            user_weights=weights, role_blend=role_blend,
        )

        if similar_df.empty:
            st.info(t("sc_no_similar"))
        else:
            for _, sim in similar_df.iterrows():
                sim_pct = float(sim['similarity_score'])
                sim_color = (
                    "#64B4FF" if sim_pct >= 88 else
                    "#4CAF50" if sim_pct >= 80 else
                    "#8BC34A" if sim_pct >= 72 else
                    "#FFC107" if sim_pct >= 64 else
                    "#FF9800"
                )
                sim_age = str(sim.get('age_', '')).split('-')[0] or '?'
                sim_team   = sim.get('team', 'Unknown') or 'Unknown'
                sim_league = sim.get('league', '') or ''

                # Same-role grade — answers "does this lookalike also fit the brief?"
                sim_grade, _, _ = compute_role_grade(sim, weights)
                sim_letter, sim_grade_color = get_letter_grade(sim_grade)

                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;
                            border-left:3px solid {sim_color};border-radius:8px;
                            padding:10px 14px;margin-bottom:8px;
                            display:flex;align-items:center;gap:14px;">
                    <div style="min-width:58px;text-align:center;">
                        <div style="font-size:20px;font-weight:800;color:{sim_color};line-height:1;">
                            {sim_pct:.0f}<span style="font-size:11px;color:#666;">%</span>
                        </div>
                        <div style="font-size:9px;color:#666;text-transform:uppercase;
                                    letter-spacing:1px;margin-top:2px;">{t('sc_match')}</div>
                    </div>
                    <div style="flex-grow:1;min-width:0;">
                        <div style="font-size:14px;font-weight:600;color:#fff;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                            {sim['player']}
                        </div>
                        <div style="font-size:11px;color:#8b949e;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                            {sim_team} · {sim_league} · {t('sc_age_short')} {sim_age}
                        </div>
                    </div>
                    <div style="text-align:center;min-width:90px;
                                border-left:1px solid #30363d;padding-left:12px;">
                        <div style="font-size:9px;color:#666;text-transform:uppercase;
                                    letter-spacing:1px;margin-bottom:4px;">{t('sc_role')}</div>
                        {render_stars(sim_grade, size=13, letter_spacing=1)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with bars_col:
        st.markdown(t("sc_attr_breakdown"))
        st.caption(t("sc_attr_caption"))

        sections_order = list(dict.fromkeys(s[2] for s in STATS))

        for section in sections_order:
            section_active = [
                col for col, _, sec in STATS
                if sec == section and weights.get(col, 0) > 0
            ]
            if not section_active:
                continue

            st.markdown(
                f"<p style='font-size:11px;color:#666;text-transform:uppercase;"
                f"letter-spacing:1px;margin-bottom:4px;margin-top:10px;'>"
                f"{cat_label(section)}</p>",
                unsafe_allow_html=True
            )

            for col in section_active:
                pct = row[f'pct_{col}']
                label = stat_label(col)
                raw = row[col]
                bar_color = (
                    "#87CEEB" if pct >= 93 else
                    "#64B4FF" if pct >= 85 else
                    "#4CAF50" if pct >= 75 else
                    "#8BC34A" if pct >= 62 else
                    "#FFC107" if pct >= 48 else
                    "#FF9800" if pct >= 35 else
                    "#FF5252"
                )
                st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;margin-bottom:-4px;">
                        <span style="font-size:12px;"{_tip(col)}>{label}</span>
                        <span style="font-size:11px;color:#666;">{raw:.2f}&nbsp;·&nbsp;
                            <span style="color:{bar_color};">{pct:.0f}th</span></span>
                    </div>
                    <div class="stat-bar-bg">
                        <div class="stat-bar-fill"
                             style="width:{pct}%;background-color:{bar_color};"></div>
                    </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PLAYER COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

elif page == "compare":
    st.title(t("cmp_title"))
    st.caption(t("cmp_caption"))

    with st.expander(t("howto_header"), expanded=False):
        st.markdown(t("howto_percentile"))
        st.markdown(t("howto_rolefit"))
        st.markdown(t("cmp_howto_similarity"))
        st.markdown(t("cmp_howto_diff"))

    PA_COLOR = "#64B4FF"   # Player A — blue
    PB_COLOR = "#FF9F43"   # Player B — orange
    DIM      = "#3a3a3a"

    # ── PLAYER SELECTORS ──────────────────────────────────────────────────────
    player_names = df_pct.sort_values('player')['player'].tolist()
    sel_a, sel_b, sel_p = st.columns([1, 1, 1])
    with sel_a:
        player_a = st.selectbox(
            t("cmp_player_a"), player_names, index=0, key="cmp_a",
            help=t("cmp_player_a_help"),
        )
    with sel_b:
        default_b = 1 if len(player_names) > 1 and player_names[1] != player_a else 2
        player_b = st.selectbox(
            t("cmp_player_b"), player_names, index=min(default_b, len(player_names) - 1),
            key="cmp_b",
            help=t("cmp_player_b_help"),
        )
    with sel_p:
        cmp_preset = st.selectbox(
            t("cmp_lens"), list(PRESETS.keys()),
            format_func=preset_label,
            index=0, key="cmp_preset",
            help=t("cmp_lens_help"),
        )

    render_preset_card(cmp_preset, accent="#9B7FE8")

    if player_a == player_b:
        st.warning(t("cmp_pick_diff"))
    else:
        row_a = df_pct[df_pct['player'] == player_a].iloc[0]
        row_b = df_pct[df_pct['player'] == player_b].iloc[0]

        # Build weights from preset (or all-neutral if Custom)
        cmp_weights = {col: 2.5 for col in ALL_COLS}
        if cmp_preset != "Custom (Manual)" and PRESETS.get(cmp_preset):
            for col, val in PRESETS[cmp_preset].items():
                cmp_weights[col] = float(val)

        grade_a, cat_scores_a, _ = compute_role_grade(row_a, cmp_weights)
        grade_b, cat_scores_b, _ = compute_role_grade(row_b, cmp_weights)

        st.divider()

        # ── BIO HEADERS (mirroring Scouting Report) ───────────────────────────
        bio_a, bio_b = st.columns(2)
        for bio_col, p_row, p_grade, p_color in [
            (bio_a, row_a, grade_a, PA_COLOR),
            (bio_b, row_b, grade_b, PB_COLOR),
        ]:
            with bio_col:
                gls = int(p_row.get('goals', 0))
                ast = int(p_row.get('assists', 0))
                mins = int(p_row.get('minutesPlayed', 0))
                age = int(str(p_row.get('age_', 0)).split('-')[0])
                rating = float(p_row.get('rating', 0) or 0)
                team = p_row.get('team', '?') or '?'
                league = p_row.get('league', '') or ''
                stars_html = render_stars(p_grade, size=22, letter_spacing=2)
                lens_label = (t("cmp_neutral_lens")
                              if cmp_preset == "Custom (Manual)"
                              else preset_label(cmp_preset))
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;'
                    f'border-top:3px solid {p_color};border-radius:10px;'
                    f'padding:14px 18px;">'
                    f'<div style="font-size:20px;font-weight:700;color:#fff;">{p_row["player"]}</div>'
                    f'<div style="font-size:11px;color:#8b949e;margin-top:2px;">'
                    f'{team} · {league} · {t("sc_age_short")} {age} · {mins} {t("sc_min_short")}</div>'
                    f'<div style="font-size:11px;color:#8b949e;margin-top:2px;">'
                    f'⚽ {gls}{t("card_goals")} · 🅰️ {ast}{t("card_assists")} · ★ {t("sc_sofascore")} {rating:.2f}</div>'
                    f'<div style="margin-top:12px;">{stars_html}</div>'
                    f'<div style="font-size:10px;color:#666;margin-top:6px;'
                    f'text-transform:uppercase;letter-spacing:1px;">'
                    f'{t("cmp_role_fit_label", preset=lens_label, grade=f"{p_grade:.0f}")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.write("")

        # ── HEAD-TO-HEAD TALLY ────────────────────────────────────────────────
        wins_a = wins_b = ties = 0
        gap_total = 0.0
        diffs = []   # (col, a_pct - b_pct, a_pct, b_pct, raw_a, raw_b)
        for col, _, _ in STATS:
            pa = float(row_a[f'pct_{col}'])
            pb = float(row_b[f'pct_{col}'])
            if   pa > pb: wins_a += 1
            elif pb > pa: wins_b += 1
            else:         ties   += 1
            gap_total += abs(pa - pb)
            diffs.append((col, pa - pb, pa, pb, float(row_a[col]), float(row_b[col])))
        n = len(STATS)
        avg_gap = gap_total / n

        # Style similarity (matches the Manhattan-percentile metric used elsewhere)
        pct_cols = [f'pct_{c}' for c in ALL_COLS]
        a_vec = row_a[pct_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        b_vec = row_b[pct_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        style_sim = 100 - float(np.abs(a_vec - b_vec).mean())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("cmp_ahead", name=player_a), t("cmp_ahead_val", w=wins_a, n=n))
        m2.metric(t("cmp_ahead", name=player_b), t("cmp_ahead_val", w=wins_b, n=n))
        m3.metric(t("cmp_ties_gap"),             t("cmp_ties_gap_val", ties=ties, gap=f"{avg_gap:.1f}"))
        m4.metric(t("cmp_style_sim"),            f"{style_sim:.0f}%")

        st.divider()

        # ── BIGGEST DIFFERENTIATORS ───────────────────────────────────────────
        st.markdown(t("cmp_diff_header"))
        st.caption(t("cmp_diff_caption"))

        top_a = sorted(diffs, key=lambda x: x[1], reverse=True)[:5]
        top_b = sorted(diffs, key=lambda x: x[1])[:5]

        diff_a_col, diff_b_col = st.columns(2)
        for diff_col, p_color, p_name, top in [
            (diff_a_col, PA_COLOR, player_a, top_a),
            (diff_b_col, PB_COLOR, player_b, top_b),
        ]:
            with diff_col:
                st.markdown(
                    f'<div style="font-size:12px;color:{p_color};font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">'
                    f'▲ {p_name}</div>',
                    unsafe_allow_html=True,
                )
                for col_key, gap, pa, pb, ra, rb in top:
                    abs_gap = abs(gap)
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'background:#161b22;border:1px solid #30363d;border-radius:6px;'
                        f'padding:6px 10px;margin-bottom:4px;font-size:12px;">'
                        f'<span style="color:#ccc;"{_tip(col_key)}>{stat_label(col_key)}</span>'
                        f'<span style="color:{p_color};font-weight:600;">'
                        f'{t("cmp_diff_gap", gap=f"{abs_gap:.0f}", pa=f"{pa:.0f}", pb=f"{pb:.0f}")}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ── OVERLAPPING RADAR ─────────────────────────────────────────────────
        st.markdown(t("cmp_radar_header"))

        labels = [cat_label(c).split(" ", 1)[-1] for c in cat_scores_a]
        a_vals = list(cat_scores_a.values())
        b_vals = list(cat_scores_b.values())
        labels.append(labels[0]); a_vals.append(a_vals[0]); b_vals.append(b_vals[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=a_vals, theta=labels, fill="toself",
            fillcolor="rgba(100, 180, 255, 0.18)",
            line=dict(color=PA_COLOR, width=2),
            name=player_a,
        ))
        fig.add_trace(go.Scatterpolar(
            r=b_vals, theta=labels, fill="toself",
            fillcolor="rgba(255, 159, 67, 0.18)",
            line=dict(color=PB_COLOR, width=2),
            name=player_b,
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickfont=dict(size=8, color="#666"),
                    gridcolor="#2a2a2a", linecolor="#2a2a2a",
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color="#ccc"),
                    gridcolor="#2a2a2a", linecolor="#333",
                ),
            ),
            showlegend=True,
            legend=dict(
                orientation="h", y=-0.08, x=0.5, xanchor="center",
                font=dict(color="#ccc", size=12),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=60, t=20, b=40),
            height=440,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.divider()

        # ── STAT-BY-STAT MIRROR BARS ──────────────────────────────────────────
        st.markdown(t("cmp_breakdown_header"))
        st.caption(t("cmp_breakdown_caption"))

        sections = list(dict.fromkeys(s[2] for s in STATS))
        for section in sections:
            section_cols = [c for c, _, sec in STATS if sec == section]

            # Mini section tally
            sec_a = sum(1 for c in section_cols
                        if float(row_a[f'pct_{c}']) > float(row_b[f'pct_{c}']))
            sec_b = sum(1 for c in section_cols
                        if float(row_b[f'pct_{c}']) > float(row_a[f'pct_{c}']))

            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-top:18px;margin-bottom:10px;">'
                f'<span style="font-size:12px;font-weight:700;color:#fff;'
                f'text-transform:uppercase;letter-spacing:1.5px;">{cat_label(section)}</span>'
                f'<span style="font-size:11px;color:#666;">'
                f'<span style="color:{PA_COLOR};">{sec_a}</span>'
                f' &nbsp;–&nbsp; '
                f'<span style="color:{PB_COLOR};">{sec_b}</span>'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            for col in section_cols:
                label = stat_label(col)
                pa = float(row_a[f'pct_{col}'])
                pb = float(row_b[f'pct_{col}'])
                ra = float(row_a[col])
                rb = float(row_b[col])

                a_better = pa > pb
                b_better = pb > pa

                a_bar  = PA_COLOR if not b_better else DIM
                b_bar  = PB_COLOR if not a_better else DIM
                a_text = PA_COLOR if not b_better else "#555"
                b_text = PB_COLOR if not a_better else "#555"
                a_op   = "1.0" if not b_better else "0.4"
                b_op   = "1.0" if not a_better else "0.4"

                st.markdown(
                    f'<div style="display:grid;grid-template-columns:1fr 200px 1fr;'
                    f'align-items:center;gap:14px;margin-bottom:5px;">'

                    # Left: Player A — bar grows right→left toward label
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<span style="font-size:11px;color:#666;min-width:38px;text-align:left;">{ra:.2f}</span>'
                    f'<div style="flex-grow:1;height:9px;background:#1a1a1a;border-radius:4px;position:relative;">'
                    f'<div style="position:absolute;right:0;top:0;height:100%;width:{pa}%;'
                    f'background:{a_bar};opacity:{a_op};border-radius:4px;"></div>'
                    f'</div>'
                    f'<span style="font-size:12px;font-weight:700;color:{a_text};'
                    f'opacity:{a_op};min-width:26px;text-align:right;">{pa:.0f}</span>'
                    f'</div>'

                    # Centre label
                    f'<div style="text-align:center;font-size:11px;color:#bbb;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"'
                    f'{_tip(col)}>{label}</div>'

                    # Right: Player B — bar grows left→right
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<span style="font-size:12px;font-weight:700;color:{b_text};'
                    f'opacity:{b_op};min-width:26px;">{pb:.0f}</span>'
                    f'<div style="flex-grow:1;height:9px;background:#1a1a1a;border-radius:4px;position:relative;">'
                    f'<div style="position:absolute;left:0;top:0;height:100%;width:{pb}%;'
                    f'background:{b_bar};opacity:{b_op};border-radius:4px;"></div>'
                    f'</div>'
                    f'<span style="font-size:11px;color:#666;min-width:38px;">{rb:.2f}</span>'
                    f'</div>'

                    f'</div>',
                    unsafe_allow_html=True,
                )
