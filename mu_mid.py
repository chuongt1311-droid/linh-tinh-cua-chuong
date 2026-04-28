import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="MU Scouting — Casemiro Replacement",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ CT's Scouting Dashboard")
st.subheader("WHO CAN REPLACE CASEMIRO? — Transfer Targets Analysis 2025-26(based on rumours)")



# ============================================================
# SECTION 2 — LOAD THE TRANSFER TARGETS DATASET
# ============================================================
@st.cache_data
def load_scouts():
    scouts = pd.read_csv('mid_scouting.csv')

    # Ball retention ratio
    scouts['Ball_Retention'] = (
        scouts['Touches_p90'] / scouts['Possesion_lost_90'].replace(0, np.nan)
    ).round(2).fillna(0)

    return scouts


scouts = load_scouts()



# Weight sliders
st.sidebar.header("⚖️ Stat Weights")
st.sidebar.write("Adjust how much each stat matters:")

# 1. Define the Preset Values
PRESETS = {
    "Custom (Manual)": None,
    "The Destroyer (#6)": {
        "Tackles": 5.0, "Blocks": 4.0, "Duels": 4.5, "Interceptions": 5.0, "Recoveries": 4.5, "Aerials": 4.0, "Retention": 3.0, "PassPct": 3.5, "KeyPass": 1.0
    },
    "Box-to-Box Engine (#8)": {
        "Tackles": 3.5, "Blocks": 2.5, "Duels": 3.5, "Interceptions": 3.0, "Recoveries": 5.0, "Aerials": 3.0, "Retention": 4.5, "PassPct": 4.0, "KeyPass": 3.0
    },
    "Deep-Lying Playmaker": {
        "Tackles": 2.5, "Blocks": 1.5, "Duels": 2.5, "Interceptions": 3.0, "Recoveries": 3.0, "Aerials": 2.0, "Retention": 5.0, "PassPct": 5.0, "KeyPass": 5.0
    },
    "The Casemiro": {
        "Tackles": 4.5, "Blocks": 3.5, "Duels": 4.0, "Interceptions": 4.5, "Recoveries": 4.0, "Aerials": 4.5, "Retention": 3.0, "PassPct": 3.0, "KeyPass": 2.5
    }
}

# 2. Preset Selection Box
preset_choice = st.sidebar.selectbox("🎯 Select Tactical Profile:", list(PRESETS.keys()))

# 3. Initialize slider values in session state if they don't exist
if 'sliders' not in st.session_state:
    st.session_state.sliders = {
        "Tackles": 4.0, "Blocks": 3.0, "Duels": 4.5, "Interceptions": 4.0,
        "Recoveries": 5.0, "Aerials": 3.0, "Retention": 5.0, "PassPct": 4.0, "KeyPass": 4.0
    }

# 4. Update session state values if a preset is chosen
if preset_choice != "Custom (Manual)":
    for stat, val in PRESETS[preset_choice].items():
        st.session_state.sliders[stat] = val

# 5. Helper function to handle manual slider changes
def update_slider(key):
    # This flips the selectbox back to "Custom" if you move a slider manually
    # (Optional, but keeps the UI clean)
    pass

# 6. Create the Sliders
w_tackles = st.sidebar.slider("Tackles Won", 1.0, 5.0, st.session_state.sliders["Tackles"], 0.5)
w_blocks = st.sidebar.slider("Blocks", 1.0, 5.0, st.session_state.sliders["Blocks"], 0.5)
w_duels = st.sidebar.slider("Ground Duels %", 1.0, 5.0, st.session_state.sliders["Duels"], 0.5)
w_interceptions = st.sidebar.slider("Interceptions", 1.0, 5.0, st.session_state.sliders["Interceptions"], 0.5)
w_recoveries = st.sidebar.slider("Ball Recoveries", 1.0, 5.0, st.session_state.sliders["Recoveries"], 0.5)
w_aerials = st.sidebar.slider("Aerial Duels %", 1.0, 5.0, st.session_state.sliders["Aerials"], 0.5)
w_carries = st.sidebar.slider("Ball Retention", 1.0, 5.0, st.session_state.sliders["Retention"], 0.5)
w_pass_pct = st.sidebar.slider("Pass Percentage", 1.0, 5.0, st.session_state.sliders["PassPct"], 0.5)
w_keypass = st.sidebar.slider("Key Passes", 1.0, 5.0, st.session_state.sliders["KeyPass"], 0.5)

# Keep your weight dictionaries as they were
not_weights = {
    'Tackle_90': w_tackles,
    'Blk_90': w_blocks,
    'Ground_duel_90': w_duels,
    'Interception_90': w_interceptions,
    'Recovery_90': w_recoveries,
    'KeyPass_90': w_keypass,
    'Pass%': w_pass_pct,
    'Aerial_90': w_aerials,
    'Ball_Retention': w_carries
}

weights_labels = {
    'Tackle_90': "Tackles Won per 90",
    'Blk_90': "Blocks per 90",
    'Ground_duel_90': "Ground Duel Win %",
    'Interception_90': "Interceptions per 90",
    'Recovery_90': "Recoveries per 90",
    'KeyPass_90': "Key Passes per 90",
    'Pass%': "Passing %",
    'Aerial_90': "Aerial Win %",
    'Ball_Retention': "Ball Retention Ratio",
}
# ============================================================
# SECTION 3 — HEAD-TO-HEAD PLAYER COMPARISON
# ============================================================
st.divider()
st.subheader("⚔️ Head-to-Head Comparison")

# 1. Get a sorted list of all available players
all_players = sorted(scouts['Player'].dropna().unique().tolist())

# 2. Side-by-side Dropdowns
col_search1, col_search2 = st.columns(2)

with col_search1:
    player_1 = st.selectbox("🔴 Select Player 1 (Target)", options=all_players, index=0)

with col_search2:
    # Safely default Player 2 to Casemiro if he exists, otherwise default to the second player
    default_p2_idx = all_players.index('Casemiro') if 'Casemiro' in all_players else 1
    player_2 = st.selectbox("🔵 Select Player 2 (Compare Against)", options=all_players, index=default_p2_idx)

# CSS to make metrics fit nicely
st.markdown("""
    <style>
    div[data-testid="stMetricLabel"] { font-size: 12px; }
    div[data-testid="stMetricValue"] { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

if player_1 and player_2:
    # 3. Fetch Data for both players
    p1_data = scouts[scouts['Player'] == player_1].iloc[0]
    p2_data = scouts[scouts['Player'] == player_2].iloc[0]

    # --- BIO CARDS ---
    col_bio1, col_bio2 = st.columns(2)

    with col_bio1:
        with st.container(border=True):
            st.write(f"### {player_1}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Age", int(pd.to_numeric(p1_data.get('Age', 0), errors='coerce') or 0))
            c2.metric("League", p1_data.get('Comp', 'Unknown'))
            c3.metric("Min", int(pd.to_numeric(p1_data.get('MP', 0), errors='coerce') or 0))
            c4.metric("G+A",
                      int(pd.to_numeric(p1_data.get('Gls', 0), errors='coerce') + pd.to_numeric(p1_data.get('Ast', 0),
                                                                                                errors='coerce')))

    with col_bio2:
        with st.container(border=True):
            st.write(f"### {player_2}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Age", int(pd.to_numeric(p2_data.get('Age', 0), errors='coerce') or 0))
            c2.metric("League", p2_data.get('Comp', 'Unknown'))
            c3.metric("Min", int(pd.to_numeric(p2_data.get('MP', 0), errors='coerce') or 0))
            c4.metric("G+A",
                      int(pd.to_numeric(p2_data.get('Gls', 0), errors='coerce') + pd.to_numeric(p2_data.get('Ast', 0),
                                                                                                errors='coerce')))
# ============================================================
# SECTION 4.5 — PLAYER CHARACTERISTICS (WHOSCORED STYLE)
# ============================================================
st.divider()
st.subheader(f"📋 Scout's Assessment: {player_1}")
st.caption("Qualitative breakdown based on performance metrics.")

# --- Mock Data Dictionary ---
# In the future, you can pull these strings directly from your CSV if you add them!
# For now, we use a dictionary to simulate the assessment.
scout_assessments = {
    "Casemiro": {
        "Strengths": [("Aerial Duels", "Strong"), ("Blocking the ball", "Strong"), ("Tackling", "Strong")],
        "Weaknesses": [("Discipline", "Weak"), ("Concentration", "Weak")],
        "Style": ["Plays the ball off the ground often", "Indirect set-piece threat", "Commits fouls often","Likes to tackle", "Likes to play long balls"]
    },
    "Adam Wharton": {
        "Strengths": [("Through balls", "Strong"), ("Taking set-pieces", "Strong"), ("Key passes", "Strong")],
        "Weaknesses": [("Passing", "Weak")],
        "Style": ["Likes to shoot from distance", "Likes to play long balls", "Plays the ball off the ground often", "Likes to tackle"]
    },
    "Carlos Baleba": {
        "Strengths": [("Ball interception", "Very Strong"), ("Dribbling", "Strong"), ("Aerial Duels", "Strong")],
        "Weaknesses": [],
        "Style": ["Likes to shoot from distance", "Likes to dribble", "Likes to play long balls"]
    },
    "Elliot Anderson": {
        "Strengths": [("Dribbling", "Very Strong"), ('Aerial Duels', "Strong"), ("Taking set-pieces", "Strong"), ("Tackling", "Strong")],
        "Weaknesses": [],
        "Style": ["Get fouled often", "Likes to dribble", "Likes to play long balls", "Likes to tackle", "Commits fouls often"]
    },
    "Angelo Stiller": {
        "Strengths": [("Passing", "Very Strong"), ("Key passes", "Strong"), ("Through balls", "Strong"), ("Taking set-pieces", "Strong")],
        "Weaknesses": [("Aerial Duels", "Weak")],
        "Style": [("Plays the ball off the ground often")]
    },
    "Sandro Tonali": {
        "Strengths": [("Concentration", "Strong"), ("Blocking the ball", "Strong")],
        "Weaknesses": [("Aerial Duels", "Weak"), ("Tackling", "Weak")],
        "Style": ["Likes to shoot from distance", "Does not dive into tackles"]
    },
    "André": {
        "Strengths": [("Passing", "Strong"), ("Tackling", "Strong"), ("Ball interception", "Strong")],
        "Weaknesses": [("Aerial Duels", "Weak")],
        "Style": ["Likes to shoot from distance", "Likes to tackle"]
    },
    "Aurélien Tchouaméni": {
        "Strengths": [("Aerial Duels", "Very Strong"), ("Passing", "Strong"), ("Ball interception", "Strong"), ("Concentration", "Strong"), ("Blocking the ball", "Strong")],
        "Weaknesses": [],
        "Style": ["Likes to play short passes"]
    },
    "Richard Ríos": {
        "Strengths": [("Key passes", "Strong"), ("Through balls", "Strong")],
        "Weaknesses": [],
        "Style": ["Gets fouled often", "Likes to play long balls", "Likes to do layoffs", "Likes to dribble", "Commits fouls often"]
    },
    "Éderson": {
        "Strengths": [("Passing", "Strong"), ("Aerial Duels", "Strong"), ("Concentration", "Strong")],
        "Weaknesses": [],
        "Style": ["Likes to play long balls"]
    },
}

# Get the assessment for the selected player (defaulting to empty if not in our mock dict)
assessment = scout_assessments.get(player_1, {"Strengths": [], "Weaknesses": [], "Style": []})

col_str, col_weak, col_style = st.columns(3)

with col_str:
    st.markdown("#### Strengths")
    if assessment["Strengths"]:
        for trait, rating in assessment["Strengths"]:
            # Green circle for strengths
            st.markdown(f"🟢 **{trait}** &nbsp;—&nbsp; *{rating}*")
    else:
        st.write("Player has no significant strengths.")

with col_weak:
    st.markdown("#### Weaknesses")
    if assessment["Weaknesses"]:
        for trait, rating in assessment["Weaknesses"]:
            # Red circle for weaknesses
            st.markdown(f"🔴 **{trait}** &nbsp;—&nbsp; *{rating}*")
    else:
        st.write("Player has no significant weaknesses.")

with col_style:
    st.markdown("#### Style of Play")
    if assessment["Style"]:
        for style in assessment["Style"]:
            # Blue diamond for playstyle
            st.markdown(f"🔹 {style}")
    else:
        st.write("No significant style of play.")

    # --- CHARTS ---
st.write(f"### Visual Breakdown")
cols = st.columns(3)

for i, (stat, label) in enumerate(weights_labels.items()):
    p1_val = pd.to_numeric(str(p1_data.get(stat, 0)).replace(',', '.'), errors='coerce')
    p2_val = pd.to_numeric(str(p2_data.get(stat, 0)).replace(',', '.'), errors='coerce')

    chart_df = pd.DataFrame({
        "Player": [player_1, player_2],
        "Value": [p1_val, p2_val]
    })

    fig = px.bar(
        chart_df, x="Value", y="Player", orientation='h',
        color="Player",
        color_discrete_map={player_1: "#003366", player_2: "#ADD8E6"},
        height=150
    )

    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title=None, yaxis_title=None)

    with cols[i % 3]:
        st.write(f"**{label}**")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- DETAILED STATS (With Deltas) ---
st.write(f"### The Numbers: {player_1} vs {player_2}")
st.info(f"The small green/red numbers show how much better or worse **{player_1}** is compared to **{player_2}**.")

m1, m2, m3 = st.columns(3)
all_metric_cols = [m1, m2, m3]

for i, (stat, label) in enumerate(weights_labels.items()):
    p1_val = pd.to_numeric(str(p1_data.get(stat, 0)).replace(',', '.'), errors='coerce')
    p2_val = pd.to_numeric(str(p2_data.get(stat, 0)).replace(',', '.'), errors='coerce')

    # Calculate the difference for the Streamlit delta indicator
    diff = p1_val - p2_val

    all_metric_cols[i % 3].metric(
        label=label,
        value=f"{p1_val:.2f}",
        delta=f"{diff:.2f}"
    )


# ============================================================
# SECTION 4 — VISUAL COMPARISON (RADAR CHART)
# ============================================================
st.divider()
st.write(f"### 📊 Performance Radar: {player_1} vs {player_2}")
st.caption(f"Showing percentile rankings (using Casemiro as the baseline standard). The outer edge (100) represents the best in the scouting pool.")

# 1. Define the stats for the radar
radar_stats = list(weights_labels.keys())
radar_labels = [
    "Tackles", "Blocks", "Ground Duels", "Interceptions",
    "Recoveries", "Key Passes", "Pass %", "Aerial Duels", "Retention"
]

# 2. Calculate Percentiles relative to the whole dataset
# This turns raw numbers into a 0-100 score based on your CSV
scouts_pct = scouts.copy()
for stat in radar_stats:
    # Handle cleaning just in case
    scouts_pct[stat] = pd.to_numeric(scouts_pct[stat].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    # Rank them 0 to 100
    scouts_pct[stat] = (scouts_pct[stat].rank(pct=True) * 100).round(1)

# 3. Extract data for the two selected players
p1_values = scouts_pct[scouts_pct['Player'] == player_1][radar_stats].values.flatten().tolist()
p2_values = scouts_pct[scouts_pct['Player'] == player_2][radar_stats].values.flatten().tolist()

# 4. "Close" the radar loop (Plotly needs the first value repeated at the end)
p1_values += p1_values[:1]
p2_values += p2_values[:1]
radar_labels_closed = radar_labels + [radar_labels[0]]

# 5. Create the Radar Chart
fig_radar = go.Figure()

# Selected Target (e.g., Wharton, Kone)
fig_radar.add_trace(go.Scatterpolar(
    r=p1_values,
    theta=radar_labels_closed,
    fill='toself',
    name=player_1,
    line_color='#DA291C', # MU Red
    fillcolor='rgba(218, 41, 28, 0.3)',
    hoverinfo='name+r'
))

# The Standard (Casemiro)
fig_radar.add_trace(go.Scatterpolar(
    r=p2_values,
    theta=radar_labels_closed,
    fill='toself',
    name=player_2,
    line_color='#FFD700', # Gold"
    fillcolor='rgba(255, 215, 0, 0.4)',
    hoverinfo='name+r'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickfont=dict(size=10)
        ),
        angularaxis=dict(
            tickfont=dict(size=12, color="white")
        )
    ),
    showlegend=True,
    height=500,
    margin=dict(l=80, r=80, t=20, b=20)
)

st.plotly_chart(fig_radar, use_container_width=True)


# ============================================================
# SECTION 5 — TRANSFER TARGETS LEADERBOARD 1: Similarity
# ============================================================
st.divider()
st.subheader("🎯 Transfer Targets — Leaderboard 1: Most Similar to Casemiro")
st.caption("Based on cosine similarity of defensive & passing profile vs Casemiro 2025-26")

target_sim_cols = [
    'Tackle_90', 'Blk_90', 'Ground_duel_90', 'Ball_Retention',
    'Interception_90', 'Recovery_90', 'KeyPass_90',
    'Aerial_90', 'Possesion_lost_90', 'Pass%',
]

scouts[target_sim_cols] = scouts[target_sim_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

casemiro_scout_vector = scouts[scouts['Player'] == 'Casemiro'][target_sim_cols].values
all_scout_vectors = scouts[target_sim_cols].values
scout_similarity = cosine_similarity(casemiro_scout_vector, all_scout_vectors)[0]
scouts['casemiro_similarity'] = scout_similarity.round(4)

leaderboard_similarity = scouts[['Player', 'Age', 'Comp', 'casemiro_similarity']]\
    .sort_values('casemiro_similarity', ascending=False).reset_index(drop=True)
leaderboard_similarity.index += 1

col_lb1, col_chart1 = st.columns([1, 1])

with col_lb1:
    st.dataframe(leaderboard_similarity, use_container_width=True, height=350)

with col_chart1:
    sim_sorted = leaderboard_similarity.sort_values('casemiro_similarity', ascending=True)
    fig_sim, ax_sim = plt.subplots(figsize=(7, 5))
    colors_sim = plt.cm.Blues([
        0.4 + 0.6 * (x - sim_sorted['casemiro_similarity'].min()) /
        (sim_sorted['casemiro_similarity'].max() - sim_sorted['casemiro_similarity'].min())
        for x in sim_sorted['casemiro_similarity']
    ])
    ax_sim.barh(sim_sorted['Player'], sim_sorted['casemiro_similarity'], color=colors_sim)
    ax_sim.set_xlabel('Similarity Score')
    ax_sim.set_title('Similarity to Casemiro Profile')
    ax_sim.set_xlim(sim_sorted['casemiro_similarity'].min() - 0.005,
                    sim_sorted['casemiro_similarity'].max() + 0.002)
    plt.tight_layout()
    st.pyplot(fig_sim)

# ============================================================
# SECTION 6 — TRANSFER TARGETS LEADERBOARD 2: Man Utd Fit
# ============================================================
st.divider()
st.subheader("🔴 Transfer Targets — Leaderboard 2: Best Fit for Man Utd ")
st.write("Ranking players based on the **Stat Weights** or the tactical profile you selected in the sidebar.")


def calculate_mu_score(df, weights_dict):
    score = pd.Series(0.0, index=df.index)
    for stat, weight in weights_dict.items():
        if stat in df.columns:
            # Clean data and handle string/numeric conversion
            col_data = pd.to_numeric(df[stat].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
            stat_min = col_data.min()
            stat_max = col_data.max()
            if stat_max > stat_min:
                normalized_stat = (col_data - stat_min) / (stat_max - stat_min)
                score += (normalized_stat * weight)
    return score


# 1. Calculate the raw weighted score from sliders
scouts['mu_weighted_score'] = calculate_mu_score(scouts, not_weights)

# 2. Normalize that score (0 to 1) so it can be blended with similarity
w_min = scouts['mu_weighted_score'].min()
w_max = scouts['mu_weighted_score'].max()
scouts['mu_score_norm'] = (scouts['mu_weighted_score'] - w_min) / (w_max - w_min) if w_max > w_min else 0

# 3. Apply the 70/30 Ratio Blend
# (70% Weighted Score + 30% Similarity Score)
scouts['mu_final_score'] = scouts['mu_score_norm'].round(4)

# 4. Create the Leaderboard DataFrame
leaderboard_mu = scouts[['Player', 'Age', 'Comp', 'mu_weighted_score', 'casemiro_similarity', 'mu_final_score']] \
    .sort_values('mu_final_score', ascending=False).reset_index(drop=True)
leaderboard_mu.index += 1
leaderboard_mu.columns = ['Player', 'Age', 'League', 'Weighted Score', 'Casemiro Similarity', 'Final Score']

# 5. Display Table and Plotly Chart
col_lb2, col_chart2 = st.columns([1, 1])

with col_lb2:
    st.write("**Top Rankings**")
    st.dataframe(leaderboard_mu.head(11), use_container_width=True, height=385)

with col_chart2:
    # Get top 10 for the chart and sort ascending for horizontal bar flow
    mu_chart_data = leaderboard_mu.head(10).sort_values(by='Final Score', ascending=True)

    fig_mu = px.bar(
        mu_chart_data,
        x='Final Score',
        y='Player',
        orientation='h',
        title='Best Fit',
        labels={'Final Score': 'Blended Final Score', 'Player': ''},
        color='Final Score',
        color_continuous_scale='Reds',
        template='plotly_white'
    )
    fig_mu.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_mu, use_container_width=True)

# ============================================================
# SECTION 7 — SCATTER: Score vs Similarity (Transfer Targets)
# ============================================================
st.divider()
st.subheader("📊 Transfer Targets — Score vs Similarity")

# 🛠️ FIX: Upgraded Scatter Plot to Plotly so you can hover over the dots!
fig_scatter = px.scatter(
    scouts,
    x='mu_weighted_score',
    y='casemiro_similarity',
    hover_name='Player',
    text='Player',
    title='Transfer Targets: Fit vs Casemiro Similarity',
    labels={
        'mu_weighted_score': 'Man Utd Weighted Score',
        'casemiro_similarity': 'Similarity to Casemiro'
    }
)

# Add reference lines (Averages)
fig_scatter.add_hline(y=scouts['casemiro_similarity'].mean(), line_dash="dash", line_color="gray",
                      annotation_text="Avg Similarity")
fig_scatter.add_vline(x=scouts['mu_weighted_score'].mean(), line_dash="dash", line_color="gray",
                      annotation_text="Avg Score")

# Make the dots look good
fig_scatter.update_traces(textposition='top center', marker=dict(size=10, color='crimson', opacity=0.7))
fig_scatter.update_layout(height=600)

st.plotly_chart(fig_scatter, use_container_width=True)
