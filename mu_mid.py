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

w_tackles       = st.sidebar.slider("Tackles Won",              0.0, 5.0, 4.0, 0.5)
w_blocks        = st.sidebar.slider("Blocks",                   0.0, 5.0, 3.0, 0.5)
w_duels         = st.sidebar.slider("Ground Duels %",           0.0, 5.0, 4.5, 0.5)
w_interceptions = st.sidebar.slider("Interceptions",            0.0, 5.0, 4.0, 0.5)
w_recoveries    = st.sidebar.slider("Ball Recoveries",          0.0, 5.0, 5.0, 0.5)
w_aerials       = st.sidebar.slider("Aerial Duels %",           0.0, 5.0, 3.0, 0.5)
w_carries       = st.sidebar.slider("Ball Retention",           0.0, 5.0, 5.0, 0.5)
w_pass_pct      = st.sidebar.slider("Pass Percentage",          0.0, 5.0, 4.0, 0.5)
w_keypass       = st.sidebar.slider("Key Passes",               0.0, 5.0, 4.0, 0.5)

not_weights = {
    'Tackle_90'         : w_tackles,
    'Blk_90'            : w_blocks,
    'Ground_duel_90'    : w_duels,
    'Interception_90'   : w_interceptions,
    'Recovery_90'       : w_recoveries,
    'KeyPass_90'        : w_keypass,
    'Pass%'             : w_pass_pct,
    'Aerial_90'         : w_aerials,
    'Ball_Retention'    : w_carries

}

# 🛠️ FIX 1: Updated to match EXACT columns in mid_scouting.csv
weights_labels = {
    'Tackle_90': "Tackles Won per 90",
    'Blk_90': "Blocks per 90",
    'Ground_duel_90': "Ground Duel Win %",
    'Interception_90': "Interceptions per 90",
    'Recovery_90': "Recoveries per 90",
    'KeyPass_90': "Key Passes per 90",
    'Pass%': "Passing %",
    'Aerial_90': "Aerial Win %",
    'Ball_Retention': "Ball Retention Ratio (Touches per game/Possession lost per game)",
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

    st.divider()

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
st.caption(f"Showing percentile rankings. The outer edge (100) represents the best in the scouting pool.")

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
st.subheader("🔴 Transfer Targets — Leaderboard 2: Best Fit for Man Utd (70% Weights + 30% Similarity)")
st.write("Ranking players based on the **Stat Weights** you selected in the sidebar.")


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
scouts['mu_final_score'] = (
        (scouts['mu_score_norm'] * 0.7) + (scouts['casemiro_similarity'] * 0.3)
).round(4)

# 4. Create the Leaderboard DataFrame
leaderboard_mu = scouts[['Player', 'Age', 'Comp', 'mu_weighted_score', 'casemiro_similarity', 'mu_final_score']] \
    .sort_values('mu_final_score', ascending=False).reset_index(drop=True)
leaderboard_mu.index += 1
leaderboard_mu.columns = ['Player', 'Age', 'League', 'Weighted Score', 'Casemiro Similarity', 'Final Score']

# 5. Display Table and Plotly Chart
col_lb2, col_chart2 = st.columns([1, 1])

with col_lb2:
    st.write("**Top 10 Rankings**")
    st.dataframe(leaderboard_mu.head(10), use_container_width=True, height=385)

with col_chart2:
    # Get top 10 for the chart and sort ascending for horizontal bar flow
    mu_chart_data = leaderboard_mu.head(10).sort_values(by='Final Score', ascending=True)

    fig_mu = px.bar(
        mu_chart_data,
        x='Final Score',
        y='Player',
        orientation='h',
        title='Best Fit (70% Weights / 30% Similarity)',
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

st.caption("Top right = high fit + high similarity to Casemiro — ideal targets")
