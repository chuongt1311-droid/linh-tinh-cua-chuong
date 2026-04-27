import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

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
    scouts = pd.read_csv('D:\\Pybaseball\\mid_scouting.csv')

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
st.caption("Weighted score based on Carrick's possession-based system requirements")

mu_weights = {
    'Tackle_90'        : 4,
    'Blk_90'           : 3,
    'Ground_duel_90'   : 4.5,
    'Interception_90'  : 4,
    'Recovery_90'      : 5,
    'KeyPass_90'       : 4,
    'Aerial_90'        : 3,
    'Possesion_lost_90': 1,
    'Pass%'            : 4,
    'Ball_Retention'   : 5,
}

mu_invert_stats = ['Possesion_lost_90']

# Make sure all mu_weight cols are numeric
mu_cols = list(mu_weights.keys())
scouts[mu_cols] = scouts[mu_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

mu_scores = []
for stat, weight in mu_weights.items():
    min_val = scouts[stat].min()
    max_val = scouts[stat].max()
    if max_val == min_val:
        normalized = pd.Series(0, index=scouts.index)
    elif stat in mu_invert_stats:
        normalized = 1 - (scouts[stat] - min_val) / (max_val - min_val)
    else:
        normalized = (scouts[stat] - min_val) / (max_val - min_val)
    mu_scores.append(normalized * weight)

scouts['mu_weighted_score'] = sum(mu_scores)

mu_score_min = scouts['mu_weighted_score'].min()
mu_score_max = scouts['mu_weighted_score'].max()
scouts['mu_score_norm'] = (scouts['mu_weighted_score'] - mu_score_min) / (mu_score_max - mu_score_min) \
    if mu_score_max > mu_score_min else 0

scouts['mu_final_score'] = (
    scouts['mu_score_norm'] * 0.7 + scouts['casemiro_similarity'] * 0.3
).round(4)

leaderboard_mu = scouts[['Player', 'Age', 'Comp', 'mu_weighted_score', 'casemiro_similarity', 'mu_final_score']]\
    .sort_values('mu_final_score', ascending=False).reset_index(drop=True)
leaderboard_mu.index += 1
leaderboard_mu.columns = ['Player', 'Age', 'League', 'Weighted Score', 'Casemiro Similarity', 'Final Score']

col_lb2, col_chart2 = st.columns([1, 1])

with col_lb2:
    st.dataframe(leaderboard_mu, use_container_width=True, height=350)

with col_chart2:
    mu_sorted = leaderboard_mu.sort_values('Final Score', ascending=True)
    fig_mu, ax_mu = plt.subplots(figsize=(7, 5))
    colors_mu = plt.cm.Reds([
        0.4 + 0.6 * (x - mu_sorted['Final Score'].min()) /
        (mu_sorted['Final Score'].max() - mu_sorted['Final Score'].min())
        for x in mu_sorted['Final Score']
    ])
    ax_mu.barh(mu_sorted['Player'], mu_sorted['Final Score'], color=colors_mu)
    ax_mu.set_xlabel('Final Score')
    ax_mu.set_title('Best Fit for Man Utd — Transfer Targets')
    ax_mu.set_xlim(mu_sorted['Final Score'].min() - 0.01,
                   mu_sorted['Final Score'].max() + 0.01)
    plt.tight_layout()
    st.pyplot(fig_mu)

# ============================================================
# SECTION 7 — SCATTER: Score vs Similarity (Transfer Targets)
# ============================================================
st.divider()
st.subheader("📊 Transfer Targets — Score vs Similarity")

fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
ax_scatter.scatter(
    scouts['mu_weighted_score'],
    scouts['casemiro_similarity'],
    s=100, alpha=0.7, color='crimson', edgecolors='white', linewidths=0.5
)
for _, row in scouts.iterrows():
    ax_scatter.annotate(
        row['Player'],
        (row['mu_weighted_score'], row['casemiro_similarity']),
        fontsize=8, xytext=(6, 4), textcoords='offset points'
    )
ax_scatter.set_xlabel('Man Utd Weighted Score')
ax_scatter.set_ylabel('Similarity to Casemiro')
ax_scatter.set_title('Transfer Targets: Fit vs Casemiro Similarity')
ax_scatter.axhline(y=scouts['casemiro_similarity'].mean(), color='gray',
                   linestyle='--', linewidth=0.8, alpha=0.6)
ax_scatter.axvline(x=scouts['mu_weighted_score'].mean(), color='gray',
                   linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
st.pyplot(fig_scatter)
st.caption("Top right = high fit + high similarity to Casemiro — ideal targets")
