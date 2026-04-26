import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(
    page_title="MU vo dich",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ CT's Scouting Dashboard")
st.subheader("WHO CAN REPLACE CASEMIRO?")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('D:\\Pybaseball\\Top5_League_Players_2017to2024_dataset.csv', sep=';')
    # --- YOUR SHARED LOGIC START ---
    # Define the lists from your snippet
    original_stats = [
        'Tackles_TklW', 'Blocks_Blocks', 'Challenges_Tkl%',
        'Int_', 'Performance_Recov', 'Progression_PrgP', 'Carries_PrgC', 'Total_Cmp%',
        'Aerial Duels_Won%', 'Performance_Fls', 'Performance_CrdY'
    ]

    counting_stats = [
        'Tackles_TklW', 'Blocks_Blocks', 'Int_',
        'Performance_Recov', 'Progression_PrgP', 'Carries_PrgC',
        'Performance_Fls', 'Performance_CrdY',
    ]

    # Clean numeric columns (Comma to Dot)
    for stat in original_stats + ['Playing Time_90s']:
        if stat in df.columns:
            df[stat] = df[stat].astype(str).str.replace(',', '.', regex=False)
            df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)

    # Calculate Per 90s
    for stat in counting_stats:
        # Avoid division by zero
        df[f'{stat}_p90'] = np.where(
            df['Playing Time_90s'] > 0,
            (df[stat] / df['Playing Time_90s']).round(3),
            0
        )
    # --- YOUR SHARED LOGIC END ---

    return df

df = load_data()

# Sidebar
st.sidebar.header("🔍 Filters")
# League filter
all_leagues = df['league'].unique().tolist()
selected_leagues = st.sidebar.multiselect(
    "Select Leagues",
    options=all_leagues,
    default=all_leagues
)
# Season filter — fixed to 2425
df_filtered = df[df['season'] == 2425].copy()
# Apply league filter
df_filtered = df_filtered[df_filtered['league'].isin(selected_leagues)]
# Age filter
df_filtered['age_'] = pd.to_numeric(df_filtered['age_'], errors='coerce')
min_age, max_age = int(df_filtered['age_'].min()), int(df_filtered['age_'].max())
age_range = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(18, 28)
)
df_filtered = df_filtered[
    (df_filtered['age_'] >= age_range[0]) &
    (df_filtered['age_'] <= age_range[1])
]
# Minimum minutes filter
min_minutes = st.sidebar.slider(
    "Minimum Minutes Played",
    min_value=0,
    max_value=3500,
    value=900,
    step=100
)

df_filtered['Playing Time_Min'] = pd.to_numeric(
    df_filtered['Playing Time_Min']
    .astype(str).str.replace(',', '.'),
    errors='coerce'
)
df_filtered = df_filtered[df_filtered['Playing Time_Min'] >= min_minutes]

# Position filter — midfielders only
df_filtered = df_filtered[df_filtered['pos_'].str.contains('MF', na=False)]

st.sidebar.write(f"Filtered Players: {len(df_filtered)}")

# Weight
st.sidebar.header("⚖️ Stat Weights (to find the one who can replace Casemiro)")
st.sidebar.write("Adjust how much each stat matters:")

w_tackles = st.sidebar.slider("Tackles Won", 0.0, 5.0, 5.0, 0.5)
w_blocks = st.sidebar.slider("Blocks", 0.0, 5.0, 5.0, 0.5)
w_duels = st.sidebar.slider("Ground Duels %", 0.0, 5.0, 5.0, 0.5)
w_interceptions = st.sidebar.slider("Interceptions", 0.0, 5.0, 4.5, 0.5)
w_recoveries = st.sidebar.slider("Ball Recoveries", 0.0, 5.0, 5.0, 0.5)
w_progressive = st.sidebar.slider("Progressive Passes", 0.0, 5.0, 4.0, 0.5)
w_aerials = st.sidebar.slider("Aerial Duels %", 0.0, 5.0, 3.5, 0.5)
w_fouls = st.sidebar.slider("Fouls (less=better)", 0.0, 5.0, 2.0, 0.5)
w_cards = st.sidebar.slider("Yellow Cards (less=better)", 0.0, 5.0, 2.0, 0.5)
w_carries = st.sidebar.slider("Ball Carrying", 0.0, 5.0, 2.0, 0.5)
w_pass_percentage = st.sidebar.slider("Pass Percentage", 0.0, 5.0, 2.0, 0.5)

not_weights = {
    'Tackles_TklW_p90'      : w_tackles,
    'Blocks_Blocks_p90'     : w_blocks,
    'Challenges_Tkl%'       : w_duels,
    'Int__p90'              : w_interceptions,
    'Performance_Recov_p90' : w_recoveries,
    'Progression_PrgP_p90'  : w_progressive,
    'Carries_PrgC_p90'      : w_carries,
    'Total_Cmp%'            : w_pass_percentage,
    'Aerial Duels_Won%'     : w_aerials,
    'Performance_Fls_p90'   : w_fouls,
    'Performance_CrdY_p90'  : w_cards
}
weights = {
    'Tackles_TklW_p90'      : "Tackles Won per 90",
    'Blocks_Blocks_p90'     : "Blocks per 90",
    'Challenges_Tkl%'       : "Duel Win %",
    'Int__p90'              : "Interceptions per 90",
    'Performance_Recov_p90' : "Recoveries per 90",
    'Progression_PrgP_p90'  : "Progressive Passes per 90",
    'Carries_PrgC_p90'      : "Progressive Carries per 90",
    'Total_Cmp%'            : "Passing %",
    'Aerial Duels_Won%'     : "Aerial Win %",
    'Performance_Fls_p90'   : "Fouls per 90",
    'Performance_CrdY_p90'  : "Yellow Cards per 90"
}

# Player Search Section
st.divider()
st.subheader("👤 Player Search & Profile (Midfielders from Top 5 Leagues in 24-25 Season)")

# Create a search bar in the sidebar or main area
search_query = st.text_input("🔍 Search Player Name", placeholder="e.g., Rodri")
if search_query:
    # Filter the main dataframe (not the filtered one) to find players across all leagues/ages
    search_results = df[df['player'].str.contains(search_query, case=False, na=False)]
    if not search_results.empty:
        # If multiple players match (e.g., "Silva"), let the user pick the specific one
        player_names = search_results['player'].unique()
        selected_player_name = st.selectbox("Select the player you're looking for:", options=player_names)
        # Get data for the selected player (using the latest season available for them)
        p_data = search_results[search_results['player'] == selected_player_name].sort_values('season', ascending=False).iloc[0]
        # --- SECTION 1: Bio Card ---
        st.markdown("""
            <style>
            div[data-testid="stMetricLabel"] { font-size: 12px; }
            div[data-testid="stMetricValue"] { font-size: 18px; }
            </style>
            """, unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3, c4, c5, c6 = st.columns([1, 2, 3, 1, 1, 1])
            c1.metric("Age", int(pd.to_numeric(p_data['age_'], errors='coerce') or 0))
            c2.metric("Club", p_data['team'])
            c3.metric("League", p_data['league'])
            c4.metric("Min", f"{int(pd.to_numeric(p_data['Playing Time_Min'], errors='coerce') or 0)}")
            c5.metric("Goals", int(pd.to_numeric(p_data.get('Performance_Gls', 0), errors='coerce')))
            c6.metric("Assists", int(pd.to_numeric(p_data.get('Performance_Ast', 0), errors='coerce')))

            # --- SECTION 2: The Comparison Chart (SIDE-BY-SIDE) ---
            st.write(f"### Stats Comparison: {selected_player_name} vs. Casemiro")
            casemiro_df = df[df['player'] == 'Casemiro'].sort_values('season', ascending=False)
            if not casemiro_df.empty:
                casemiro_data = casemiro_df.iloc[0]

                # Create 3 columns for the grid
                cols = st.columns(3)

                # Loop through every stat in your weights dictionary
                for i, (stat, weight) in enumerate(weights.items()):
                    # Get the value for both players
                    p_val = pd.to_numeric(str(p_data[stat]).replace(',', '.'), errors='coerce')
                    c_val = pd.to_numeric(str(casemiro_data[stat]).replace(',', '.'), errors='coerce')

                    # Create a tiny dataframe for this specific stat
                    stat_name = weights.get(stat, stat)
                    chart_df = pd.DataFrame({
                        "Player": [selected_player_name, "Casemiro"],
                        "Value": [p_val, c_val]
                    })

                    # Create the small plot
                    fig = px.bar(
                        chart_df,
                        x="Value",
                        y="Player",
                        orientation='h',
                        color="Player",
                        color_discrete_map={selected_player_name: "#003366", "Casemiro": "#ADD8E6"},
                        height=150  # Make it compact
                    )

                    # Styling cleanup
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title=None,
                        yaxis_title=None
                    )

                    # Display in the current column
                    with cols[i % 3]:
                        st.write(f"**{stat_name}**")
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Could not find 'Casemiro' for comparison.")

        # --- SECTION 2: The Chart (FULL WIDTH) ---
        #st.write(f"### {selected_player_name}'s ")
        #chart_data = pd.DataFrame({
            #"Stat": [weights.get(s, s) for s in weights.keys()],
            #"Value": [pd.to_numeric(str(p_data[s]).replace(',', '.'), errors='coerce') for s in weights.keys()]
        #})
        # Using a horizontal bar chart so labels are easy to read
        #st.bar_chart(chart_data.set_index("Stat"), horizontal=True, height=400)

        st.divider()

        # 2. Detailed Metrics (Dynamic 3-Column Grid)
        st.write("### Detailed Stats (Per 90)")

        # Create 3 columns
        m1, m2, m3 = st.columns(3)
        # Store them in a list so we can access them by index
        all_cols = [m1, m2, m3]

        for i, (stat, weight) in enumerate(weights.items()):
            # Use % 3 to cycle through column 0, 1, 2
            current_col = all_cols[i % 3]

            val = pd.to_numeric(str(p_data[stat]).replace(',', '.'), errors='coerce')
            label = weights.get(stat, stat)

            # Write to the specific column
            current_col.metric(label=label, value=f"{val:.2f}")

    else:
        st.error("Player not found.")
else:
    ()