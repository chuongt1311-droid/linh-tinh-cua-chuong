import pandas as pd
import unicodedata

# 1. Load the fixed FBref + Understat data
print("Loading FBref/Understat master...")
master_df = pd.read_csv('ultimate_scouting_2026_fixed.csv')

# 2. Load the Sofascore data we successfully scraped earlier
print("Loading Sofascore raw data...")
try:
    sofa_df = pd.read_csv('sofascore_2526_raw.csv')
except FileNotFoundError:
    print("ERROR: Please make sure 'sofascore_2526_raw.csv' is in the same folder.")
    exit()

# 3. Create the robust merge key
def create_merge_key(df, player_col='player'):
    if player_col not in df.columns:
        return df
    df['join_key'] = (
        df[player_col]
        .astype(str)
        .str.strip()                          # ← kills leading/trailing spaces in names
        .apply(lambda x: unicodedata.normalize('NFKD', x)
               .encode('ASCII', 'ignore')
               .decode('utf-8')
               .lower()
               .replace(" ", "")
               .replace("-", ""))
    )
    return df

master_df = create_merge_key(master_df, 'player')

# Note: Check if Sofascore column is 'player' or 'player_name'
sofa_player_col = 'player' if 'player' in sofa_df.columns else 'player_name'
sofa_df = create_merge_key(sofa_df, sofa_player_col)

# 4. Filter Sofascore columns (Keep only what we need to keep it clean)
#sofa_cols_to_keep = [
    #'join_key', 'rating', 'accuratePassesPercentage', 'accurateLongBalls',
    #'accurateFinalThirdPasses', 'bigChancesCreated', 'interceptions',
    #'ballRecovery', 'keyPasses', 'accurateChippedPasses', 'touches', 'aerialDuelsWon',
    #'aerialDuelsWonPercentage',
    #'groundDuelsWon',
    #'groundDuelsWonPercentage',
    #'possessionWonAttThird',
    #'possessionLost',
    #'errorLeadToShot',
    #'errorLeadToGoal',
    #'totalDuelsWon',
    #'totalDuelsWonPercentage',
    #'successfulDribbles',
    #'dribbledPast',
    #'tacklesWon'
#]
# Ensure all these exist in the df before filtering
#sofa_cols_to_keep = [c for c in sofa_cols_to_keep if c in sofa_df.columns]

# Drop duplicates just in case (e.g., a player moved teams mid-season)
sofa_subset = sofa_df.drop_duplicates(subset=['join_key'])

# 5. The Final Merge
print("Merging Sofascore data...")
final_perfect_df = master_df.merge(sofa_subset, on='join_key', how='left')

# Cleanup the join key
final_perfect_df = final_perfect_df.drop(columns=['join_key'])

# 6. Final Export
final_perfect_df.to_csv("PERFECT_scouting_data_2026.csv", index=False)
print("SUCCESS! 'PERFECT_scouting_data_2026.csv' created.")

# Quick validation
missing_longballs = final_perfect_df['accurateLongBalls'].isnull().sum()
total = len(final_perfect_df)
print(f"Stats: {total - missing_longballs} out of {total} players matched with Sofascore data.")