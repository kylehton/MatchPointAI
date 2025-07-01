### This file is to create the train test set using calculated differences between players
import tqdm
from sqlalchemy import create_engine
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Connect to your database
sql_engine = create_engine(os.getenv('POSTGRES_NEON_STRING'))
player_table = os.getenv("PLAYER_TABLE")

def load_match_history():
    print("Loading match history from CSV files...")
    
    # Get all CSV files from data/main directory
    csv_files = glob.glob('data/main/atp_matches_*.csv')
    
    all_matches = []
    for file in tqdm.tqdm(csv_files, desc="Loading match files"):
        df = pd.read_csv(file)
        all_matches.append(df)
    
    # Combine all matches
    combined_matches = pd.concat(all_matches, ignore_index=True)
    print(f"Loaded {len(combined_matches)} total matches")
    
    return combined_matches

def get_player_stats_from_db():
    """Get all player stats from the database"""
    print("Loading player stats from database...")
    
    query = f"SELECT * FROM {player_table}"
    player_stats = pd.read_sql(query, sql_engine)
    print(f"Loaded stats for {len(player_stats)} players")
    
    return player_stats

def create_feature_set():    
    # Load match history
    matches = load_match_history()
    
    # Load player stats
    player_stats = get_player_stats_from_db()
    
    # Create a mapping from player names to their stats
    player_stats_dict = {}
    for _, row in player_stats.iterrows():
        player_stats_dict[row['name']] = row.to_dict()
    
    print("Creating feature set from matchups...")
    
    # List to store all matchup features
    matchup_features = []
    
    # Define the stat columns we want to use as features
    stat_columns = [
        'first_in', 'first_won', 'second_won', 'avg_ace', 'avg_df', 
        'avg_bp_faced', 'avg_bp_saved', 'relative_first_in', 'relative_first_won', 
        'relative_second_won', 'relative_avg_ace', 'relative_avg_df', 
        'relative_bp_faced', 'relative_bp_saved', 'overall_wr', 'hand', 'height'
    ]
    surface_wr_map = {
        'Hard': 'hard_court_wr',
        'Clay': 'clay_court_wr',
        'Grass': 'grass_court_wr',
        'Carpet': 'carpet_court_wr'
    }
    
    # Process each match
    for _, match in tqdm.tqdm(matches.iterrows(), total=len(matches), desc="Processing matches"):
        winner_name = match['winner_name']
        loser_name = match['loser_name']
        
        # Skip if we don't have stats for either player
        if winner_name not in player_stats_dict or loser_name not in player_stats_dict:
            continue
        
        winner_stats = player_stats_dict[winner_name]
        loser_stats = player_stats_dict[loser_name]
        
        # Randomly decide which player is Player A (50/50 chance)
        # This ensures we have a balanced dataset with both 0s and 1s
        if np.random.random() < 0.5:
            # Winner is Player A
            player_a_name = winner_name
            player_b_name = loser_name
            player_a_stats = winner_stats
            player_b_stats = loser_stats
            target = 1  # Player A wins
        else:
            # Loser is Player A
            player_a_name = loser_name
            player_b_name = winner_name
            player_a_stats = loser_stats
            player_b_stats = winner_stats
            target = 0  # Player A loses
        
        # Create feature row
        feature_row = {
            'match_id': match.get('tourney_id', '') + '_' + str(match.get('match_num', '')),
            'surface': match.get('surface', ''),
            'tourney_date': match.get('tourney_date', ''),
            'player_a_name': player_a_name,
            'player_b_name': player_b_name,
            'winner_name': winner_name,
            'loser_name': loser_name,
            'winner_id': match.get('winner_id', ''),
            'loser_id': match.get('loser_id', ''),
            'target': target
        }
        
        # Calculate differences for each stat (Player A - Player B)
        for col in stat_columns:
            player_a_val = player_a_stats.get(col, 0)
            player_b_val = player_b_stats.get(col, 0)
            if pd.isna(player_a_val):
                player_a_val = 0
            if pd.isna(player_b_val):
                player_b_val = 0
            feature_row[f'diff_{col}'] = player_a_val - player_b_val
        match_surface = match.get('surface', '')
        wr_col = surface_wr_map.get(str(match_surface).title())
        if wr_col:
            player_a_wr = player_a_stats.get(wr_col, 0)
            player_b_wr = player_b_stats.get(wr_col, 0)
            if pd.isna(player_a_wr):
             player_a_wr = 0
            if pd.isna(player_b_wr):
                player_b_wr = 0
            feature_row['diff_surface_wr'] = player_a_wr - player_b_wr
        else:
            feature_row['diff_surface_wr'] = 0
        
        matchup_features.append(feature_row)
    
    # Convert to DataFrame
    df = pd.DataFrame(matchup_features)
    
    print(f"Created feature set with {len(df)} matchups")
    print(f"Target distribution: {df['target'].value_counts()}")
    print(f"Feature columns: {[col for col in df.columns if col.startswith('diff_')]}")
    
    return df

# Create the feature set
df = create_feature_set()

# Define feature columns (all the diff_ columns)
feature_cols = [col for col in df.columns if col.startswith('diff_')]

# Select features and target
X = df[feature_cols]
y = df['target']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=6, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Number of features: {len(feature_cols)}")

# Save the processed data
df.to_csv('processed_matchups.csv', index=False)
print("Saved processed matchups to processed_matchups.csv")

