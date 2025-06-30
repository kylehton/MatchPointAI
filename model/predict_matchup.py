import pandas as pd
from sqlalchemy import create_engine
import os
import joblib
import sys

# Add the parent directory to the path so we can import from dataset_creation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_creation.manage_features import retrieve_player_stats

# Load the trained model
model = joblib.load("model/xgb_model.joblib")

# Connect to database
sql_engine = create_engine(os.getenv('POSTGRES_NEON_STRING'))
player_table = os.getenv("PLAYER_TABLE")


def predict_matchup(player_a_name, player_b_name) -> dict:
    # Get stats for both players
    try:
        player_a_stats = retrieve_player_stats(player_a_name)
        player_b_stats = retrieve_player_stats(player_b_name)
        
        # Convert to dictionaries if they're not already
        if player_a_stats is None or player_b_stats is None:
            return {"error": f"Player stats not found for one or both players"}
            
        # Convert to dict if it's a Row object
        if hasattr(player_a_stats, '_asdict'):
            player_a_stats = player_a_stats._asdict()
        if hasattr(player_b_stats, '_asdict'):
            player_b_stats = player_b_stats._asdict()
            
    except Exception as e:
        return {"error": str(e)}
    
    # Define the stat columns (same as in training)
    stat_columns = [
        'first_in', 'first_won', 'second_won', 'avg_ace', 'avg_df', 
        'avg_bp_faced', 'avg_bp_saved', 'relative_first_in', 'relative_first_won', 
        'relative_second_won', 'relative_avg_ace', 'relative_avg_df', 
        'relative_bp_faced', 'relative_bp_saved', 'overall_wr', 'hard_court_wr', 
        'grass_court_wr', 'clay_court_wr', 'carpet_court_wr', 'hand', 'height'
    ]
    
    # Calculate differences (Player A - Player B)
    feature_dict = {}
    for col in stat_columns:
        player_a_val = player_a_stats.get(col, 0)
        player_b_val = player_b_stats.get(col, 0)
        
        # Handle NaN values
        if pd.isna(player_a_val):
            player_a_val = 0
        if pd.isna(player_b_val):
            player_b_val = 0
            
        feature_dict[f'diff_{col}'] = player_a_val - player_b_val
    
    # Convert to DataFrame (model expects same format as training)
    X = pd.DataFrame([feature_dict])
    
    # Make prediction
    win_probability = model.predict_proba(X)[0][1]  # Probability that Player A wins
    prediction = model.predict(X)[0]  # 1 if Player A wins, 0 if Player B wins
    
    # Determine winner
    if prediction == 1:
        winner = player_a_name
        loser = player_b_name
    else:
        winner = player_b_name
        loser = player_a_name

    if winner == player_a_name:
        winner_probability = win_probability
    else:
        winner_probability = 1 - win_probability

    return {"winner": winner, "win_probability": round(float(winner_probability*100), 2)}

