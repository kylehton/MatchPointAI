import pandas as pd
from sqlalchemy import create_engine
import os
import joblib
import sys
from dotenv import load_dotenv
load_dotenv()

# Add the parent directory to the path so we can import from dataset_creation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_creation.manage_features import retrieve_player_stats

# Load the trained model
# 6795 -> cv accuracy (67.95)
# 7450 -> roc auc score (74.50)
model = joblib.load("model/xgb_model_6795_7450.joblib")

# Connect to database
sql_engine = create_engine(os.getenv('POSTGRES_NEON_STRING'))
player_table = os.getenv("PLAYER_TABLE")


def predict_matchup(player_a_name: str, player_b_name: str, surface: str) -> dict:
    # Get stats for both players
    try:
        player_a_stats = retrieve_player_stats(player_a_name)
        player_b_stats = retrieve_player_stats(player_b_name)
        if player_a_stats is None or player_b_stats is None:
            return {"error": f"Player stats not found for one or both players"}
        if hasattr(player_a_stats, '_asdict'):
            player_a_stats = player_a_stats._asdict()
        if hasattr(player_b_stats, '_asdict'):
            player_b_stats = player_b_stats._asdict()
    except Exception as e:
        return {"error": str(e)}

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
    feature_dict = {}
    for col in stat_columns:
        player_a_val = player_a_stats.get(col, 0)
        player_b_val = player_b_stats.get(col, 0)
        if pd.isna(player_a_val):
            player_a_val = 0
        if pd.isna(player_b_val):
            player_b_val = 0
        feature_dict[f'diff_{col}'] = player_a_val - player_b_val
    # Add only the relevant surface winrate diff
    wr_col = surface_wr_map.get(str(surface).title())
    if wr_col:
        player_a_wr = player_a_stats.get(wr_col, 0)
        player_b_wr = player_b_stats.get(wr_col, 0)
        if pd.isna(player_a_wr):
            player_a_wr = 0
        if pd.isna(player_b_wr):
            player_b_wr = 0
        feature_dict['diff_surface_wr'] = player_a_wr - player_b_wr
    else:
        feature_dict['diff_surface_wr'] = 0
    X = pd.DataFrame([feature_dict])
    win_probability = model.predict_proba(X)[0][1]
    prediction = model.predict(X)[0]
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

result1 = predict_matchup("Roger Federer", "Rafael Nadal", "Clay")
result2 = predict_matchup("Roger Federer", "Rafael Nadal", "Hard")
result3 = predict_matchup("Roger Federer", "Rafael Nadal", "Grass")

print("Clay:", result1['winner'], result1['win_probability'])
print("Hard:", result2['winner'], result2['win_probability'])
print("Grass:", result3['winner'], result3['win_probability'])