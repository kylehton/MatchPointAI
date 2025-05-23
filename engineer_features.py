import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sqlalchemy import create_engine, String, Integer, Float, Date, text
import dotenv

dotenv.load_dotenv()

def load_data(directory):
    """Load all match data from CSV files in the directory"""
    print("Loading data files...")
    dataframes = []
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    for filename in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(os.path.join(directory, filename))
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


#TODO: Correctly parse data from files into stats
def calculate_player_stats(df):
    player_stats = {
        # Basic Info
        'player_name': str,
        'current_rank': int,
        'current_rank_points': int,
        'age': float,
        'height': float,
        'hand': str,
        
        # Overall Performance
        'overall_win_rate': float,  # Career win rate
        'recent_win_rate': float,   # Last 12 months
        'matches_played': int,      # Total matches
        'recent_matches': int,      # Matches in last 12 months
        
        # Surface-Specific Stats
        'hard_win_rate': float,
        'clay_win_rate': float,
        'grass_win_rate': float,
        'recent_hard_win_rate': float,
        'recent_clay_win_rate': float,
        'recent_grass_win_rate': float,
        
        # Match Performance Metrics
        'avg_aces_per_match': float,
        'avg_double_faults_per_match': float,
        'avg_first_serve_percentage': float,
        'avg_first_serve_won_percentage': float,
        'avg_second_serve_won_percentage': float,
        'avg_break_points_saved_percentage': float,
        'avg_break_points_converted_percentage': float,
        
        # Recent Form (last 3 months)
        'recent_avg_aces': float,
        'recent_avg_double_faults': float,
        'recent_avg_first_serve': float,
        'recent_avg_first_serve_won': float,
        'recent_avg_second_serve_won': float,
        'recent_avg_break_points_saved': float,
        'recent_avg_break_points_converted': float,
        
        # Tournament Level Performance
        'grand_slam_win_rate': float,
        'masters_win_rate': float,
        'atp_win_rate': float,
        
    }

def create_match_features(player1_stats, player2_stats, match_info):
    """Create features for a specific match prediction"""
    features = {
        # Basic Match Info
        'surface': str,
        'tournament_level': str,
        'round': str,
        
        # Player Comparison Features
        'rank_diff': int,  # player1_rank - player2_rank
        'rank_points_diff': int,
        'age_diff': float,
        'height_diff': float,
        'hand_matchup': str,  # e.g., 'R-R', 'L-R'
        
        # Win Rate Differences
        'overall_win_rate_diff': float,
        'recent_win_rate_diff': float,
        'surface_win_rate_diff': float,  # specific to match surface
        'recent_surface_win_rate_diff': float,
        
        # Performance Metric Differences
        'aces_per_match_diff': float,
        'double_faults_per_match_diff': float,
        'first_serve_percentage_diff': float,
        'first_serve_won_percentage_diff': float,
        'second_serve_won_percentage_diff': float,
        'break_points_saved_percentage_diff': float,
        'break_points_converted_percentage_diff': float,
        
        # Recent Form Differences
        'recent_aces_diff': float,
        'recent_double_faults_diff': float,
        'recent_first_serve_diff': float,
        'recent_first_serve_won_diff': float,
        'recent_second_serve_won_diff': float,
        'recent_break_points_saved_diff': float,
        'recent_break_points_converted_diff': float,
        
        # Tournament Level Differences
        'tournament_level_win_rate_diff': float,
        'round_win_rate_diff': float,
        
    }

