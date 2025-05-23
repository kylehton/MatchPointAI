import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # Import tqdm for progress bars

# Load data
def load_data(directory):
    print("Loading data files...")
    dataframes = []
    # Get list of CSV files
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Use tqdm to show progress while loading files
    for filename in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(os.path.join(directory, filename))
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def calculate_player_stats(df, window=10): 
    df = df.copy()
    df = df.sort_values('tourney_date')
    
    # Initialize empty DataFrames for winner and loser stats
    winner_stats = pd.DataFrame()
    loser_stats = pd.DataFrame()
    
    # Get unique player lists
    winners = df['winner_name'].unique()
    losers = df['loser_name'].unique()
    all_players = set(winners) | set(losers)
    
    print("Calculating player stats...")
    for player in tqdm(set(winners) | set(losers), desc="Processing players"):
        # Get all matches for this player
        player_matches = df[(df['winner_name'] == player) | (df['loser_name'] == player)].sort_values('tourney_date')
        
        if len(player_matches) > 0:
            # Calculate surface-specific win rates using a more direct approach
            surface_win_rates = {}
            for surface in ['Hard', 'Clay', 'Grass']:
                surface_matches = player_matches[player_matches['surface'] == surface]
                if len(surface_matches) > 0:
                    surface_win_rates[surface] = (surface_matches['winner_name'] == player).mean()
                else:
                    surface_win_rates[surface] = 0.5  # Default win rate if no matches on surface
            
            # Calculate recent form (last window matches)
            recent_matches = player_matches.tail(window)
            recent_win_rate = (recent_matches['winner_name'] == player).mean()
            
            # Calculate overall stats
            total_matches = len(player_matches)
            total_wins = (player_matches['winner_name'] == player).sum()
            overall_win_rate = total_wins / total_matches
            
            # Store stats
            stats = pd.DataFrame({
                'player_name': [player],
                'total_matches': [total_matches],
                'overall_win_rate': [overall_win_rate],
                'recent_win_rate': [recent_win_rate],
                'current_rank': [player_matches.iloc[-1]['winner_rank'] if player_matches.iloc[-1]['winner_name'] == player 
                               else player_matches.iloc[-1]['loser_rank']],
                'current_rank_points': [player_matches.iloc[-1]['winner_rank_points'] if player_matches.iloc[-1]['winner_name'] == player 
                                      else player_matches.iloc[-1]['loser_rank_points']],
                'age': [player_matches.iloc[-1]['winner_age'] if player_matches.iloc[-1]['winner_name'] == player 
                       else player_matches.iloc[-1]['loser_age']],
                'height': [player_matches.iloc[-1]['winner_ht'] if player_matches.iloc[-1]['winner_name'] == player 
                          else player_matches.iloc[-1]['loser_ht']],
                'hand': [player_matches.iloc[-1]['winner_hand'] if player_matches.iloc[-1]['winner_name'] == player 
                        else player_matches.iloc[-1]['loser_hand']],
                'hard_win_rate': [surface_win_rates['Hard']],
                'clay_win_rate': [surface_win_rates['Clay']],
                'grass_win_rate': [surface_win_rates['Grass']]
            })
            
            if player in winners:
                winner_stats = pd.concat([winner_stats, stats])
            if player in losers:
                loser_stats = pd.concat([loser_stats, stats])
    
    return winner_stats, loser_stats

def calculate_h2h_stats(df):
    """Calculate head-to-head statistics between players"""
    print("Calculating head-to-head records...")
    
    # Create a list of all matchups
    matchups = []
    
    # Use tqdm to track matchup processing
    with tqdm(total=len(df), desc="Processing matchups") as h2h_pbar:
        for _, row in df.iterrows():
            # Add matchup in both directions
            matchups.append({
                'player1': row['winner_name'],
                'player2': row['loser_name'],
                'winner': 'player1',
                'surface': row['surface'],
                'date': row['tourney_date']
            })
            matchups.append({
                'player1': row['loser_name'],
                'player2': row['winner_name'],
                'winner': 'player2',
                'surface': row['surface'],
                'date': row['tourney_date']
            })
            h2h_pbar.update(1)
    
    h2h_df = pd.DataFrame(matchups)
    
    # Calculate overall H2H with progress indicator
    print("Calculating overall head-to-head statistics...")
    h2h_stats = h2h_df.groupby(['player1', 'player2']).agg(
        total_matches=('winner', 'count'),
        player1_wins=('winner', lambda x: (x == 'player1').sum())
    ).reset_index()
    
    h2h_stats['h2h_winrate'] = h2h_stats['player1_wins'] / h2h_stats['total_matches']
    
    # Calculate surface-specific H2H with progress tracking
    print("Calculating surface-specific head-to-head statistics...")
    surfaces = ['Hard', 'Clay', 'Grass']
    with tqdm(total=len(surfaces), desc="Processing surfaces") as surface_pbar:
        for surface in surfaces:
            surface_h2h = h2h_df[h2h_df['surface'] == surface].groupby(['player1', 'player2']).agg({
                f'{surface.lower()}_matches': ('winner', 'count'),
                f'{surface.lower()}_player1_wins': ('winner', lambda x: (x == 'player1').sum())
            }).reset_index()
            
            surface_h2h[f'{surface.lower()}_h2h_winrate'] = surface_h2h[f'{surface.lower()}_player1_wins'] / \
                                                           surface_h2h[f'{surface.lower()}_matches']
            
            h2h_stats = h2h_stats.merge(surface_h2h, on=['player1', 'player2'], how='left')
            surface_pbar.update(1)
    
    return h2h_stats

# Feature Engineering
def engineer_features(df_input):
    print("Engineering features...")
    # Make a copy of the input DataFrame to avoid SettingWithCopyWarning
    df = df_input.copy()
    
    # Use tqdm for tracking progress through major steps
    steps = ["Surface encoding", "Hand encoding", "Height features", "H2H records", 
             "Rolling stats", "Merging features", "Final cleaning"]
    progress_bar = tqdm(steps, desc="Feature engineering steps")
    
    # 1. Surface encoding (Hard=0, Clay=1, Grass=2)
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
    df['surface_encoded'] = df['surface'].map(surface_map)
    progress_bar.update(1)  # Update progress bar
    
    # 2. Hand encoding (L=0, R=1)
    df['winner_hand_encoded'] = df['winner_hand'].map({'L': 0, 'R': 1})
    df['loser_hand_encoded'] = df['loser_hand'].map({'L': 0, 'R': 1})
    progress_bar.update(1)  # Update progress bar
    
    # 3. Height features
    df['height_diff'] = df['winner_ht'] - df['loser_ht']
    df['winner_height'] = df['winner_ht']
    df['loser_height'] = df['loser_ht']
    progress_bar.update(1)  # Update progress bar
    
    # 4. Calculate head-to-head records
    print("Calculating head-to-head records...")
    # Create progress bar for H2H data preparation
    with tqdm(total=2, desc="Preparing H2H data") as h2h_prep_bar:
        df_reversed = df.copy()
        df_reversed[['player1', 'player2']] = df[['loser_name', 'winner_name']]
        df_reversed['winner'] = 'player2'
        h2h_prep_bar.update(1)
        
        df_original = df.copy()
        df_original[['player1', 'player2']] = df[['winner_name', 'loser_name']]
        df_original['winner'] = 'player1'
        h2h_prep_bar.update(1)
    
    print("Concatenating H2H data...")
    h2h_df = pd.concat([df_original[['player1', 'player2', 'winner']], 
                       df_reversed[['player1', 'player2', 'winner']]])
    
    print("Calculating H2H statistics...")
    # Create progress bar for unique player pairs
    unique_pairs = h2h_df[['player1', 'player2']].drop_duplicates()
    
    h2h_stats = h2h_df.groupby(['player1', 'player2']).agg(
        wins_player1=('winner', lambda x: (x == 'player1').sum()),
        total_matches=('winner', 'count')
    ).reset_index()
    
    h2h_stats['winrate_player1'] = h2h_stats['wins_player1'] / h2h_stats['total_matches']
    progress_bar.update(1)  # Update progress bar
    
    # 5. Calculate rolling statistics
    print("Calculating player rolling statistics...")
    winner_stats, loser_stats = calculate_player_stats(df)
    progress_bar.update(1)  # Update progress bar
    
    # Merge all features
    print("Merging all features...")
    # Use tqdm to track merging operations
    merge_steps = ["H2H merge", "Winner stats merge", "Loser stats merge", "Final processing"]
    with tqdm(total=len(merge_steps), desc="Merging features") as merge_bar:
        # Merge H2H stats
        result_df = df.merge(
            h2h_stats[['player1', 'player2', 'winrate_player1']],
            left_on=['winner_name', 'loser_name'],
            right_on=['player1', 'player2'],
            how='left'
        )
        merge_bar.update(1)
        
        # Merge winner stats
        result_df = result_df.merge(
            winner_stats,
            left_on='winner_name',
            right_on='player_name',
            how='left',
            suffixes=('', '_winner')
        )
        merge_bar.update(1)
        
        # Merge loser stats
        result_df = result_df.merge(
            loser_stats,
            left_on='loser_name',
            right_on='player_name',
            how='left',
            suffixes=('', '_loser')
        )
        merge_bar.update(1)
        
        # Add target variable and clean up
        result_df['target'] = 1  # Since we're using winner_name as player1, this is always 1
        result_df = result_df.drop(['player1', 'player2', 'player_name', 'player_name_winner', 'player_name_loser'], axis=1, errors='ignore')
        merge_bar.update(1)
    
    progress_bar.update(1)  # Update progress bar for merging features
    
    # Select only the features we want
    selected_features = [
        'tourney_date',
        'winner_name',
        'loser_name',
        'surface_encoded',
        'winner_hand_encoded',
        'loser_hand_encoded',
        'height_diff',
        'winrate_player1',
        'overall_win_rate_winner',
        'overall_win_rate_loser',
        'recent_win_rate_winner',
        'recent_win_rate_loser',
        'hard_win_rate_winner',
        'hard_win_rate_loser',
        'clay_win_rate_winner',
        'clay_win_rate_loser',
        'grass_win_rate_winner',
        'grass_win_rate_loser',
        'target'
    ]
    
    # Create a balanced dataset by duplicating rows and swapping winner/loser
    print("Creating balanced dataset...")
    balance_steps = ["Creating winner perspective", "Creating loser perspective", "Combining datasets", "Final cleanup"]
    with tqdm(total=len(balance_steps), desc="Balancing dataset") as balance_bar:
        # Original rows (winner perspective)
        winner_df = result_df.copy()
        winner_df['target'] = 1
        balance_bar.update(1)
        
        # Create loser perspective rows
        loser_df = result_df.copy()
        # Swap winner and loser columns
        loser_df = loser_df.rename(columns={
            'winner_name': 'loser_name',
            'loser_name': 'winner_name',
            'winner_hand_encoded': 'loser_hand_encoded',
            'loser_hand_encoded': 'winner_hand_encoded',
            'height_diff': 'height_diff',
            'overall_win_rate_winner': 'overall_win_rate_loser',
            'overall_win_rate_loser': 'overall_win_rate_winner',
            'recent_win_rate_winner': 'recent_win_rate_loser',
            'recent_win_rate_loser': 'recent_win_rate_winner',
            'hard_win_rate_winner': 'hard_win_rate_loser',
            'hard_win_rate_loser': 'hard_win_rate_winner',
            'clay_win_rate_winner': 'clay_win_rate_loser',
            'clay_win_rate_loser': 'clay_win_rate_winner',
            'grass_win_rate_winner': 'grass_win_rate_loser',
            'grass_win_rate_loser': 'grass_win_rate_winner'
        })
        loser_df['height_diff'] = -loser_df['height_diff']  # Invert height difference
        loser_df['target'] = 0
        balance_bar.update(1)
        
        # Combine both perspectives
        balanced_df = pd.concat([winner_df, loser_df], ignore_index=True)
        balance_bar.update(1)
        
        # Handle missing values
        print("Handling missing values...")
        # Fill missing values with appropriate defaults
        balanced_df['height_diff'] = balanced_df['height_diff'].fillna(0)  # No height difference if missing
        balanced_df['winrate_player1'] = balanced_df['winrate_player1'].fillna(0.5)  # Equal chance if no H2H
        
        # Fill missing win rates with 0.5 (equal chance)
        win_rate_columns = [col for col in balanced_df.columns if 'win_rate' in col]
        balanced_df[win_rate_columns] = balanced_df[win_rate_columns].fillna(0.5)
        
        # Fill missing hand encodings with 1 (right-handed as default)
        balanced_df['winner_hand_encoded'] = balanced_df['winner_hand_encoded'].fillna(1)
        balanced_df['loser_hand_encoded'] = balanced_df['loser_hand_encoded'].fillna(1)
        
        # Check if all selected features exist in the DataFrame
        available_features = [f for f in selected_features if f in balanced_df.columns]
        final_df = balanced_df[available_features]
        
        # Sort by date to ensure chronological order
        final_df = final_df.sort_values('tourney_date')
        balance_bar.update(1)
    
    progress_bar.update(1)  # Update progress bar for final cleaning
    
    # Close the progress bar
    progress_bar.close()
    
    return final_df

# Main function
def main():
    print("=== Tennis Match Feature Engineering ===")
    directory = 'datasets/atp_matches/'
    
    # Overall progress tracking
    main_steps = ["Loading data", "Date conversion", "Train/test split", "Train feature engineering", 
                 "Test feature engineering", "Saving results"]
    main_progress = tqdm(main_steps, desc="Overall progress")
    
    df = load_data(directory)
    main_progress.update(1)  # Update main progress
    
    print(f"Data loaded: {len(df)} matches")
    
    # Convert tourney_date to datetime
    print("Converting dates...")
    with tqdm(total=1, desc="Converting dates") as date_bar:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
        date_bar.update(1)
    
    # Sort the entire dataset by date first
    print("Sorting dataset by date...")
    with tqdm(total=1, desc="Sorting dataset") as sort_bar:
        df = df.sort_values('tourney_date')
        sort_bar.update(1)
    
    main_progress.update(1)  # Update main progress
    
    # Split data into train and test sets based on year
    print("Splitting into train and test sets...")
    with tqdm(total=2, desc="Splitting data") as split_bar:
        train_df = df[df['tourney_date'].dt.year < 2008].copy()
        split_bar.update(1)
        
        test_df = df[df['tourney_date'].dt.year >= 2008].copy()
        split_bar.update(1)
    
    print(f"Training set: {len(train_df)} matches (1968-2007)")
    print(f"Test set: {len(test_df)} matches (2008-2024)")
    
    main_progress.update(1)  # Update main progress
    
    # Engineer features for both sets
    print("\nProcessing training set...")
    train_features = engineer_features(train_df)
    main_progress.update(1)  # Update main progress
    
    print("\nProcessing test set...")
    test_features = engineer_features(test_df)
    main_progress.update(1)  # Update main progress
    
    # Save to CSV files
    print("\nSaving results...")
    with tqdm(total=2, desc="Saving files") as save_bar:
        train_features.to_csv('train_new_features.csv', index=False)
        save_bar.update(1)
        
        test_features.to_csv('test_new_features.csv', index=False)
        save_bar.update(1)
    
    main_progress.update(1)  # Update main progress
    main_progress.close()
    
    print("\n=== Feature engineering complete ===")
    print(f"Train features (1968-2007): {len(train_features)} samples saved to 'train_new_features.csv'")
    print(f"Test features (2008-2024): {len(test_features)} samples saved to 'test_new_features.csv'")

if __name__ == "__main__":
    main()