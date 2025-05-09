import numpy as np
import pandas as pd
import os

# Load data
def load_data(directory):
    # Assuming you have CSV files in the directory
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename))
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def calculate_player_stats(df):
    # Calculate average statistics for each player
    player_stats = {}
    
    # Group by winner and loser to get their stats
    winner_stats = df.groupby('winner_name').agg({
        'w_ace': 'mean',
        'w_df': 'mean',
        'w_1stIn': 'mean',
        'w_1stWon': 'mean',
        'w_2ndWon': 'mean',
        'w_bpFaced': 'mean',
        'w_bpSaved': 'mean'
    }).reset_index()
    
    loser_stats = df.groupby('loser_name').agg({
        'l_ace': 'mean',
        'l_df': 'mean',
        'l_1stIn': 'mean',
        'l_1stWon': 'mean',
        'l_2ndWon': 'mean',
        'l_bpFaced': 'mean',
        'l_bpSaved': 'mean'
    }).reset_index()
    
    # Rename columns to be consistent
    winner_stats.columns = ['player_name', 'avg_ace', 'avg_df', 'avg_1stIn', 'avg_1stWon', 
                          'avg_2ndWon', 'avg_bpFaced', 'avg_bpSaved']
    loser_stats.columns = ['player_name', 'avg_ace', 'avg_df', 'avg_1stIn', 'avg_1stWon', 
                          'avg_2ndWon', 'avg_bpFaced', 'avg_bpSaved']
    
    # Combine winner and loser stats
    all_stats = pd.concat([winner_stats, loser_stats])
    player_stats = all_stats.groupby('player_name').mean().reset_index()
    
    return player_stats

# Feature Engineering
def engineer_features(df_input):
    # Make a copy of the input DataFrame to avoid SettingWithCopyWarning
    df = df_input.copy()
    
    # 1. Surface encoding (Hard=0, Clay=1, Grass=2)
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
    df['surface_encoded'] = df['surface'].map(surface_map)
    
    # 2. Hand encoding (L=0, R=1)
    df['winner_hand_encoded'] = df['winner_hand'].map({'L': 0, 'R': 1})
    df['loser_hand_encoded'] = df['loser_hand'].map({'L': 0, 'R': 1})
    
    # 3. Calculate head-to-head records
    df_reversed = df.copy()
    df_reversed[['player1', 'player2']] = df[['loser_name', 'winner_name']]
    df_reversed['winner'] = 'player2'
    
    df_original = df.copy()
    df_original[['player1', 'player2']] = df[['winner_name', 'loser_name']]
    df_original['winner'] = 'player1'
    
    h2h_df = pd.concat([df_original[['player1', 'player2', 'winner']], 
                       df_reversed[['player1', 'player2', 'winner']]])
    
    h2h_stats = h2h_df.groupby(['player1', 'player2']).agg(
        wins_player1=('winner', lambda x: (x == 'player1').sum()),
        total_matches=('winner', 'count')
    ).reset_index()
    
    h2h_stats['winrate_player1'] = h2h_stats['wins_player1'] / h2h_stats['total_matches']
    
    # 4. Calculate player statistics
    player_stats = calculate_player_stats(df)
    
    # Merge all features
    result_df = df.merge(
        h2h_stats[['player1', 'player2', 'winrate_player1']],
        left_on=['winner_name', 'loser_name'],
        right_on=['player1', 'player2'],
        how='left'
    )
    
    result_df = result_df.merge(
        player_stats,
        left_on='winner_name',
        right_on='player_name',
        how='left',
        suffixes=('', '_winner')
    )
    
    result_df = result_df.merge(
        player_stats,
        left_on='loser_name',
        right_on='player_name',
        how='left',
        suffixes=('', '_loser')
    )
    
    # Select only the features we want
    selected_features = [
        'tourney_date',
        'surface_encoded',
        'winner_hand_encoded',
        'loser_hand_encoded',
        'winrate_player1',
        'avg_ace',
        'avg_df',
        'avg_1stIn',
        'avg_1stWon',
        'avg_2ndWon',
        'avg_bpFaced',
        'avg_bpSaved',
        'avg_ace_loser',
        'avg_df_loser',
        'avg_1stIn_loser',
        'avg_1stWon_loser',
        'avg_2ndWon_loser',
        'avg_bpFaced_loser',
        'avg_bpSaved_loser'
    ]
    
    # Clean up the dataframe and select only desired features
    result_df = result_df.drop(['player1', 'player2', 'player_name', 'player_name_loser'], axis=1, errors='ignore')
    
    # Check if all selected features exist in the DataFrame
    available_features = [f for f in selected_features if f in result_df.columns]
    final_df = result_df[available_features]
    
    # Sort by date to ensure chronological order
    final_df = final_df.sort_values('tourney_date')
    
    return final_df

# Main function
def main():
    directory = 'datasets/atp_matches/'
    df = load_data(directory)
    
    # Convert tourney_date to datetime
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    
    # Sort the entire dataset by date first
    df = df.sort_values('tourney_date')
    
    # Split data into train and test sets based on year
    train_df = df[df['tourney_date'].dt.year < 2008].copy()
    test_df = df[df['tourney_date'].dt.year >= 2008].copy()
    
    # Engineer features for both sets
    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)
    
    # Save to CSV files
    train_features.to_csv('train_new_features.csv', index=False)
    test_features.to_csv('test_new_features.csv', index=False)
    
    print("Feature engineering complete.")
    print("Train features (1968-2007) saved to 'train_new_features.csv'")
    print("Test features (2008-2024) saved to 'test_new_features.csv'")
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")

if __name__ == "__main__":
    main()