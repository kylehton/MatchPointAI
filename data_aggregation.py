import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
from tqdm import tqdm

class TennisPlayerAggregator:
    def __init__(self, data_path=None):
        """
        Initialize the aggregator
        data_path: path to directory containing CSV files or list of file paths
        """
        self.data_path = data_path
        self.combined_data = None
        self.player_stats = None
        
    def load_datasets(self, file_pattern="*.csv"):
        """
        Load and combine multiple tennis datasets
        """
        if isinstance(self.data_path, list):
            # If list of file paths provided
            dataframes = []
            for file_path in tqdm(self.data_path, desc="Loading datasets"):
                df = pd.read_csv(file_path)
                dataframes.append(df)
        else:
            # If directory path provided
            file_paths = glob.glob(os.path.join(self.data_path, file_pattern))
            dataframes = []
            for file_path in tqdm(file_paths, desc="Loading datasets"):
                df = pd.read_csv(file_path)
                dataframes.append(df)
        
        # Combine all datasets
        self.combined_data = pd.concat(dataframes, ignore_index=True)
        
        # Standardize column names (adjust based on your data structure)
        self.standardize_columns()
        
        # Convert date column to datetime
        if 'tourney_date' in self.combined_data.columns:
            self.combined_data['tourney_date'] = pd.to_datetime(self.combined_data['tourney_date'])
        elif 'date' in self.combined_data.columns:
            self.combined_data['date'] = pd.to_datetime(self.combined_data['date'])
            
        # Sort by date
        date_col = 'tourney_date' if 'tourney_date' in self.combined_data.columns else 'date'
        self.combined_data = self.combined_data.sort_values([date_col]).reset_index(drop=True)
        
        print(f"Loaded {len(self.combined_data)} matches from {len(dataframes)} datasets")
        return self.combined_data
    
    def standardize_columns(self):
        """
        Standardize column names across different datasets
        Adjust this based on your specific data structure
        """
        # Example column mappings - adjust based on your data
        column_mappings = {
            'winner_name': 'winner_name',
            'loser_name': 'loser_name',
            'winner_rank': 'winner_rank',
            'loser_rank': 'loser_rank',
            'surface': 'surface',
            'w_1stIn': 'winner_1st_serve_in',
            'w_1stWon': 'winner_1st_serve_won',
            'w_2ndWon': 'winner_2nd_serve_won',
            'w_SvGms': 'winner_serve_games',
            'w_bpSaved': 'winner_bp_saved',
            'w_bpFaced': 'winner_bp_faced',
            'l_1stIn': 'loser_1st_serve_in',
            'l_1stWon': 'loser_1st_serve_won',
            'l_2ndWon': 'loser_2nd_serve_won',
            'l_SvGms': 'loser_serve_games',
            'l_bpSaved': 'loser_bp_saved',
            'l_bpFaced': 'loser_bp_faced'
        }
        
        # Rename columns that exist
        existing_mappings = {k: v for k, v in column_mappings.items() if k in self.combined_data.columns}
        self.combined_data.rename(columns=existing_mappings, inplace=True)
    
    def create_player_match_records(self):
        """
        Create individual player match records (both winners and losers)
        """
        date_col = 'tourney_date' if 'tourney_date' in self.combined_data.columns else 'date'
        
        # Winner records
        winner_records = self.combined_data.copy()
        winner_records['player_name'] = winner_records['winner_name']
        winner_records['opponent_name'] = winner_records['loser_name']
        winner_records['player_rank'] = winner_records['winner_rank']
        winner_records['opponent_rank'] = winner_records['loser_rank']
        winner_records['won_match'] = 1
        winner_records['match_result'] = 'W'
        
        # Add serve statistics for winner
        serve_cols = ['1st_serve_in', '1st_serve_won', '2nd_serve_won', 'serve_games', 'bp_saved', 'bp_faced']
        for col in tqdm(serve_cols, desc="Adding serve statistics for winner"):
            winner_col = f'winner_{col}'
            if winner_col in winner_records.columns:
                winner_records[f'player_{col}'] = winner_records[winner_col]
            
        # Loser records
        loser_records = self.combined_data.copy()
        loser_records['player_name'] = loser_records['loser_name']
        loser_records['opponent_name'] = loser_records['winner_name']
        loser_records['player_rank'] = loser_records['loser_rank']
        loser_records['opponent_rank'] = loser_records['winner_rank']
        loser_records['won_match'] = 0
        loser_records['match_result'] = 'L'
        
        # Add serve statistics for loser
        for col in tqdm(serve_cols, desc="Adding serve statistics for loser"):
            loser_col = f'loser_{col}'
            if loser_col in loser_records.columns:
                loser_records[f'player_{col}'] = loser_records[loser_col]
        
        # Select relevant columns
        relevant_cols = [date_col, 'player_name', 'opponent_name', 'player_rank', 'opponent_rank', 
                        'surface', 'won_match', 'match_result'] + [f'player_{col}' for col in serve_cols]
        relevant_cols = [col for col in relevant_cols if col in winner_records.columns]
        
        # Combine winner and loser records
        all_player_records = pd.concat([
            winner_records[relevant_cols],
            loser_records[relevant_cols]
        ], ignore_index=True)
        
        # Sort by player and date
        all_player_records = all_player_records.sort_values(['player_name', date_col]).reset_index(drop=True)
        
        return all_player_records
    
    def calculate_rolling_stats(self, player_records, window_size=20):
        """
        Calculate rolling averages for each player
        window_size: number of matches to use for rolling average
        """
        date_col = 'tourney_date' if 'tourney_date' in player_records.columns else 'date'
        
        # Group by player
        grouped = player_records.groupby('player_name')
        
        rolling_stats = []
        
        for player_name, player_data in tqdm(grouped, desc="Calculating rolling statistics"):
            player_data = player_data.sort_values(date_col).copy()
            
            # Calculate rolling win rate
            player_data['rolling_win_rate'] = player_data['won_match'].rolling(
                window=window_size, min_periods=1
            ).mean()
            
            # Calculate rolling win rate by surface
            for surface in tqdm(['Hard', 'Clay', 'Grass'], desc="Calculating rolling win rate by surface"):
                surface_mask = player_data['surface'] == surface
                if surface_mask.sum() > 0:
                    surface_data = player_data[surface_mask].copy()
                    if len(surface_data) > 0:
                        rolling_surface_wr = surface_data['won_match'].rolling(
                            window=min(window_size, len(surface_data)), min_periods=1
                        ).mean()
                        # Map back to original dataframe
                        player_data.loc[surface_mask, f'rolling_win_rate_{surface.lower()}'] = rolling_surface_wr
            
            # Calculate rolling serve statistics
            serve_stats = ['1st_serve_in', '1st_serve_won', '2nd_serve_won']
            for stat in tqdm(serve_stats, desc="Calculating rolling serve statistics"):
                col_name = f'player_{stat}'
                if col_name in player_data.columns:
                    # Convert to numeric and handle percentages
                    player_data[col_name] = pd.to_numeric(player_data[col_name], errors='coerce')
                    player_data[f'rolling_{stat}'] = player_data[col_name].rolling(
                        window=window_size, min_periods=1
                    ).mean()
            
            # Calculate rolling rank (best rank achieved in window)
            player_data['rolling_best_rank'] = player_data['player_rank'].rolling(
                window=window_size, min_periods=1
            ).min()
            
            rolling_stats.append(player_data)
        
        return pd.concat(rolling_stats, ignore_index=True)
    
    def aggregate_career_stats(self, rolling_data):
        """
        Aggregate career statistics for each player
        """
        date_col = 'tourney_date' if 'tourney_date' in rolling_data.columns else 'date'
        
        career_stats = []
        
        for player_name, player_data in tqdm(rolling_data.groupby('player_name'), desc="Aggregating career statistics"):
            player_data = player_data.sort_values(date_col)
            
            stats = {
                'player_name': player_name,
                'career_start': player_data[date_col].min(),
                'career_end': player_data[date_col].max(),
                'total_matches': len(player_data),
                'total_wins': player_data['won_match'].sum(),
                'career_win_rate': player_data['won_match'].mean(),
                'highest_ranking': player_data['player_rank'].min() if not player_data['player_rank'].isna().all() else None
            }
            
            # Surface-specific win rates
            for surface in tqdm(['hard', 'clay', 'grass'], desc="Calculating surface-specific win rates"):
                surface_data = player_data[player_data['surface'].str.lower() == surface]
                if len(surface_data) > 0:
                    stats[f'{surface}_matches'] = len(surface_data)
                    stats[f'{surface}_wins'] = surface_data['won_match'].sum()
                    stats[f'{surface}_win_rate'] = surface_data['won_match'].mean()
                    
                    # Latest rolling average for surface
                    rolling_col = f'rolling_win_rate_{surface}'
                    if rolling_col in surface_data.columns:
                        latest_rolling = surface_data[rolling_col].dropna().iloc[-1] if not surface_data[rolling_col].dropna().empty else None
                        stats[f'{surface}_rolling_win_rate_latest'] = latest_rolling
                else:
                    stats[f'{surface}_matches'] = 0
                    stats[f'{surface}_wins'] = 0
                    stats[f'{surface}_win_rate'] = None
                    stats[f'{surface}_rolling_win_rate_latest'] = None
            
            # Serve statistics (career averages)
            serve_stats = ['1st_serve_in', '1st_serve_won', '2nd_serve_won']
            for stat in tqdm(serve_stats, desc="Calculating serve statistics"):
                col_name = f'player_{stat}'
                rolling_col = f'rolling_{stat}'
                
                if col_name in player_data.columns:
                    # Career average
                    career_avg = player_data[col_name].mean()
                    stats[f'career_avg_{stat}'] = career_avg
                    
                    # Latest rolling average
                    if rolling_col in player_data.columns:
                        latest_rolling = player_data[rolling_col].dropna().iloc[-1] if not player_data[rolling_col].dropna().empty else None
                        stats[f'latest_rolling_{stat}'] = latest_rolling
            
            career_stats.append(stats)
        
        return pd.DataFrame(career_stats)
    
    def run_full_analysis(self, window_size=20, min_matches=10):
        """
        Run the complete analysis pipeline
        """
        print("Step 1: Loading datasets...")
        if self.combined_data is None:
            self.load_datasets()
        
        print("Step 2: Creating player match records...")
        player_records = self.create_player_match_records()
        
        print("Step 3: Calculating rolling statistics...")
        rolling_data = self.calculate_rolling_stats(player_records, window_size)
        
        print("Step 4: Aggregating career statistics...")
        career_stats = self.aggregate_career_stats(rolling_data)
        
        # Filter players with minimum matches
        career_stats = career_stats[career_stats['total_matches'] >= min_matches]
        
        # Sort by total matches (or another metric)
        career_stats = career_stats.sort_values('total_matches', ascending=False)
        
        self.player_stats = career_stats
        
        print(f"Analysis complete! Found {len(career_stats)} players with {min_matches}+ matches")
        return career_stats, rolling_data

# Example usage:
if __name__ == "__main__":
    # Initialize aggregator
    # Option 1: Provide directory path
    aggregator = TennisPlayerAggregator("data/main")
    
    # Option 2: Provide list of file paths
    # file_paths = ["data1.csv", "data2.csv", "data3.csv"]
    # aggregator = TennisPlayerAggregator(file_paths)
    
    # Run analysis
    career_stats, rolling_data = aggregator.run_full_analysis(
        window_size=25,  # 25-match rolling window
        min_matches=50   # Only include players with 50+ matches
    )
    
    # Display top players by matches played
    print("\nTop 10 players by total matches:")
    print(career_stats[['player_name', 'total_matches', 'career_win_rate', 'highest_ranking']].head(10))
    
    # Display surface specialists
    print("\nTop clay court players (by win rate with 20+ clay matches):")
    clay_specialists = career_stats[career_stats['clay_matches'] >= 20].nlargest(10, 'clay_win_rate')
    print(clay_specialists[['player_name', 'clay_matches', 'clay_win_rate', 'clay_rolling_win_rate_latest']])
    
    # Save results
    career_stats.to_csv('tennis_career_stats.csv', index=False)
    rolling_data.to_csv('tennis_rolling_data.csv', index=False)
    
    print("\nResults saved to CSV files!")