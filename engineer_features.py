from data_aggregation import TennisPlayerAggregator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class TennisMatchPredictor:
    def __init__(self, aggregator_data):
        """
        INITIALIZATION CHUNK:
        - Takes output from TennisPlayerAggregator (career stats + rolling data)
        - Sets up empty containers for model, features, encoders
        - This is like setting up your workspace before building the model
        """
        self.career_stats, self.rolling_data = aggregator_data  # Data from aggregator
        self.match_features = None      # Will store match-level features
        self.model = None              # Will store trained XGBoost model
        self.feature_names = []        # List of feature column names
        self.label_encoders = {}       # For encoding categorical variables
        
    def create_match_features(self):
        """
        DATA TRANSFORMATION CHUNK:
        This is the most complex part - converts player-level data into match-level data
        
        Problem: Our data has individual player records (one row per player per match)
        Solution: Combine two player records into one match record with features for both players
        
        Steps:
        1. Group matches by date and surface
        2. Find player pairs (winner vs loser)
        3. Extract features for both players
        4. Create one row per match with player1_features + player2_features + outcome
        """
        print("Creating match-level features...")
        
        # Get the date column name
        date_col = 'tourney_date' if 'tourney_date' in self.rolling_data.columns else 'date'
        
        # Create match-level dataset
        matches = []
        
        # Group by match (assuming we can identify matches by date, players, surface)
        match_groups = self.rolling_data.groupby([date_col, 'surface'])
        
        processed_matches = set()
        
        for (match_date, surface), group in match_groups:
            # Get unique players in this group
            players = group['player_name'].unique()
            
            # Find actual matches (need winner/loser pairs)
            for _, match in group.iterrows():
                player = match['player_name']
                opponent = match['opponent_name']
                
                # Create unique match identifier
                match_id = f"{match_date}_{min(player, opponent)}_{max(player, opponent)}_{surface}"
                
                if match_id in processed_matches:
                    continue
                    
                processed_matches.add(match_id)
                
                # Get player data
                player_data = match.copy()
                
                # Get opponent data from the same match/date
                opponent_data = group[
                    (group['player_name'] == opponent) & 
                    (group['opponent_name'] == player)
                ]
                
                if len(opponent_data) == 0:
                    continue
                    
                opponent_data = opponent_data.iloc[0]
                
                # Create feature row
                match_features = self.extract_match_features(
                    player_data, opponent_data, match_date, surface
                )
                
                if match_features is not None:
                    matches.append(match_features)
        
        self.match_features = pd.DataFrame(matches)
        print(f"Created {len(self.match_features)} match feature sets")
        
        return self.match_features
    
    def extract_match_features(self, player_data, opponent_data, match_date, surface):
        """
        FEATURE ENGINEERING CHUNK:
        This is where the magic happens - creates all the ML features for one match
        
        Creates 4 types of features:
        1. ROLLING FEATURES (recent form) - most important for prediction
        2. SURFACE-SPECIFIC FEATURES - how they play on this surface recently  
        3. CAREER FEATURES - baseline skill level
        4. COMPARATIVE FEATURES - advantages/disadvantages between players
        
        Returns: Dictionary with ~30-40 features per match
        """
        try:
            # Get career stats for both players
            player_career = self.career_stats[
                self.career_stats['player_name'] == player_data['player_name']
            ]
            opponent_career = self.career_stats[
                self.career_stats['player_name'] == opponent_data['player_name']
            ]
            
            if len(player_career) == 0 or len(opponent_career) == 0:
                return None
                
            player_career = player_career.iloc[0]
            opponent_career = opponent_career.iloc[0]
            
            features = {
                'match_date': match_date,
                'surface': surface,
                'player_name': player_data['player_name'],
                'opponent_name': opponent_data['player_name'],
                'player_won': player_data['won_match'],  # Target variable
            }
            
            # Rolling form features (multiple windows)
            rolling_windows = [10, 20, 50]
            
            """
            ROLLING WINDOWS EXPLANATION:
            - Window 10: Very recent form (last 10 matches) - captures hot/cold streaks
            - Window 20: Short-term form - balances recent performance with stability  
            - Window 50: Medium-term form - shows longer trends, less noise
            
            For each window, we get:
            - Win rate in last N matches
            - First serve % in last N matches  
            - First serve won % in last N matches
            
            Why multiple windows? XGBoost learns when to trust recent vs longer-term form
            """
            
            for window in rolling_windows:
                # Get rolling stats for this window (if available)
                player_recent = self.get_recent_matches(
                    player_data['player_name'], match_date, window
                )
                opponent_recent = self.get_recent_matches(
                    opponent_data['player_name'], match_date, window
                )
                
                if player_recent is not None and len(player_recent) > 0:
                    features[f'player_rolling_wr_{window}'] = player_recent['won_match'].mean()
                    features[f'player_rolling_1st_in_{window}'] = player_recent['player_1st_serve_in'].mean()
                    features[f'player_rolling_1st_won_{window}'] = player_recent['player_1st_serve_won'].mean()
                else:
                    features[f'player_rolling_wr_{window}'] = player_career['career_win_rate']
                    features[f'player_rolling_1st_in_{window}'] = player_career.get('career_avg_1st_serve_in', 0.6)
                    features[f'player_rolling_1st_won_{window}'] = player_career.get('career_avg_1st_serve_won', 0.7)
                
                if opponent_recent is not None and len(opponent_recent) > 0:
                    features[f'opponent_rolling_wr_{window}'] = opponent_recent['won_match'].mean()
                    features[f'opponent_rolling_1st_in_{window}'] = opponent_recent['player_1st_serve_in'].mean()
                    features[f'opponent_rolling_1st_won_{window}'] = opponent_recent['player_1st_serve_won'].mean()
                else:
                    features[f'opponent_rolling_wr_{window}'] = opponent_career['career_win_rate']
                    features[f'opponent_rolling_1st_in_{window}'] = opponent_career.get('career_avg_1st_serve_in', 0.6)
                    features[f'opponent_rolling_1st_won_{window}'] = opponent_career.get('career_avg_1st_serve_won', 0.7)
            
            # Surface-specific rolling stats
            surface_lower = surface.lower()
            player_surface_recent = self.get_recent_surface_matches(
                player_data['player_name'], match_date, surface, 25
            )
            opponent_surface_recent = self.get_recent_surface_matches(
                opponent_data['player_name'], match_date, surface, 25
            )
            
            if player_surface_recent is not None and len(player_surface_recent) > 0:
                features['player_surface_rolling_wr'] = player_surface_recent['won_match'].mean()
            else:
                features['player_surface_rolling_wr'] = player_career.get(f'{surface_lower}_win_rate', player_career['career_win_rate'])
            
            if opponent_surface_recent is not None and len(opponent_surface_recent) > 0:
                features['opponent_surface_rolling_wr'] = opponent_surface_recent['won_match'].mean()
            else:
                features['opponent_surface_rolling_wr'] = opponent_career.get(f'{surface_lower}_win_rate', opponent_career['career_win_rate'])
            
            # Career statistics
            features['player_career_wr'] = player_career['career_win_rate']
            features['opponent_career_wr'] = opponent_career['career_win_rate']
            features['player_highest_rank'] = player_career.get('highest_ranking', 999)
            features['opponent_highest_rank'] = opponent_career.get('highest_ranking', 999)
            features['player_total_matches'] = player_career['total_matches']
            features['opponent_total_matches'] = opponent_career['total_matches']
            
            # Ranking difference (lower rank number = better)
            if features['player_highest_rank'] is not None and features['opponent_highest_rank'] is not None:
                features['ranking_advantage'] = features['opponent_highest_rank'] - features['player_highest_rank']
            else:
                features['ranking_advantage'] = 0
            
            # Surface experience
            features['player_surface_matches'] = player_career.get(f'{surface_lower}_matches', 0)
            features['opponent_surface_matches'] = opponent_career.get(f'{surface_lower}_matches', 0)
            
            # Form momentum (recent vs career)
            features['player_form_momentum'] = features['player_rolling_wr_10'] - features['player_career_wr']
            features['opponent_form_momentum'] = features['opponent_rolling_wr_10'] - features['opponent_career_wr']
            
            # Head-to-head (simplified - would need actual H2H data)
            features['h2h_advantage'] = 0  # Placeholder
            
            return features
            
        except Exception as e:
            print(f"Error processing match: {e}")
            return None
    
    def get_recent_matches(self, player_name, match_date, window):
        """
        TIME-AWARE DATA RETRIEVAL CHUNK:
        
        CRITICAL: Only uses matches BEFORE the prediction date
        This prevents data leakage (using future information to predict past)
        
        Gets the last N matches for a player before a specific date
        - Sorts by date descending (most recent first)
        - Takes only the top N matches
        - Returns None if no matches found
        
        This is essential for realistic model evaluation
        """
        date_col = 'tourney_date' if 'tourney_date' in self.rolling_data.columns else 'date'
        
        player_matches = self.rolling_data[
            (self.rolling_data['player_name'] == player_name) &
            (self.rolling_data[date_col] < match_date)
        ].sort_values(date_col, ascending=False).head(window)
        
        return player_matches if len(player_matches) > 0 else None
    
    def get_recent_surface_matches(self, player_name, match_date, surface, window):
        """
        Get recent matches on specific surface for a player before a given date
        """
        date_col = 'tourney_date' if 'tourney_date' in self.rolling_data.columns else 'date'
        
        player_matches = self.rolling_data[
            (self.rolling_data['player_name'] == player_name) &
            (self.rolling_data[date_col] < match_date) &
            (self.rolling_data['surface'] == surface)
        ].sort_values(date_col, ascending=False).head(window)
        
        return player_matches if len(player_matches) > 0 else None
    
    def prepare_model_data(self):
        """
        MODEL PREPARATION CHUNK:
        Converts raw features into ML-ready format
        
        Steps:
        1. Handle missing values and clean data
        2. Encode categorical variables (surface: Clay->0, Grass->1, Hard->2)
        3. Separate features (X) from target variable (y)
        4. Create feature name list for interpretability
        
        Input: Raw feature dataframe with mixed data types
        Output: Clean numpy arrays ready for XGBoost
        """
        if self.match_features is None:
            self.create_match_features()
        
        print("Preparing model data...")
        
        # Remove rows with missing target
        model_data = self.match_features.dropna(subset=['player_won']).copy()
        
        # Encode categorical variables
        categorical_cols = ['surface']
        for col in categorical_cols:
            if col in model_data.columns:
                le = LabelEncoder()
                model_data[col] = le.fit_transform(model_data[col].astype(str))
                self.label_encoders[col] = le
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = ['match_date', 'player_name', 'opponent_name', 'player_won']
        feature_cols = [col for col in model_data.columns if col not in exclude_cols]
        
        # Handle missing values
        X = model_data[feature_cols].fillna(0)
        y = model_data['player_won'].astype(int)
        
        self.feature_names = feature_cols
        
        print(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y, model_data
    
    def train_model(self, test_size=0.2, use_time_split=True):
        """
        MODEL TRAINING CHUNK:
        The core machine learning training process
        
        Key decision: TIME-BASED SPLIT vs RANDOM SPLIT
        - Time split: Train on 2010-2020, test on 2021-2024 (realistic)
        - Random split: Mix all years together (unrealistic but sometimes higher scores)
        
        XGBoost Parameters Explained:
        - max_depth=6: How complex each tree can be (prevents overfitting)
        - learning_rate=0.1: How fast the model learns (smaller = more stable)
        - n_estimators=200: Number of trees to build
        - subsample=0.8: Use 80% of data for each tree (prevents overfitting)
        - early_stopping: Stop if validation score doesn't improve for 20 rounds
        """
        X, y, model_data = self.prepare_model_data()
        
        # Time-based split (more realistic for time series)
        if use_time_split:
            model_data_sorted = model_data.sort_values('match_date')
            split_idx = int(len(model_data_sorted) * (1 - test_size))
            
            train_idx = model_data_sorted.index[:split_idx]
            test_idx = model_data_sorted.index[split_idx:]
            
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 20
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        self.plot_feature_importance()

        joblib.dump(self.model, 'xgb_tennis_model.pkl')
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_feature_importance(self, top_n=20):
        """
        MODEL INTERPRETATION CHUNK:
        Shows which features the model thinks are most important
        
        XGBoost automatically calculates feature importance based on:
        - How often a feature is used in splits
        - How much it improves the model when used
        
        This helps you understand:
        - Is the model using logical features? (recent form should be high)
        - Are there surprising patterns? (maybe surface matters less than expected)
        - Should you engineer new features?
        
        Creates horizontal bar chart for easy reading
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def predict_match(self, player1, player2, surface, match_date=None):
        """
        Predict outcome of a specific match
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        if match_date is None:
            match_date = datetime.now()
        
        # Create dummy match data (simplified)
        # In practice, you'd extract actual features for these players
        print(f"Predicting: {player1} vs {player2} on {surface}")
        print("Note: This is a simplified prediction - would need actual player data")
        
        return None
    
    def cross_validate(self, cv_folds=5):
        """
        MODEL VALIDATION CHUNK:
        Tests model performance more rigorously than single train/test split
        
        Uses TimeSeriesSplit instead of regular cross-validation:
        - Fold 1: Train on 2010-2015, test on 2016
        - Fold 2: Train on 2010-2017, test on 2018  
        - Fold 3: Train on 2010-2019, test on 2020
        - etc.
        
        This mimics real-world usage where you always predict future matches
        Regular CV would mix time periods unrealistically
        
        Returns average accuracy across all folds + standard deviation
        """
        X, y, _ = self.prepare_model_data()
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42
        )
        
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        print(f"\nCross-Validation Results:")
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual fold scores: {cv_scores}")
        
        return cv_scores


if __name__ == "__main__":
    aggregator = TennisPlayerAggregator("data/main")
    career_stats, rolling_data = aggregator.run_full_analysis()
    
    predictor = TennisMatchPredictor((career_stats, rolling_data))
    
    results = predictor.train_model(test_size=0.2, use_time_split=True)
    
    cv_scores = predictor.cross_validate(cv_folds=5)
    
    feature_importance = predictor.plot_feature_importance(top_n=15)
    