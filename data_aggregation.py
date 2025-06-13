import glob
import os
import warnings
import pickle
import hashlib
from pathlib import Path
from typing import List, Union, Tuple, Optional
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


SURFACES: List[str] = {
    1: "Hard",
    2: "Clay", 
    3: "Grass",
    4: "Carpet"
}

SERVE_COLS: List[str] = [
    "1st_serve_in",
    "1st_serve_won",
    "2nd_serve_won",
    "serve_games",
    "bp_saved",
    "bp_faced",
]
CRITICAL_COLS = ["rolling_win_rate", "rolling_best_rank"]
DEFAULT_SIMILARITY_FEATS = ["player_rank", "age", "height", "hand"]


def _parse_yyyymmdd(col: pd.Series) -> pd.Series:
    """Parse integers / strings in yyyymmdd format to datetime."""
    return pd.to_datetime(col.astype(str), format="%Y%m%d", errors="coerce")

def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to memory-efficient dtypes"""
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            # Try to convert string columns to categories if they have limited unique values
            nunique = df[col].nunique()
            if nunique / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    
    return df

def _get_cache_key(data_path: Union[str, List[str]], **kwargs) -> str:
    """Generate cache key based on data path and parameters"""
    path_str = str(data_path) if isinstance(data_path, str) else str(sorted(data_path))
    params_str = str(sorted(kwargs.items()))
    return hashlib.md5((path_str + params_str).encode()).hexdigest()

###############################################################################
# Optimized KNN Imputation
###############################################################################
def knn_imputation_tennis_optimized(
    df: pd.DataFrame, 
    *, 
    n_neighbors: int = 5, 
    features_for_similarity: List[str] | None = None,
    max_sample_size: int = 10000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    Optimized KNN imputation with sampling for large datasets
    """
    if n_neighbors <= 0:
        return df.copy(), {}

    if features_for_similarity is None:
        features_for_similarity = DEFAULT_SIMILARITY_FEATS

    print(f"Starting KNN imputation with {n_neighbors} neighbors...")
    df_imp = df.copy()

    # Identify categorical columns to exclude from imputation
    categorical_cols = []
    for col in df_imp.columns:
        if df_imp[col].dtype == "object" or df_imp[col].dtype.name == 'category':
            categorical_cols.append(col)
    
    print(f"Excluding categorical columns from imputation: {categorical_cols}")

    # Encode categorical columns for similarity features only (not for imputation)
    cat_maps: dict[str, dict] = {}
    for col in categorical_cols:
        uniques = df_imp[col].dropna().unique()
        mapping = {v: i for i, v in enumerate(uniques)}
        cat_maps[col] = mapping
        # Only encode if it's used for similarity
        if col in features_for_similarity:
            df_imp[col] = df_imp[col].map(mapping)

    # Identify columns for similarity and statistics
    sim_cols = [c for c in features_for_similarity if c in df_imp.columns]
    exclude_cols = sim_cols + ["player_name", "opponent_name", "_match_date"] + categorical_cols
    stat_cols = [c for c in df_imp.columns if c not in exclude_cols]
    
    # Filter out columns that are all NaN or non-numeric
    valid_stat_cols = []
    rank_cols = []  # Track rank columns for rounding
    for col in stat_cols:
        # Convert to numeric if possible
        df_imp[col] = pd.to_numeric(df_imp[col], errors='coerce')
        # Check if column has any non-NaN values
        if df_imp[col].notna().any():
            valid_stat_cols.append(col)
            # Identify rank columns for rounding
            if 'rank' in col.lower():
                rank_cols.append(col)
        else:
            print(f"Warning: Skipping column '{col}' - all values are NaN")
    
    stat_cols = valid_stat_cols
    
    if not stat_cols:
        print("Warning: No valid statistical columns found for imputation")
        return df.copy(), cat_maps

    print(f"Using {len(sim_cols)} similarity features: {sim_cols}")
    print(f"Imputing {len(stat_cols)} statistical features: {stat_cols[:5]}..." + 
          (f" and {len(stat_cols)-5} more" if len(stat_cols) > 5 else ""))
    if rank_cols:
        print(f"Rank columns that will be rounded to integers: {rank_cols}")

    # Sample data if too large for efficient KNN
    use_sampling = len(df_imp) > max_sample_size
    if use_sampling:
        print(f"Dataset has {len(df_imp):,} rows. Sampling {max_sample_size:,} for KNN fitting...")
        np.random.seed(random_state)
        sample_idx = np.random.choice(len(df_imp), max_sample_size, replace=False)
        df_sample = df_imp.iloc[sample_idx].copy()
    else:
        df_sample = df_imp.copy()

    # Prepare similarity features
    if sim_cols:
        sim_data_sample = df_sample[sim_cols].fillna(df_sample[sim_cols].median())
        sim_data_full = df_imp[sim_cols].fillna(df_imp[sim_cols].median())
        
        # Scale similarity features
        scaler = StandardScaler()
        sim_scaled_sample = scaler.fit_transform(sim_data_sample)
        sim_scaled_full = scaler.transform(sim_data_full)
    else:
        print("Warning: No similarity features found")
        sim_scaled_sample = np.empty((len(df_sample), 0))
        sim_scaled_full = np.empty((len(df_imp), 0))

    # Prepare statistical features for imputation
    stat_data_sample = df_sample[stat_cols].values
    stat_data_full = df_imp[stat_cols].values

    # Fit imputer on sample
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    
    # Combine similarity and statistical features for sample
    if sim_scaled_sample.shape[1] > 0:
        combined_sample = np.column_stack([sim_scaled_sample, stat_data_sample])
        combined_full = np.column_stack([sim_scaled_full, stat_data_full])
        n_sim_features = sim_scaled_sample.shape[1]
    else:
        combined_sample = stat_data_sample
        combined_full = stat_data_full
        n_sim_features = 0
    
    # Check for any remaining issues
    if combined_sample.shape[1] == 0:
        print("Error: No features available for imputation")
        return df.copy(), cat_maps
    
    print(f"Training KNN imputer on {combined_sample.shape[0]:,} samples with {combined_sample.shape[1]} features...")
    
    try:
        imputer.fit(combined_sample)
        print("Applying imputation to full dataset...")
        imputed_full = imputer.transform(combined_full)
        
        # Extract imputed statistical columns
        df_out = df.copy()
        if n_sim_features > 0:
            imputed_stats = imputed_full[:, n_sim_features:]
        else:
            imputed_stats = imputed_full
        
        # Verify dimensions match
        if imputed_stats.shape[1] != len(stat_cols):
            print(f"Warning: Dimension mismatch. Expected {len(stat_cols)} columns, got {imputed_stats.shape[1]}")
            # Handle the mismatch by taking the minimum
            n_cols_to_use = min(imputed_stats.shape[1], len(stat_cols))
            for idx in range(n_cols_to_use):
                col = stat_cols[idx]
                df_out[col] = imputed_stats[:, idx]
        else:
            for idx, col in enumerate(stat_cols):
                df_out[col] = imputed_stats[:, idx]
                
                # Round rank columns to integers
                if col in rank_cols:
                    df_out[col] = df_out[col].round().astype('Int64')  # Use Int64 to handle NaN values
                    print(f"Rounded {col} to integers")

        print("✓ KNN imputation completed successfully")
        return df_out, cat_maps
        
    except Exception as e:
        print(f"Error during KNN imputation: {str(e)}")
        print("Returning original data without imputation")
        return df.copy(), cat_maps

###############################################################################
# Parallel processing helper
###############################################################################

def _process_player_rolling(args) -> pd.DataFrame:
    """Process rolling stats for a chunk of players"""
    player_data, window = args
    
    frames = []
    for player, grp in player_data.groupby("player_name"):
        grp = grp.sort_values("_match_date").copy()
        
        # Basic rolling stats
        grp["rolling_win_rate"] = grp["won_match"].rolling(window, min_periods=1).mean()
        grp["rolling_best_rank"] = pd.to_numeric(grp["player_rank"], errors="coerce").rolling(window, min_periods=1).min()
        
        # Surface-specific win rates
        for surf in SURFACES:
            mask = grp["surface"].str.strip().str.title() == surf
            col = f"rolling_win_rate_{SURFACES[surf].lower()}"
            grp[col] = np.nan
            if mask.any():
                grp.loc[mask, col] = grp.loc[mask, "won_match"].rolling(window, min_periods=1).mean().values
        
        # Serve stats
        for stat in SERVE_COLS[:3]:
            player_col = f"player_{stat}"
            if player_col in grp.columns:
                grp[f"rolling_{stat}"] = pd.to_numeric(grp[player_col], errors="coerce").rolling(window, min_periods=1).mean()
        
        frames.append(grp)
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

###############################################################################
# Main Optimized Aggregator class
###############################################################################

class TennisPlayerAggregatorOptimized:
    def __init__(self, data_path: Union[str, List[str]], cache_dir: str = "./cache"):
        self.data_path = data_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.combined_data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Caching system
    # ------------------------------------------------------------------

    def _get_cache_path(self, cache_key: str, suffix: str = "") -> Path:
        """Get cache file path"""
        return self.cache_dir / f"tennis_cache_{cache_key}{suffix}.pkl"

    def _save_to_cache(self, data: any, cache_key: str, suffix: str = ""):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key, suffix)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self, cache_key: str, suffix: str = "") -> any:
        """Load data from cache"""
        cache_path = self._get_cache_path(cache_key, suffix)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                # Remove corrupted cache
                cache_path.unlink(missing_ok=True)
        return None

    def clean_dataframe(self, df):
        df = df.copy()
        
        # Convert empty strings to NaN for consistency
        df = df.replace('', np.nan)
        
        # Handle surface column specifically - convert numeric to string
        if 'surface' in df.columns:
            # If surface is numeric, map to string names
            if df['surface'].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df['surface']):
                df['surface'] = df['surface'].map(SURFACES).fillna('Unknown')
            else:
                # If already string, just fill missing values
                df['surface'] = df['surface'].fillna('Unknown')
        
        # Handle player names
        name_cols = [col for col in df.columns if 'name' in col.lower()]
        for col in name_cols:
            df[col] = df[col].fillna('Unknown Player')
        
        return df

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        *,
        window_size: int = 20,
        min_matches: int = 10,
        drop_na: bool = False,
        knn_neighbors: int = 0,
        use_cache: bool = True,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
        knn_sample_size: int = 10000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        # Generate cache key
        cache_params = {
            'window_size': window_size,
            'min_matches': min_matches,
            'drop_na': drop_na,
            'knn_neighbors': knn_neighbors,
            'knn_sample_size': knn_sample_size,
        }
        cache_key = _get_cache_key(self.data_path, **cache_params)

        # Try to load from cache
        if use_cache:
            cached_result = self._load_from_cache(cache_key, "_full_analysis")
            if cached_result is not None:
                print("✓ Loaded results from cache")
                return cached_result

        # Load and process data
        if self.combined_data is None:
            self._load_datasets(use_cache=use_cache)

        player_records = self._create_player_match_records()
        
        # Rolling stats calculation
        if use_parallel and len(player_records["player_name"].unique()) > 100:
            rolling_raw = self._calculate_rolling_stats_parallel(player_records, window_size, max_workers)
        else:
            rolling_raw = self._calculate_rolling_stats_vectorized(player_records, window_size)

        # Missing data summary
        self._missing_summary(rolling_raw)

        # Handle missing data
        rolling = rolling_raw.copy()
        if drop_na:
            before = len(rolling)
            rolling = rolling.dropna(subset=CRITICAL_COLS)
            print(f"Dropped {before - len(rolling):,} rows missing {CRITICAL_COLS}")

        # KNN imputation
        if knn_neighbors > 0:
            rolling_imputed, _ = knn_imputation_tennis_optimized(
                rolling, 
                n_neighbors=knn_neighbors,
                max_sample_size=knn_sample_size
            )
        else:
            rolling_imputed = rolling

        # Career aggregation
        career = self._aggregate_career_stats_vectorized(rolling_imputed)
        if not career.empty:
            career = career.query("total_matches >= @min_matches")
            career = career.sort_values("total_matches", ascending=False)

        result = (career, rolling_raw, rolling_imputed)

        # Save to cache
        if use_cache:
            self._save_to_cache(result, cache_key, "_full_analysis")
            print("✓ Results saved to cache")

        return result

    # ------------------------------------------------------------------
    # 1. Optimized data loading
    # ------------------------------------------------------------------

    def _load_datasets(self, pattern: str = "*.csv", chunksize: int = 50000, use_cache: bool = True) -> None:
        cache_key = _get_cache_key(self.data_path, pattern=pattern)
        
        # Try cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key, "_raw_data")
            if cached_data is not None:
                self.combined_data = cached_data
                print(f"✓ Loaded {len(self.combined_data):,} rows from cache")
                return

        # Load from files
        paths = self.data_path if isinstance(self.data_path, list) else glob.glob(os.path.join(self.data_path, pattern))
        if not paths:
            raise FileNotFoundError("No CSV files found at given data_path")

        print(f"Loading {len(paths)} CSV files...")
        frames = []
        
        for path in tqdm(paths, desc="Loading CSVs"):
            try:
                # Try to load entire file first
                df = pd.read_csv(path, low_memory=False)
                frames.append(df)
            except MemoryError:
                # Fall back to chunked loading for very large files
                chunks = pd.read_csv(path, chunksize=chunksize, low_memory=False)
                for chunk in chunks:
                    frames.append(chunk)

        self.combined_data = pd.concat(frames, ignore_index=True)
        
        # Optimize and prepare data
        self._standardise_columns()
        self._parse_and_sort_dates()
        self._coerce_numeric()
        self.combined_data = _optimize_dtypes(self.combined_data)
        
        print(f"✓ Loaded {len(self.combined_data):,} rows from {len(paths)} file(s)")
        
        # Cache the processed data
        if use_cache:
            self._save_to_cache(self.combined_data, cache_key, "_raw_data")

    def _standardise_columns(self):
        mapping = {
            "tourney_date": "tourney_date",
            "date": "date",
            "surface": "surface",
            "winner_name": "winner_name",
            "loser_name": "loser_name",
            "winner_rank": "winner_rank",
            "loser_rank": "loser_rank",
            "w_1stIn": "winner_1st_serve_in",
            "w_1stWon": "winner_1st_serve_won",
            "w_2ndWon": "winner_2nd_serve_won",
            "w_SvGms": "winner_serve_games",
            "w_bpSaved": "winner_bp_saved",
            "w_bpFaced": "winner_bp_faced",
            "l_1stIn": "loser_1st_serve_in",
            "l_1stWon": "loser_1st_serve_won",
            "l_2ndWon": "loser_2nd_serve_won",
            "l_SvGms": "loser_serve_games",
            "l_bpSaved": "loser_bp_saved",
            "l_bpFaced": "loser_bp_faced",
        }
        self.combined_data.rename(columns={k: v for k, v in mapping.items() if k in self.combined_data.columns}, inplace=True)

    def _parse_and_sort_dates(self):
        if "tourney_date" in self.combined_data.columns:
            self.combined_data["_match_date"] = _parse_yyyymmdd(self.combined_data["tourney_date"])
        elif "date" in self.combined_data.columns:
            self.combined_data["_match_date"] = pd.to_datetime(self.combined_data["date"], errors="coerce")
        else:
            raise KeyError("Date column not found")
        self.combined_data.sort_values("_match_date", inplace=True, ignore_index=True)

    def _coerce_numeric(self):
        num_cols = [c for c in self.combined_data.columns if any(k in c for k in ("_serve", "_rank", "_bp", "_games"))]
        self.combined_data[num_cols] = self.combined_data[num_cols].apply(pd.to_numeric, errors="coerce")

    # ------------------------------------------------------------------
    # 2. Player records (same as original)
    # ------------------------------------------------------------------

    def _create_player_match_records(self) -> pd.DataFrame:
        date = "_match_date"
        base = [date, "surface", "player_name", "opponent_name", "player_rank", "opponent_rank", "won_match", "match_result"]

        def _expand(side: str):
            other = "winner" if side == "loser" else "loser"
            df = self.combined_data.copy()
            df["player_name"] = df[f"{side}_name"]
            df["opponent_name"] = df[f"{other}_name"]
            df["player_rank"] = df[f"{side}_rank"]
            df["opponent_rank"] = df[f"{other}_rank"]
            df["won_match"] = 1 if side == "winner" else 0
            df["match_result"] = "W" if side == "winner" else "L"
            for col in SERVE_COLS:
                src = f"{side}_{col}"
                if src in df.columns:
                    df[f"player_{col}"] = df[src]
            return df[[c for c in base + [f"player_{c}" for c in SERVE_COLS] if c in df.columns]]

        return pd.concat([_expand("winner"), _expand("loser")], ignore_index=True).sort_values(["player_name", date], ignore_index=True)

    # ------------------------------------------------------------------
    # 3. Optimized rolling stats
    # ------------------------------------------------------------------

        # Also update the vectorized rolling stats function to handle the surface conversion:
    def _calculate_rolling_stats_vectorized(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Vectorized rolling calculations - much faster than player-by-player loops"""
        print("Calculating rolling stats (vectorized)...")
        df = self.clean_dataframe(df)  # This will now handle numeric surfaces properly
        df = df.sort_values(["player_name", "_match_date"]).copy()
        
        # Basic rolling stats - vectorized
        df["rolling_win_rate"] = df.groupby("player_name")["won_match"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        df["rolling_best_rank"] = df.groupby("player_name")["player_rank"].transform(
            lambda x: pd.to_numeric(x, errors="coerce").rolling(window, min_periods=1).min()
        )
        
        # Surface-specific rolling win rates
        for surf in SURFACES:
            surf_lower =SURFACES[surf].lower()
            # Now surface should be string after clean_dataframe conversion
            mask = df["surface"] == surf  # Simplified comparison since surface is now properly mapped
            
            # Create a temporary column for surface-specific wins
            df[f"_temp_{surf_lower}_win"] = df["won_match"].where(mask)
            
            # Calculate rolling mean for each player on this surface
            df[f"rolling_win_rate_{surf_lower}"] = df.groupby("player_name")[f"_temp_{surf_lower}_win"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Clean up temporary column
            df.drop(f"_temp_{surf_lower}_win", axis=1, inplace=True)
        
        # Serve stats rolling means
        for stat in SERVE_COLS[:3]:
            player_col = f"player_{stat}"
            if player_col in df.columns:
                df[f"rolling_{stat}"] = df.groupby("player_name")[player_col].transform(
                    lambda x: pd.to_numeric(x, errors="coerce").rolling(window, min_periods=1).mean()
                )
        
        print("✓ Rolling stats calculated")
        return df

    # Update the career aggregation function as well:
    def _aggregate_career_stats_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized career aggregation - much faster than manual loops"""
        print("Aggregating career stats...")
        
        # Basic aggregations
        career_basic = df.groupby("player_name").agg({
            "_match_date": ["min", "max"],
            "won_match": ["count", "sum", "mean"],
            "player_rank": lambda x: pd.to_numeric(x, errors="coerce").min()
        }).round(4)
        
        # Flatten column names
        career_basic.columns = ["career_start", "career_end", "total_matches", "total_wins", "career_win_rate", "highest_ranking"]
        career_basic = career_basic.reset_index()
        
        # Get all rolling columns (excluding basic ones already handled)
        rolling_cols = [col for col in df.columns if col.startswith('rolling_') and col not in ['rolling_win_rate', 'rolling_best_rank']]
        
        # Get all player-specific columns (like player_1st_serve_in, etc.)
        player_cols = [col for col in df.columns if col.startswith('player_') and col not in ['player_name', 'player_rank']]
        
        # Get all opponent columns
        opponent_cols = [col for col in df.columns if col.startswith('opponent_') and col not in ['opponent_name']]
        
        # Get other numeric columns that might be useful
        other_numeric_cols = [col for col in df.columns if col not in ['player_name', 'opponent_name', '_match_date', 'surface', 'match_result'] 
                             and col not in rolling_cols + player_cols + opponent_cols
                             and df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']]
        
        print(f"Adding {len(rolling_cols)} rolling features to career stats")
        print(f"Adding {len(player_cols)} player-specific features to career stats")
        print(f"Adding {len(opponent_cols)} opponent features to career stats")
        print(f"Adding {len(other_numeric_cols)} other numeric features to career stats")
        
        # Add latest rolling stats for each player
        if rolling_cols:
            # Get the latest values for each rolling column per player
            latest_rolling_data = []
            for player in df['player_name'].unique():
                player_data = df[df['player_name'] == player].sort_values('_match_date')
                latest_values = {}
                for col in rolling_cols:
                    non_null_values = player_data[col].dropna()
                    latest_values[col] = non_null_values.iloc[-1] if len(non_null_values) > 0 else np.nan
                latest_values['player_name'] = player
                latest_rolling_data.append(latest_values)
            
            latest_rolling = pd.DataFrame(latest_rolling_data)
            
            # Rename columns to indicate they're latest values
            latest_rolling.columns = [f"latest_{col}" if col != 'player_name' else col for col in latest_rolling.columns]
            career_basic = career_basic.merge(latest_rolling, on="player_name", how="left")
        
        # Add career averages for player-specific stats
        if player_cols:
            career_player_stats = df.groupby("player_name")[player_cols].mean().round(4)
            career_player_stats.columns = [f"career_avg_{col}" for col in player_cols]
            career_basic = career_basic.merge(career_player_stats, left_on="player_name", right_index=True, how="left")
        
        # Add career averages for opponent stats
        if opponent_cols:
            career_opponent_stats = df.groupby("player_name")[opponent_cols].mean().round(4)
            career_opponent_stats.columns = [f"career_avg_{col}" for col in opponent_cols]
            career_basic = career_basic.merge(career_opponent_stats, left_on="player_name", right_index=True, how="left")
        
        # Add career averages for other numeric stats
        if other_numeric_cols:
            career_other_stats = df.groupby("player_name")[other_numeric_cols].mean().round(4)
            career_other_stats.columns = [f"career_avg_{col}" for col in other_numeric_cols]
            career_basic = career_basic.merge(career_other_stats, left_on="player_name", right_index=True, how="left")
        
        # Surface-specific stats (keep existing logic)
        for surf in SURFACES:
            surf_lower = SURFACES[surf].lower()
            # Use exact match since surface is now properly mapped to strings
            surf_data = df[df["surface"] == surf].groupby("player_name").agg({
                "won_match": ["count", "sum", "mean"],
                f"rolling_win_rate_{surf_lower}": lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan
            }).round(4)
            
            if not surf_data.empty:
                surf_data.columns = [f"{surf_lower}_matches", f"{surf_lower}_wins", f"{surf_lower}_win_rate", f"{surf_lower}_rolling_win_rate_latest"]
                career_basic = career_basic.merge(surf_data, left_on="player_name", right_index=True, how="left")
        
        # Add latest values for basic rolling stats (these might have been overwritten)
        latest_basic_rolling = df.groupby("player_name").agg({
            "rolling_win_rate": lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan,
            "rolling_best_rank": lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan
        }).round(4)
        
        latest_basic_rolling.columns = ["latest_rolling_win_rate", "latest_rolling_best_rank"]
        career_basic = career_basic.merge(latest_basic_rolling, left_on="player_name", right_index=True, how="left")
        
        print("✓ Career stats aggregated")
        return career_basic

    
    def _calculate_rolling_stats_parallel(self, df: pd.DataFrame, window: int, max_workers: Optional[int] = None) -> pd.DataFrame:
        """Parallel rolling calculations for large datasets"""
        print("Calculating rolling stats (parallel)...")
        
        players = df["player_name"].unique()
        n_cores = max_workers or min(mp.cpu_count() - 1, 8)
        n_cores = min(n_cores, len(players))  # Don't use more cores than players
        
        # Split players into chunks
        chunk_size = max(1, len(players) // n_cores)
        player_chunks = [players[i:i + chunk_size] for i in range(0, len(players), chunk_size)]
        
        # Create data chunks
        data_chunks = []
        for chunk in player_chunks:
            chunk_data = df[df["player_name"].isin(chunk)]
            data_chunks.append((chunk_data, window))
        
        # Process in parallel
        with Pool(n_cores) as pool:
            results = pool.map(_process_player_rolling, data_chunks)
        
        result = pd.concat([r for r in results if not r.empty], ignore_index=True)
        print("✓ Rolling stats calculated (parallel)")
        return result

    # ------------------------------------------------------------------
    # 5. Missing data report (same as original)
    # ------------------------------------------------------------------

    def _missing_summary(self, df: pd.DataFrame):
        miss = df.isna().sum()
        perc = miss / len(df) * 100
        summary = (
            pd.DataFrame({"missing_count": miss, "missing_percent": perc})
            .query("missing_count > 0")
            .sort_values("missing_percent", ascending=False)
        )
        summary.to_csv("rolling_missing_report.csv")
        print("\nTop missing columns (%):\n", summary.head(15))

###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("ATP/WTA rolling & career stats generator (optimized)")
    parser.add_argument("data_path")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--min", type=int, default=50)
    parser.add_argument("--drop-na", action="store_true")
    parser.add_argument("--knn-neighbors", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true", help="Use caching system")
    parser.add_argument("--use-parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--max-workers", type=int, help="Max parallel workers")
    parser.add_argument("--knn-sample-size", type=int, default=10000, help="Max samples for KNN fitting")
    args = parser.parse_args()

    agg = TennisPlayerAggregatorOptimized(args.data_path)
    career_df, rolling_raw, rolling_imp = agg.run_full_analysis(
        window_size=args.window,
        min_matches=args.min,
        drop_na=args.drop_na,
        knn_neighbors=args.knn_neighbors,
        use_cache=args.use_cache,
        use_parallel=args.use_parallel,
        max_workers=args.max_workers,
        knn_sample_size=args.knn_sample_size,
    )

    career_df.to_csv("tennis_career_stats.csv", index=False)
    rolling_raw.to_csv("tennis_rolling_data.csv", index=False)
    if args.knn_neighbors > 0:
        rolling_imp.to_csv("tennis_rolling_data_imputed.csv", index=False)
        print("Saved tennis_career_stats.csv | tennis_rolling_data.csv | tennis_rolling_data_imputed.csv | rolling_missing_report.csv")
    else:
        print("Saved tennis_career_stats.csv | tennis_rolling_data.csv | rolling_missing_report.csv")