import pandas as pd
import os
from tqdm import tqdm
from sqlalchemy import insert, create_engine, Table, MetaData, text, select

import dotenv

dotenv.load_dotenv()

sql_engine = create_engine(os.getenv('POSTGRES_NEON_STRING'))
player_table = os.getenv("PLAYER_TABLE")

def load_data(directory: str):
    print("Loading data files...")
    dataframes = []
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    for filename in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(os.path.join(directory, filename))
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Create player rows from the atp_players.csv file
# Will have features: name, hand, height
def create_player_rows():
    metadata = MetaData()
    # Reflect the existing table from the DB
    players = Table(player_table, metadata, autoload_with=sql_engine)

    # Reset the sequence and clear existing data
    with sql_engine.connect() as conn:
        # Delete all existing data
        conn.execute(players.delete())
        # Reset the sequence to 1 using text()
        conn.execute(text(f"ALTER SEQUENCE {player_table}_id_seq RESTART WITH 1"))
        conn.commit()
        print("Reset table and sequence")

    # Read CSV with explicit dtype for name columns to ensure they're strings
    all_players = pd.read_csv('data/atp_players.csv', 
                            dtype={'name_first': str, 'name_last': str},
                            low_memory=False)
    
    print(f"Total players in CSV: {len(all_players)}")
    print("\nSample of raw data:")
    print(all_players[['name_first', 'name_last', 'hand', 'height']].head())

    median_height = all_players['height'].median()

    # Process the data in pandas first
    all_players['player_name'] = all_players.apply(
        lambda row: f"{'' if pd.isna(row['name_first']) else row['name_first']} {'' if pd.isna(row['name_last']) else row['name_last']}".strip(),
        axis=1
    )
    
    # Check for empty names
    empty_names = all_players[all_players['player_name'].str.len() == 0]
    print(f"\nPlayers with empty names: {len(empty_names)}")
    if len(empty_names) > 0:
        print("Sample of empty names:")
        print(empty_names[['name_first', 'name_last']].head())

    # Convert hand to integer (1 for R, 0 for L)
    all_players['hand'] = (all_players['hand'] == 'R').astype(int)
    # Convert height to numeric, keeping NaN values
    all_players['height'] = pd.to_numeric(all_players['height'], errors='coerce')

    # Filter out rows with no valid name
    valid_players = all_players[all_players['player_name'].str.len() > 0].copy()
    print(f"\nPlayers after filtering empty names: {len(valid_players)}")
    
    # Prepare the data for bulk insert, ensuring proper types
    records = []
    for _, row in valid_players.iterrows():
        record = {
            'name': row['player_name'],
            'hand': int(row['hand']),
            'height': median_height if pd.isna(row['height']) else int(row['height'])
        }
        records.append(record)

    print(f"\nFinal number of records to insert: {len(records)}")
    print("\nSample of records to be inserted:")
    for record in records[:5]:
        print(record)

    # Perform bulk insert
    with sql_engine.connect() as conn:
        conn.execute(insert(players), records)
        conn.commit()
        
    # Verify the number of inserted records
    with sql_engine.connect() as conn:
        result = conn.execute(players.select()).fetchall()
        print(f"\nNumber of records in database after insert: {len(result)}")

# Create player statistics from all match history
# need to compute rolling averages for all avgs, since it is across time
# will have features: average_1st_in, avg_1st_won, avg_2nd_in, avg_2nd_won, avg_ace, avg_df, avg_bp_faced, avp_bp_saved, 
# highest_rank, overall_win_rate, and the 3 win rates by surface
def upload_player_statistics():
    metadata = MetaData()
    # Reflect the existing table from the DB
    players = Table(player_table, metadata, autoload_with=sql_engine)

    career_stats = pd.read_csv('tennis_career_stats.csv')
    #rolling_stats = pd.read_csv('tennis_rolling_data.csv')
    
    with sql_engine.connect() as conn:
        for row in tqdm(career_stats.itertuples(index=False)):
            # Extract values using dot notation for itertuples, more efficient than iterrows
            conn.execute(
                players.update().where(players.c.name == row.player_name).values(
                    first_in=row.career_avg_player_1st_serve_in,
                    first_won=row.career_avg_player_1st_serve_won,
                    second_won=row.career_avg_player_2nd_serve_won,
                    avg_ace=row.career_avg_player_ace,
                    avg_df=row.career_avg_player_df,
                    avg_bp_faced=row.career_avg_player_bp_faced,
                    avg_bp_saved=row.career_avg_player_bp_saved,
                    relative_first_in=row.latest_rolling_1st_serve_in,
                    relative_first_won=row.latest_rolling_1st_serve_won,
                    relative_second_won=row.latest_rolling_2nd_serve_won,
                    relative_avg_ace=row.latest_rolling_ace,
                    relative_avg_df=row.latest_rolling_df,
                    relative_bp_faced=row.latest_rolling_bp_faced,
                    relative_bp_saved=row.latest_rolling_bp_saved,
                    overall_wr=row.avg_win_rate,
                    hard_court_wr=row.hard_win_rate,
                    grass_court_wr=row.grass_win_rate,
                    clay_court_wr=row.clay_win_rate,
                    carpet_court_wr=row.carpet_win_rate,
                )
            )
            conn.commit()

def retrieve_player_stats(name: str):
    metadata = MetaData()
    players = Table(player_table, metadata, autoload_with=sql_engine)
    
    with sql_engine.connect() as conn:
        player = conn.execute(select(players).where(players.c.name == name))
        row = player.fetchone()
        return row



#upload_player_statistics()
retrieve_player_stats("Roger Federer")

