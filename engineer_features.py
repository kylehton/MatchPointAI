import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sqlalchemy import insert, create_engine, Table, MetaData, text

import dotenv

dotenv.load_dotenv()

sql_engine = create_engine(os.getenv('POSTGRES_NEON_STRING'))
player_table = os.getenv("PLAYER_TABLE")

def load_data(directory: str):
    """Load all match data from CSV files in the directory"""
    print("Loading data files...")
    dataframes = []
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    for filename in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(os.path.join(directory, filename))
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

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
            'height': None if pd.isna(row['height']) else float(row['height'])
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


def create_player_statistics(df):
    pass

create_player_rows()