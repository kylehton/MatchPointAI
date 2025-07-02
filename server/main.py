import fastapi as FastAPI
from model.predict_matchup import predict_matchup
from fastapi import FastAPI
from mangum import Mangum
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Database connection
sql_engine = create_engine(os.getenv('POSTGRES_NEON_STRING'))
player_table = os.getenv("PLAYER_TABLE")

@app.get("/")
def ping():
    return {"Status": "MatchPointAI pinging back!"}

@app.get("/players")
async def get_players():
    """Get all players from the database"""
    try:
        query = f"SELECT name FROM {player_table} ORDER BY name"
        with sql_engine.connect() as conn:
            result = conn.execute(text(query))
            players = [row[0] for row in result.fetchall()]
        return {"players": players}
    except Exception as e:
        return {"error": f"Failed to fetch players: {str(e)}"}

@app.post("/predict")
async def predict_match(p1: str, p2: str, surface: str):
    print("[PROCESS]: Predicting match with XGBoost Model...")
    
    try:
        prediction = predict_matchup(p1, p2, surface)
        if "error" in prediction:
            return {"error": prediction["error"]}
        return prediction
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

handler = Mangum(app)
