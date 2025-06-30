import fastapi as FastAPI
from model.predict_matchup import predict_matchup
from fastapi import FastAPI
from mangum import Mangum


app = FastAPI()

@app.get("/")
def ping():
    return {"Status": "MatchPointAI pinging back!"}

@app.post("/predict")
async def predict_match(p1: str, p2: str):
    print("[PROCESS]: Predicting match with XGBoost Model...")
    
    try:
        prediction = predict_matchup(p1, p2)
        if "error" in prediction:
            return {"error": prediction["error"]}
        return prediction
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

handler = Mangum(app)
