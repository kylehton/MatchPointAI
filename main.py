import fastapi as FastAPI

app = FastAPI()

@app.get("/")
def ping():
    return {"Status": "200 OK"}

@app.post("/predict")
async def predict_match(player1: str, player2: str):
    pass
