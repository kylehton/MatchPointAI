import joblib

model = joblib.load('xgb_model.joblib')

def predict_match(p1: str, p2: str) -> float:
    