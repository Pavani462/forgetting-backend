from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

features = ["time_gap", "difficulty", "retention"]

@app.get("/")
def home():
    return {"message": "Forgetting Prediction API Running"}

@app.post("/predict")
def predict(concept: str, difficulty: str):
    
    # 🔥 Convert difficulty
    difficulty_map = {
        "Easy": 0.3,
        "Medium": 0.6,
        "Hard": 0.9
    }

    difficulty_value = difficulty_map.get(difficulty, 0.6)

    # 🔥 Simulated values (for now)
    time_gap = 5   # assume 5 days since last study
    retention = 0.7  # assume some retention

    input_df = pd.DataFrame(
        [[time_gap, difficulty_value, retention]],
        columns=features
    )

    scaled_input = scaler.transform(input_df)
    prob = model.predict_proba(scaled_input)[0][1]

    if prob > 0.7:
        recommendation = "Revise immediately"
    elif prob > 0.4:
        recommendation = "Revise within 2 days"
    else:
        recommendation = "No immediate revision needed"

    return {
        "concept": concept,
        "forgetting_probability": round(float(prob), 3),
        "recommendation": recommendation,
        "retention": round(1 - prob, 3)
    }