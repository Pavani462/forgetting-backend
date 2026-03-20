from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# =========================
# Enable CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load Model
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

features = ["time_gap", "difficulty", "retention"]

# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "Forgetting Prediction API Running"}

@app.post("/predict")
def predict(time_gap: float, difficulty: float, retention: float):

    input_df = pd.DataFrame(
        [[time_gap, difficulty, retention]],
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
        "forgetting_probability": round(float(prob), 3),
        "recommendation": recommendation
    }