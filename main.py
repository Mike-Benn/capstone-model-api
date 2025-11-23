from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
pass_model = joblib.load("models/passing_model.pkl")
rush_model = joblib.load("models/rushing_model.pkl")
app = FastAPI()

class Demographics(BaseModel):
    height: float
    weight: float

@app.post("/model")
def predict_yards(dems: Demographics):
    input_dataframe = pd.DataFrame([dems.model_dump()])

    pass_yards = pass_model.predict(input_dataframe)[0]
    rush_yards = rush_model.predict(input_dataframe)[0]

    return {
        "pred_pass_yards": pass_yards,
        "pred_rush_yards": rush_yards
    }