from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
from typing import List
import sys
import os 
from models.meta_learner import MetaLearner, load_model
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/model_output'))

input_dim = 3  # Adjust if needed
model_path = "models/model_output/meta_learner.pth"
model = MetaLearner(input_dim)
model = load_model(model, model_path)

app = FastAPI()

class PredictionRequest(BaseModel):
    features: List[float]

@app.post("/predict/")
async def predict(request: PredictionRequest):
    features = request.features
    if len(features) != input_dim:
        raise HTTPException(status_code=400, detail=f"Expected {input_dim} features, but got {len(features)}")
    
    input_df = pd.DataFrame([features])
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
    predicted_value = prediction.item()

    return {"prediction": predicted_value}
