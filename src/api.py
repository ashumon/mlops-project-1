from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict_model import predict
import logging
import os

app = FastAPI()

# Create logs folder
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO
)

class InputData(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"message": "ML API running"}

@app.post("/predict")
def get_prediction(input: InputData):
    try:
        result = predict(input.data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logging.info("prediction_request=%s prediction=%s", input.data, result)
    return {"prediction": result}