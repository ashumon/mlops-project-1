"""Prediction helpers for the placement API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


def _get_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "model.pkl"


@lru_cache(maxsize=1)
def load_model():
    model_path = _get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Train the model before calling /predict."
        )
    return joblib.load(model_path)


def predict(features_dict: dict) -> object:
    if not features_dict:
        raise ValueError("features_dict cannot be empty")

    frame = pd.DataFrame([features_dict])
    model = load_model()
    prediction = model.predict(frame)
    return prediction[0].item() if hasattr(prediction[0], "item") else prediction[0]
