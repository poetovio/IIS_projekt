from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGRESSION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "regression" / "SI.keras"
)

REGRESSION_SCALER_PATH = (
    PROJECT_ROOT / "models" / "regression" / "scalers" / "SI.pkl"
)

CLASSIFICATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "classification" / "SI.keras"
)

CLASSIFICATION_SCALER_PATH = (
    PROJECT_ROOT / "models" / "classification" / "SI_scaler.pkl"
)

WINDOW_SIZE = 24

FEATURES = [
    "price",
    "hour",
    "day_of_week",
    "day",
    "month",
    "year",
    "is_weekend",
]

CLASS_NAMES = [
    "LOW",
    "NORMAL",
    "HIGH",
]


app = FastAPI(
    title="Energy Price Intelligence API",
    description="Production API for electricity price prediction and classification.",
    version="1.0.0",
)


regression_model = None
regression_scaler = None
classification_model = None
classification_scaler = None


class RegressionInput(BaseModel):
    data: list[list[float]] = Field(
        ...,
        description="24 consecutive observations containing the seven model features.",
    )


class ClassificationInput(BaseModel):
    price: float
    hour: int
    day_of_week: int
    day: int
    month: int
    year: int
    is_weekend: bool


@app.on_event("startup")
def load_models():
    global regression_model
    global regression_scaler
    global classification_model
    global classification_scaler

    required_files = [
        REGRESSION_MODEL_PATH,
        REGRESSION_SCALER_PATH,
        CLASSIFICATION_MODEL_PATH,
        CLASSIFICATION_SCALER_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Missing model files: "
            + ", ".join(missing_files)
        )

    regression_model = tf.keras.models.load_model(
        REGRESSION_MODEL_PATH
    )

    with open(
        REGRESSION_SCALER_PATH,
        "rb",
    ) as file:
        regression_scaler = pickle.load(file)

    classification_model = tf.keras.models.load_model(
        CLASSIFICATION_MODEL_PATH
    )

    with open(
        CLASSIFICATION_SCALER_PATH,
        "rb",
    ) as file:
        classification_scaler = pickle.load(file)


@app.get("/")
def root():
    return {
        "name": "Energy Price Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/models",
            "/predict/regression",
            "/predict/classification",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "regression_model_loaded": regression_model is not None,
        "classification_model_loaded": classification_model is not None,
    }


@app.get("/models")
def models():
    return {
        "regression": {
            "model": "SI.keras",
            "type": "LSTM regression",
            "window_size": WINDOW_SIZE,
            "features": FEATURES,
        },
        "classification": {
            "model": "SI.keras",
            "type": "Dense neural network",
            "classes": CLASS_NAMES,
            "features": FEATURES,
        },
    }


@app.post("/predict/regression")
def predict_regression(request: RegressionInput):
    if regression_model is None or regression_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Regression model is not loaded.",
        )

    data = np.asarray(
        request.data,
        dtype=float,
    )

    if data.shape != (
        WINDOW_SIZE,
        len(FEATURES),
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected input shape "
                f"({WINDOW_SIZE}, {len(FEATURES)}), "
                f"received {data.shape}."
            ),
        )

    scaled_data = regression_scaler.transform(data)

    X = scaled_data.reshape(
        1,
        WINDOW_SIZE,
        len(FEATURES),
    )

    prediction_scaled = regression_model.predict(
        X,
        verbose=0,
    )

    prediction_array = np.zeros(
        (1, len(FEATURES))
    )

    prediction_array[0, 0] = prediction_scaled[0, 0]

    prediction = regression_scaler.inverse_transform(
        prediction_array
    )[0, 0]

    return {
        "zone": "SI",
        "prediction": float(prediction),
        "unit": "EUR/MWh",
        "model": "LSTM regression",
    }


@app.post("/predict/classification")
def predict_classification(
    request: ClassificationInput,
):
    if (
        classification_model is None
        or classification_scaler is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Classification model is not loaded.",
        )

    features = np.array(
        [[
            request.price,
            request.hour,
            request.day_of_week,
            request.day,
            request.month,
            request.year,
            int(request.is_weekend),
        ]],
        dtype=float,
    )

    scaled_features = classification_scaler.transform(
        features
    )

    probabilities = classification_model.predict(
        scaled_features,
        verbose=0,
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    return {
        "zone": "SI",
        "prediction": CLASS_NAMES[predicted_class],
        "class_id": predicted_class,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        },
        "model": "Dense neural network",
    }