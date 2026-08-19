from pathlib import Path
import pickle
import sqlite3
import uuid

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGRESSION_MODEL_DIRECTORY = (
    PROJECT_ROOT / "models" / "regression"
)

REGRESSION_SCALER_DIRECTORY = (
    REGRESSION_MODEL_DIRECTORY / "scalers"
)

CLASSIFICATION_MODEL_DIRECTORY = (
    PROJECT_ROOT / "models" / "classification"
)

MONITORING_DATABASE = (
    PROJECT_ROOT / "data" / "monitoring.db"
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
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

regression_models = {}
regression_scalers = {}

classification_models = {}
classification_scalers = {}


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


def get_available_zones():
    regression_zones = {
        path.stem
        for path in REGRESSION_MODEL_DIRECTORY.glob("*.keras")
    }

    classification_zones = {
        path.stem
        for path in CLASSIFICATION_MODEL_DIRECTORY.glob("*.keras")
    }

    return sorted(
        regression_zones & classification_zones
    )


def validate_zone(zone: str):
    available_zones = get_available_zones()

    if zone not in available_zones:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Zone '{zone}' is not available.",
                "available_zones": available_zones,
            },
        )


def load_regression_model(zone: str):
    validate_zone(zone)

    if zone in regression_models:
        return (
            regression_models[zone],
            regression_scalers[zone],
        )

    model_path = (
        REGRESSION_MODEL_DIRECTORY
        / f"{zone}.keras"
    )

    scaler_path = (
        REGRESSION_SCALER_DIRECTORY
        / f"{zone}.pkl"
    )

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Regression model for {zone} not found.",
        )

    if not scaler_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Regression scaler for {zone} not found.",
        )

    model = tf.keras.models.load_model(
        model_path
    )

    with open(
        scaler_path,
        "rb",
    ) as file:
        scaler = pickle.load(file)

    regression_models[zone] = model
    regression_scalers[zone] = scaler

    return model, scaler


def load_classification_model(zone: str):
    validate_zone(zone)

    if zone in classification_models:
        return (
            classification_models[zone],
            classification_scalers[zone],
        )

    model_path = (
        CLASSIFICATION_MODEL_DIRECTORY
        / f"{zone}.keras"
    )

    scaler_path = (
        CLASSIFICATION_MODEL_DIRECTORY
        / f"{zone}_scaler.pkl"
    )

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Classification model for {zone} not found.",
        )

    if not scaler_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Classification scaler for {zone} not found.",
        )

    model = tf.keras.models.load_model(
        model_path
    )

    with open(
        scaler_path,
        "rb",
    ) as file:
        scaler = pickle.load(file)

    classification_models[zone] = model
    classification_scalers[zone] = scaler

    return model, scaler


def init_monitoring_database():
    MONITORING_DATABASE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        MONITORING_DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            zone TEXT NOT NULL,
            model_type TEXT NOT NULL,
            prediction REAL,
            prediction_class TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()


def save_prediction(
    prediction_id,
    zone,
    model_type,
    prediction=None,
    prediction_class=None,
    confidence=None,
):
    connection = sqlite3.connect(
        MONITORING_DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            prediction_id,
            zone,
            model_type,
            prediction,
            prediction_class,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            zone,
            model_type,
            prediction,
            prediction_class,
            confidence,
        ),
    )

    connection.commit()
    connection.close()


@app.on_event("startup")
def startup():
    init_monitoring_database()


@app.get("/")
def root():
    return {
        "name": "Energy Price Intelligence API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/zones",
            "/models",
            "/models/{zone}",
            "/predict/regression/{zone}",
            "/predict/classification/{zone}",
        ],
    }


@app.get("/health")
def health():
    available_zones = get_available_zones()

    return {
        "status": "healthy",
        "regression_models_loaded": len(
            regression_models
        ),
        "classification_models_loaded": len(
            classification_models
        ),
        "available_zones": len(
            available_zones
        ),
        "monitoring_database": (
            MONITORING_DATABASE.exists()
        ),
    }


@app.get("/zones")
def zones():
    available_zones = get_available_zones()

    return {
        "zones": available_zones,
        "count": len(available_zones),
    }


@app.get("/models")
def models():
    available_zones = get_available_zones()

    return {
        "zones": available_zones,
        "count": len(available_zones),
        "regression": {
            "type": "LSTM regression",
            "window_size": WINDOW_SIZE,
            "features": FEATURES,
        },
        "classification": {
            "type": "Dense neural network",
            "classes": CLASS_NAMES,
            "features": FEATURES,
        },
    }


@app.get("/models/{zone}")
def model_info(zone: str):
    validate_zone(zone)

    regression_exists = (
        (
            REGRESSION_MODEL_DIRECTORY
            / f"{zone}.keras"
        ).exists()
    )

    regression_scaler_exists = (
        (
            REGRESSION_SCALER_DIRECTORY
            / f"{zone}.pkl"
        ).exists()
    )

    classification_exists = (
        (
            CLASSIFICATION_MODEL_DIRECTORY
            / f"{zone}.keras"
        ).exists()
    )

    classification_scaler_exists = (
        (
            CLASSIFICATION_MODEL_DIRECTORY
            / f"{zone}_scaler.pkl"
        ).exists()
    )

    return {
        "zone": zone,
        "regression": {
            "model": f"{zone}.keras",
            "type": "LSTM regression",
            "window_size": WINDOW_SIZE,
            "scaler_available": regression_scaler_exists,
            "available": regression_exists,
        },
        "classification": {
            "model": f"{zone}.keras",
            "type": "Dense neural network",
            "classes": CLASS_NAMES,
            "scaler_available": classification_scaler_exists,
            "available": classification_exists,
        },
    }


@app.post("/predict/regression/{zone}")
def predict_regression(
    zone: str,
    request: RegressionInput,
):
    model, scaler = load_regression_model(
        zone
    )

    data = np.asarray(
        request.data,
        dtype=float,
    )

    expected_shape = (
        WINDOW_SIZE,
        len(FEATURES),
    )

    if data.shape != expected_shape:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected input shape "
                f"{expected_shape}, "
                f"received {data.shape}."
            ),
        )

    scaled_data = scaler.transform(
        data
    )

    X = scaled_data.reshape(
        1,
        WINDOW_SIZE,
        len(FEATURES),
    )

    prediction_scaled = model.predict(
        X,
        verbose=0,
    )

    prediction_array = np.zeros(
        (1, len(FEATURES))
    )

    prediction_array[0, 0] = (
        prediction_scaled[0, 0]
    )

    prediction = scaler.inverse_transform(
        prediction_array
    )[0, 0]

    prediction_id = str(
        uuid.uuid4()
    )

    save_prediction(
        prediction_id=prediction_id,
        zone=zone,
        model_type="regression",
        prediction=float(prediction),
    )

    return {
        "prediction_id": prediction_id,
        "zone": zone,
        "prediction": float(prediction),
        "unit": "EUR/MWh",
        "model": "LSTM regression",
    }


@app.post("/predict/classification/{zone}")
def predict_classification(
    zone: str,
    request: ClassificationInput,
):
    model, scaler = load_classification_model(
        zone
    )

    features = np.array(
        [
            [
                request.price,
                request.hour,
                request.day_of_week,
                request.day,
                request.month,
                request.year,
                int(request.is_weekend),
            ]
        ],
        dtype=float,
    )

    scaled_features = scaler.transform(
        features
    )

    probabilities = model.predict(
        scaled_features,
        verbose=0,
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    prediction_class = (
        CLASS_NAMES[predicted_class]
    )

    confidence = float(
        probabilities[predicted_class]
    )

    prediction_id = str(
        uuid.uuid4()
    )

    save_prediction(
        prediction_id=prediction_id,
        zone=zone,
        model_type="classification",
        prediction_class=prediction_class,
        confidence=confidence,
    )

    return {
        "prediction_id": prediction_id,
        "zone": zone,
        "prediction": prediction_class,
        "class_id": predicted_class,
        "probabilities": {
            CLASS_NAMES[i]: float(
                probabilities[i]
            )
            for i in range(
                len(CLASS_NAMES)
            )
        },
        "model": "Dense neural network",
    }


@app.get("/monitoring/predictions")
def monitoring_predictions(
    limit: int = 100,
):
    limit = max(
        1,
        min(limit, 1000),
    )

    connection = sqlite3.connect(
        MONITORING_DATABASE
    )

    connection.row_factory = (
        sqlite3.Row
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = [
        dict(row)
        for row in cursor.fetchall()
    ]

    connection.close()

    return {
        "predictions": rows,
        "count": len(rows),
    }

@app.get("/data/{zone}")
def get_latest_data(zone: str):
    validate_zone(zone)

    data_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "electricity"
        / "prices"
        / f"{zone}.csv"
    )

    if not data_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data for {zone} not found.",
        )

    import pandas as pd

    df = pd.read_csv(
        data_file,
        parse_dates=["datetime_utc"],
    )

    df = df.sort_values(
        "datetime_utc"
    ).tail(WINDOW_SIZE)

    if len(df) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough data for {zone}. "
                f"Required {WINDOW_SIZE} observations."
            ),
        )

    return {
        "zone": zone,
        "data": df[
            [
                "datetime_utc",
                *FEATURES,
            ]
        ].to_dict(
            orient="records"
        ),
    }