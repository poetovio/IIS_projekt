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
        "*",
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

class RegressionEvaluationInput(BaseModel):
    prediction_id: str
    actual_value: float


class ClassificationEvaluationInput(BaseModel):
    prediction_id: str
    actual_class: str


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

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            zone TEXT NOT NULL,
            predicted_value REAL,
            actual_value REAL,
            error REAL,
            absolute_error REAL,
            squared_error REAL,
            predicted_class TEXT,
            actual_class TEXT,
            is_correct INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()

def get_db_connection():
    connection = sqlite3.connect(MONITORING_DATABASE)
    connection.row_factory = sqlite3.Row
    return connection

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
            "/admin/metrics",
            "/admin/predictions",
            "/admin/evaluations",
            "/admin/models",
            "/admin/evaluate/regression",
            "/admin/evaluate/classification",
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

@app.post("/admin/evaluate/regression")
def evaluate_regression(request: RegressionEvaluationInput):
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT prediction_id, zone, prediction
        FROM predictions
        WHERE prediction_id = ?
        AND model_type = 'regression'
        """,
        (request.prediction_id,),
    )

    prediction = cursor.fetchone()

    if prediction is None:
        connection.close()
        raise HTTPException(
            status_code=404,
            detail="Regression prediction not found.",
        )

    predicted_value = float(prediction["prediction"])
    actual_value = float(request.actual_value)
    error = predicted_value - actual_value
    absolute_error = abs(error)
    squared_error = error ** 2

    cursor.execute(
        """
        INSERT INTO evaluations (
            prediction_id,
            model_type,
            zone,
            predicted_value,
            actual_value,
            error,
            absolute_error,
            squared_error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction["prediction_id"],
            "regression",
            prediction["zone"],
            predicted_value,
            actual_value,
            error,
            absolute_error,
            squared_error,
        ),
    )

    connection.commit()
    connection.close()

    return {
        "prediction_id": prediction["prediction_id"],
        "zone": prediction["zone"],
        "model_type": "regression",
        "predicted_value": predicted_value,
        "actual_value": actual_value,
        "error": error,
        "absolute_error": absolute_error,
    }


@app.post("/admin/evaluate/classification")
def evaluate_classification(request: ClassificationEvaluationInput):
    actual_class = request.actual_class.upper()

    if actual_class not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid actual class.",
                "allowed_classes": CLASS_NAMES,
            },
        )

    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT prediction_id, zone, prediction_class
        FROM predictions
        WHERE prediction_id = ?
        AND model_type = 'classification'
        """,
        (request.prediction_id,),
    )

    prediction = cursor.fetchone()

    if prediction is None:
        connection.close()
        raise HTTPException(
            status_code=404,
            detail="Classification prediction not found.",
        )

    predicted_class = prediction["prediction_class"]
    is_correct = int(predicted_class == actual_class)

    cursor.execute(
        """
        INSERT INTO evaluations (
            prediction_id,
            model_type,
            zone,
            predicted_class,
            actual_class,
            is_correct
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            prediction["prediction_id"],
            "classification",
            prediction["zone"],
            predicted_class,
            actual_class,
            is_correct,
        ),
    )

    connection.commit()
    connection.close()

    return {
        "prediction_id": prediction["prediction_id"],
        "zone": prediction["zone"],
        "model_type": "classification",
        "predicted_class": predicted_class,
        "actual_class": actual_class,
        "correct": bool(is_correct),
    }


@app.get("/admin/metrics")
def admin_metrics():
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS count,
            AVG(absolute_error) AS mae,
            AVG(squared_error) AS mse
        FROM evaluations
        WHERE model_type = 'regression'
        """
    )
    regression = cursor.fetchone()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS count,
            AVG(is_correct) AS accuracy
        FROM evaluations
        WHERE model_type = 'classification'
        """
    )
    classification = cursor.fetchone()

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM predictions
        """
    )
    predictions_total = cursor.fetchone()["total"]

    cursor.execute(
        """
        SELECT model_type, COUNT(*) AS count
        FROM predictions
        GROUP BY model_type
        """
    )
    prediction_counts = {
        row["model_type"]: row["count"]
        for row in cursor.fetchall()
    }

    cursor.execute(
        """
        SELECT zone, COUNT(*) AS count
        FROM predictions
        GROUP BY zone
        ORDER BY count DESC
        """
    )
    predictions_by_zone = {
        row["zone"]: row["count"]
        for row in cursor.fetchall()
    }

    connection.close()

    regression_mae = (
        float(regression["mae"])
        if regression["mae"] is not None
        else None
    )
    regression_mse = (
        float(regression["mse"])
        if regression["mse"] is not None
        else None
    )
    regression_rmse = (
        float(regression_mse ** 0.5)
        if regression_mse is not None
        else None
    )
    classification_accuracy = (
        float(classification["accuracy"])
        if classification["accuracy"] is not None
        else None
    )

    return {
        "predictions": {
            "total": predictions_total,
            "regression": prediction_counts.get("regression", 0),
            "classification": prediction_counts.get("classification", 0),
            "by_zone": predictions_by_zone,
        },
        "regression": {
            "evaluated_predictions": regression["count"],
            "mae": regression_mae,
            "rmse": regression_rmse,
        },
        "classification": {
            "evaluated_predictions": classification["count"],
            "accuracy": classification_accuracy,
        },
    }


@app.get("/admin/predictions")
def admin_predictions(limit: int = 50):
    limit = max(1, min(limit, 1000))

    connection = get_db_connection()
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

    rows = [dict(row) for row in cursor.fetchall()]
    connection.close()

    return {
        "predictions": rows,
        "count": len(rows),
    }


@app.get("/admin/evaluations")
def admin_evaluations(limit: int = 100):
    limit = max(1, min(limit, 1000))

    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM evaluations
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = [dict(row) for row in cursor.fetchall()]
    connection.close()

    return {
        "evaluations": rows,
        "count": len(rows),
    }


@app.get("/admin/models")
def admin_models():
    available_zones = get_available_zones()

    return {
        "models": [
            {
                "name": "Energy Price Regression",
                "type": "LSTM",
                "task": "regression",
                "status": "production",
                "zones": available_zones,
            },
            {
                "name": "Energy Price Classification",
                "type": "Dense Neural Network",
                "task": "classification",
                "status": "production",
                "zones": available_zones,
            },
        ]
    }
