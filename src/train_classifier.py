from pathlib import Path
import os
import pickle
import random

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "electricity"
    / "prices"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "classification"
)

PARAMS_PATH = PROJECT_ROOT / "params.yaml"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set.")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)["classification"]


TEST_RATIO = params["test_ratio"]
VALIDATION_RATIO = params["validation_ratio"]
EPOCHS = params["epochs"]
BATCH_SIZE = params["batch_size"]
RANDOM_STATE = params["random_state"]
LEARNING_RATE = params["learning_rate"]
PATIENCE = params["patience"]
LOW_QUANTILE = params["low_quantile"]
HIGH_QUANTILE = params["high_quantile"]


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


def create_target(
    df,
    low_threshold,
    high_threshold,
):
    next_price = df["price"].shift(-1)

    target = pd.Series(
        np.nan,
        index=df.index,
    )

    target[next_price < low_threshold] = 0

    target[
        (next_price >= low_threshold)
        & (next_price < high_threshold)
    ] = 1

    target[next_price >= high_threshold] = 2

    return target


def build_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape
            ),
            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                3,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_classifier(file_path):
    zone = file_path.stem

    print()
    print("=" * 60)
    print(f"Training classification model for {zone}")
    print("=" * 60)

    os.environ["PYTHONHASHSEED"] = str(
        RANDOM_STATE
    )

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    df = pd.read_csv(
        file_path,
        parse_dates=["datetime_utc"],
    )

    df = (
        df.sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    df["is_weekend"] = df[
        "is_weekend"
    ].astype(int)

    split_index = int(
        len(df) * (1 - TEST_RATIO)
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    low_threshold = train_df[
        "price"
    ].quantile(LOW_QUANTILE)

    high_threshold = train_df[
        "price"
    ].quantile(HIGH_QUANTILE)

    train_df["target"] = create_target(
        train_df,
        low_threshold,
        high_threshold,
    )

    test_df["target"] = create_target(
        test_df,
        low_threshold,
        high_threshold,
    )

    train_df = train_df.dropna(
        subset=["target"]
    )

    test_df = test_df.dropna(
        subset=["target"]
    )

    X_train = train_df[
        FEATURES
    ].astype(float)

    y_train = train_df[
        "target"
    ].astype(int)

    X_test = test_df[
        FEATURES
    ].astype(float)

    y_test = test_df[
        "target"
    ].astype(int)

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Low threshold: {low_threshold:.4f}"
    )

    print(
        f"High threshold: {high_threshold:.4f}"
    )

    mlflow.set_experiment(
        "Energy_Price_Classification"
    )

    with mlflow.start_run(
        run_name=f"classification_{zone}"
    ):
        mlflow.tensorflow.autolog(
            log_models=False
        )

        mlflow.log_params(
            {
                "zone": zone,
                "test_ratio": TEST_RATIO,
                "validation_ratio": VALIDATION_RATIO,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "random_state": RANDOM_STATE,
                "learning_rate": LEARNING_RATE,
                "patience": PATIENCE,
                "low_quantile": LOW_QUANTILE,
                "high_quantile": HIGH_QUANTILE,
                "low_threshold": float(
                    low_threshold
                ),
                "high_threshold": float(
                    high_threshold
                ),
                "model_type": (
                    "Dense Neural Network"
                ),
                "classes": (
                    "LOW,NORMAL,HIGH"
                ),
                "features": ",".join(
                    FEATURES
                ),
            }
        )

        model = build_model(
            input_shape=(
                X_train.shape[1],
            )
        )

        early_stopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            )
        )

        model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_RATIO,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1,
            shuffle=False,
        )

        probabilities = model.predict(
            X_test,
            verbose=0,
        )

        y_pred = np.argmax(
            probabilities,
            axis=1,
        )

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            y_pred,
        )

        print(
            f"Test Accuracy: {accuracy:.6f}"
        )

        print(
            f"Test Precision: {precision:.6f}"
        )

        print(
            f"Test Recall: {recall:.6f}"
        )

        print(
            f"Test F1: {f1:.6f}"
        )

        print("Confusion matrix:")

        print(matrix)

        mlflow.log_metrics(
            {
                "test_accuracy": float(
                    accuracy
                ),
                "test_precision": float(
                    precision
                ),
                "test_recall": float(
                    recall
                ),
                "test_f1": float(
                    f1
                ),
            }
        )

        MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            MODEL_DIRECTORY
            / f"{zone}.keras"
        )

        scaler_path = (
            MODEL_DIRECTORY
            / f"{zone}_scaler.pkl"
        )

        model.save(model_path)

        with open(
            scaler_path,
            "wb",
        ) as file:
            pickle.dump(
                scaler,
                file,
            )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="models",
        )

        mlflow.log_artifact(
            str(scaler_path),
            artifact_path="scalers",
        )

        print(
            f"Model saved: {model_path}"
        )

        print(
            f"Scaler saved: {scaler_path}"
        )


def main():
    files = sorted(
        DATA_DIRECTORY.glob("*.csv")
    )

    print(
        f"Found {len(files)} energy price files."
    )

    if not files:
        raise FileNotFoundError(
            "No energy price files found."
        )

    for file_path in files:
        train_classifier(file_path)

    print()
    print(
        "All classification models trained successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())