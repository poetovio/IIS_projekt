from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "electricity"
    / "prices"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"
SCALER_DIRECTORY = MODEL_DIRECTORY / "scalers"

WINDOW_SIZE = 24
TRAIN_RATIO = 0.8

FEATURES = [
    "price",
    "hour",
    "day_of_week",
    "day",
    "month",
    "year",
    "is_weekend",
]


def create_sequences(data, window_size):
    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_model(file_path):
    zone = file_path.stem

    print()
    print(f"Training model for {zone}...")

    df = pd.read_csv(
        file_path,
        parse_dates=["datetime_utc"],
    )

    df = df.sort_values("datetime_utc").reset_index(drop=True)

    df["is_weekend"] = df["is_weekend"].astype(int)

    data = df[FEATURES].astype(float).values

    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data[:split_index]
    test_data = data[split_index:]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(
        train_scaled,
        WINDOW_SIZE,
    )

    test_data_with_history = np.concatenate(
        [
            train_scaled[-WINDOW_SIZE:],
            test_scaled,
        ],
        axis=0,
    )

    X_test, y_test = create_sequences(
        test_data_with_history,
        WINDOW_SIZE,
    )

    print(f"Rows: {len(df)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Testing rows: {len(test_data)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    model = build_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        )
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False,
    )

    test_loss, test_mae = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test MSE: {test_loss:.6f}")

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    SCALER_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIRECTORY / f"{zone}.keras"
    scaler_path = SCALER_DIRECTORY / f"{zone}.pkl"

    model.save(model_path)

    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")


def main():
    files = sorted(
        DATA_DIRECTORY.glob("*.csv")
    )

    print(
        f"Found {len(files)} energy price files."
    )

    if not files:
        print("ERROR: No energy price files found.")
        return 1

    for file_path in files:
        train_model(file_path)

    print()
    print("All energy models trained successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())