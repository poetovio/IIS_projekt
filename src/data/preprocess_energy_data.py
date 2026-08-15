from pathlib import Path

import pandas as pd


def preprocess_energy_data():
    input_directory = Path(
        "data/raw/electricity/prices"
    )

    output_directory = Path(
        "data/processed/electricity/prices"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for input_file in input_directory.glob("*.csv"):
        output_file = output_directory / input_file.name

        raw_df = pd.read_csv(
            input_file,
            parse_dates=["datetime_utc"]
        )

        if output_file.exists():
            existing_df = pd.read_csv(
                output_file,
                parse_dates=["datetime_utc"]
            )
        else:
            existing_df = pd.DataFrame()

        df = pd.concat(
            [existing_df, raw_df],
            ignore_index=True
        )

        df = df.drop_duplicates(
            subset=["datetime_utc"]
        )

        df = df.sort_values(
            "datetime_utc"
        )

        df = df.rename(
            columns={"value": "price"}
        )

        df["hour"] = df["datetime_utc"].dt.hour
        df["day_of_week"] = df["datetime_utc"].dt.dayofweek
        df["day"] = df["datetime_utc"].dt.day
        df["month"] = df["datetime_utc"].dt.month
        df["year"] = df["datetime_utc"].dt.year
        df["is_weekend"] = df["day_of_week"] >= 5

        df = df[
            [
                "datetime_utc",
                "price",
                "hour",
                "day_of_week",
                "day",
                "month",
                "year",
                "is_weekend",
            ]
        ]

        df.to_csv(
            output_file,
            index=False
        )


if __name__ == "__main__":
    preprocess_energy_data()