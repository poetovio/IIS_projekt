from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from obsyd import Obsyd

START_DATE = "2024-01-01"


def fetch_zone(client, zone, output_directory, end_date):
    output_file = output_directory / f"{zone}.csv"

    if output_file.exists():
        existing_df = pd.read_csv(
            output_file,
            parse_dates=["datetime_utc"]
        )

        existing_df = existing_df.reset_index(drop=True)

        if existing_df.empty:
            start_date = START_DATE
        else:
            last_date = existing_df["datetime_utc"].max()
            start_date = (
                last_date + timedelta(hours=1)
            ).strftime("%Y-%m-%d")

        if start_date >= end_date:
            return

        try:
            new_df = client.series(
                "price.dayahead",
                zone,
                start=start_date,
                end=end_date,
            )
        except Exception:
            return

        if new_df.empty:
            return

        new_df = new_df.reset_index()

        df = pd.concat(
            [existing_df, new_df],
            ignore_index=True
        )

    else:
        try:
            df = client.series(
                "price.dayahead",
                zone,
                start=START_DATE,
                end=end_date,
            )
        except Exception:
            return

        if df.empty:
            return

        df = df.reset_index()

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        utc=True
    )

    df = df.drop_duplicates(
        subset=["datetime_utc"],
        keep="last"
    )

    df = df.sort_values("datetime_utc")

    df.to_csv(
        output_file,
        index=False
    )


def fetch_energy_data():
    client = Obsyd()

    zones = client.zones()
    enabled_zones = zones["enabled_keys"]

    end_date = (
        datetime.now() + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    output_directory = Path(
        "data/raw/electricity/prices"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for zone in enabled_zones:
        fetch_zone(
            client,
            zone,
            output_directory,
            end_date
        )


if __name__ == "__main__":
    fetch_energy_data()