from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from obsyd import Obsyd, ObsydNoData


START_DATE = "2024-01-01"


def fetch_energy_data():
    client = Obsyd()

    zones = client.zones()
    enabled_zones = zones["enabled_keys"]

    output_directory = Path("data/raw/electricity/prices")
    output_directory.mkdir(parents=True, exist_ok=True)

    end_date = (
        datetime.now() + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    for zone in enabled_zones:
        output_file = output_directory / f"{zone}.csv"

        if output_file.exists():
            existing_df = pd.read_csv(
                output_file,
                parse_dates=["datetime_utc"]
            )

            if existing_df.empty:
                start_date = START_DATE
            else:
                last_timestamp = existing_df["datetime_utc"].max()
                next_timestamp = (
                    last_timestamp + pd.Timedelta(hours=1)
                )

                start_date = next_timestamp.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )

                end_timestamp = pd.Timestamp(
                    end_date,
                    tz="UTC"
                )

                if next_timestamp >= end_timestamp:
                    continue
        else:
            existing_df = pd.DataFrame()
            start_date = START_DATE

        try:
            df = client.series(
                "price.dayahead",
                zone,
                start=start_date,
                end=end_date,
            )
        except ObsydNoData:
            continue

        if df.empty:
            continue

        if not existing_df.empty:
            df = pd.concat(
                [existing_df, df],
                ignore_index=True
            )

        df = df.drop_duplicates(
            subset=["datetime_utc"]
        )

        df = df.sort_values(
            "datetime_utc"
        )

        df.to_csv(
            output_file,
            index=False
        )


if __name__ == "__main__":
    fetch_energy_data()