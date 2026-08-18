from pathlib import Path
import sys

import great_expectations as gx


def main():
    project_root = Path(__file__).resolve().parents[2]
    gx_root = project_root / "gx"
    data_root = project_root / "data"

    context = gx.get_context(
        context_root_dir=str(gx_root)
    )

    datasource_name = "energy_prices"
    data_asset_name = "energy_prices_by_zone"
    expectation_suite_name = "energy_prices_suite"

    datasource = context.sources.add_or_update_pandas_filesystem(
        name=datasource_name,
        base_directory=str(data_root),
    )

    try:
        data_asset = datasource.get_asset(data_asset_name)
    except Exception:
        data_asset = datasource.add_csv_asset(
            name=data_asset_name,
            batching_regex=r"processed/electricity/prices/(?P<zone>[^/]+)\.csv",
        )

    batch_request = data_asset.build_batch_request()

    batches = datasource.get_batch_list_from_batch_request(
        batch_request
    )

    print(f"Data directory: {data_root}")
    print(f"Number of batches: {len(batches)}")

    if not batches:
        print("ERROR: No batches found.")
        print("Available processed files:")

        processed_directory = (
            data_root / "processed" / "electricity" / "prices"
        )

        if processed_directory.exists():
            for file in sorted(processed_directory.glob("*.csv")):
                print(file)
        else:
            print(
                f"Directory does not exist: {processed_directory}"
            )

        return False

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name,
    )

    result = validator.validate()

    context.build_data_docs()

    if result["success"]:
        print("Validation passed for energy prices!")
        return True

    print("Validation failed for energy prices!")
    return False


if __name__ == "__main__":
    success = main()

    if not success:
        sys.exit(1)