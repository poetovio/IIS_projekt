from pathlib import Path

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
    checkpoint_name = "energy_prices_checkpoint"

    datasource = context.sources.add_or_update_pandas_filesystem(
        name=datasource_name,
        base_directory=str(data_root),
    )

    try:
        data_asset = datasource.get_asset(data_asset_name)
    except Exception:
        data_asset = datasource.add_csv_asset(
            name=data_asset_name,
            batching_regex=r"processed/electricity/prices/.*\.csv",
        )

    context.add_or_update_expectation_suite(
        expectation_suite_name=expectation_suite_name
    )

    batch_request = data_asset.build_batch_request()

    batches = datasource.get_batch_list_from_batch_request(
        batch_request
    )

    if not batches:
        print("No batches found for validation.")
        raise SystemExit(1)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name,
    )

    validator.expect_table_columns_to_match_set(
        column_set=[
            "datetime_utc",
            "price",
            "hour",
            "day_of_week",
            "day",
            "month",
            "year",
            "is_weekend",
        ]
    )

    validator.expect_column_values_to_not_be_null(
        "datetime_utc"
    )

    validator.expect_column_values_to_be_unique(
        "datetime_utc"
    )

    validator.expect_column_values_to_not_be_null(
        "price"
    )

    validator.expect_column_values_to_be_between(
        "price",
        min_value=-500,
        max_value=5000,
        mostly=1.0,
    )

    validator.expect_column_values_to_be_between(
        "hour",
        min_value=0,
        max_value=23,
    )

    validator.expect_column_values_to_be_between(
        "day_of_week",
        min_value=0,
        max_value=6,
    )

    validator.expect_column_values_to_be_between(
        "day",
        min_value=1,
        max_value=31,
    )

    validator.expect_column_values_to_be_between(
        "month",
        min_value=1,
        max_value=12,
    )

    validator.expect_column_values_to_be_between(
        "year",
        min_value=2024,
        max_value=2100,
    )

    validator.save_expectation_suite(
        discard_failed_expectations=False
    )

    context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name,
            }
        ],
    )

    checkpoint = context.get_checkpoint(
        checkpoint_name
    )

    checkpoint_result = checkpoint.run(
        run_id="energy_prices_run"
    )

    context.build_data_docs()

    if checkpoint_result["success"]:
        print("Energy data validation passed!")
        raise SystemExit(0)

    print("Energy data validation failed!")
    raise SystemExit(1)


if __name__ == "__main__":
    main()