import os
from pathlib import Path

import great_expectations as gx


DATASOURCE_NAME = "energy_prices"
DATA_ASSET_NAME = "energy_prices_by_zone"
EXPECTATION_SUITE_NAME = "energy_prices_suite"


def validate_energy_data():
    project_root = Path(__file__).resolve().parents[2]
    gx_directory = project_root / "gx"

    os.chdir(gx_directory)

    context = gx.get_context(
        context_root_dir=gx_directory
    )

    data_source = context.get_datasource(
        DATASOURCE_NAME
    )

    data_asset = data_source.get_asset(
        DATA_ASSET_NAME
    )

    batch_request = data_asset.build_batch_request()

    batches = data_asset.get_batch_list_from_batch_request(
        batch_request
    )

    print(f"Number of batches: {len(batches)}")

    if not batches:
        print("ERROR: No batches found.")
        return False

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=EXPECTATION_SUITE_NAME,
    )

    result = validator.validate()

    context.build_data_docs()

    if result["success"]:
        print("Energy data validation successful.")
        return True

    print("Energy data validation failed.")
    return False


if __name__ == "__main__":
    success = validate_energy_data()

    if not success:
        raise SystemExit(1)