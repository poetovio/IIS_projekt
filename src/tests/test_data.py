import os
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset


def test_energy_data(current_path, reference_path, report_path):
    current = pd.read_csv(current_path)

    if not reference_path.exists():
        print(
            f"Reference file not found for {current_path.name}. "
            f"Creating reference data."
        )

        reference_path.parent.mkdir(parents=True, exist_ok=True)
        current.to_csv(reference_path, index=False)

    reference = pd.read_csv(reference_path)

    columns_to_remove = ["datetime_utc"]

    for column in columns_to_remove:
        if column in reference.columns:
            del reference[column]

        if column in current.columns:
            del current[column]

    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(),
        ],
        include_tests=True,
    )

    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.save_html(str(report_path))

    if not report_path.exists():
        print(
            f"ERROR: Report was not created for "
            f"{current_path.name}: {report_path}"
        )
        return False

    print(f"Report created: {report_path}")

    all_tests_passed = True
    result_dict = result.dict()

    if "tests" in result_dict:
        for test in result_dict["tests"]:
            if (
                "status" in test
                and test["status"] != "SUCCESS"
            ):
                all_tests_passed = False
                break

    if not all_tests_passed:
        print(f"Data tests failed for {current_path.name}.")
        return False

    print(f"Data tests passed for {current_path.name}.")

    current = pd.read_csv(current_path)
    current.to_csv(reference_path, index=False)

    return True


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_directory = (
        project_root
        / "data"
        / "processed"
        / "electricity"
        / "prices"
    )

    reference_directory = (
        project_root
        / "data"
        / "reference"
        / "electricity"
        / "prices"
    )

    report_directory = (
        project_root
        / "reports"
        / "data_testing"
        / "electricity"
    )

    files = sorted(data_directory.glob("*.csv"))

    print(f"Found {len(files)} energy price files.")

    if not files:
        print("ERROR: No energy price files found.")
        return 1

    all_tests_passed = True

    for current_path in files:
        print()
        print(f"Testing {current_path.name}...")

        reference_path = (
            reference_directory / current_path.name
        )

        report_path = (
            report_directory
            / f"{current_path.stem}.html"
        )

        success = test_energy_data(
            current_path=current_path,
            reference_path=reference_path,
            report_path=report_path,
        )

        if not success:
            all_tests_passed = False

    print()

    if all_tests_passed:
        print("All energy data tests passed.")
        return 0

    print("One or more energy data tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())