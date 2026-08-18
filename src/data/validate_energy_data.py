from pathlib import Path
import os

import great_expectations as gx


def main():
    project_root = Path(__file__).resolve().parents[2]
    gx_root = project_root / "gx"

    os.chdir(gx_root)

    context = gx.get_context(
        context_root_dir=str(gx_root)
    )

    checkpoint = context.get_checkpoint(
        "energy_prices_checkpoint"
    )

    checkpoint_result = checkpoint.run(
        run_id="energy_prices_run"
    )

    context.build_data_docs()

    if checkpoint_result["success"]:
        print("Validation passed for energy prices!")
        return True

    print("Validation failed for energy prices!")
    return False


if __name__ == "__main__":
    success = main()

    if not success:
        raise SystemExit(1)