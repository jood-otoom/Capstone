from __future__ import annotations

import argparse

from check_dataset import run_dataset_check
from compare_results import compare_mode
from evaluate_models import evaluate_mode
from predict_samples import predict_samples_for_mode
from train_models import train_models_for_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end smoke test (YOLOv8n only).")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip sample prediction step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "smoke_test"

    print("Running dataset check...")
    run_dataset_check()
    print("Running smoke-test training...")
    train_models_for_mode(mode)
    print("Running smoke-test evaluation...")
    evaluate_mode(mode)
    print("Running smoke-test comparison...")
    compare_mode(mode)

    if not args.skip_predict:
        print("Running sample predictions...")
        predict_samples_for_mode(mode, num_images=5)

    print("Smoke test pipeline finished.")


if __name__ == "__main__":
    main()
