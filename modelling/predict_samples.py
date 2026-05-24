from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    IMAGE_EXTENSIONS,
    ensure_mode_directories,
    ensure_ultralytics_env,
    get_mode_config,
    read_csv,
    to_float,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample predictions using the best trained model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke_test",
        choices=["smoke_test", "full_experiment"],
        help="Which experiment mode to use.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of sample images to predict.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include heavy models in full experiment configuration.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1",
        choices=["stage1", "stage2"],
        help="Experiment stage.",
    )
    return parser.parse_args()


def pick_best_model(cfg) -> tuple[str, Path]:
    comparison_rows = read_csv(cfg.comparison_csv)
    if comparison_rows:
        for row in comparison_rows:
            best_pt = Path(row.get("best_pt", ""))
            if row.get("mode") == cfg.mode and row.get("status") == "success" and best_pt.exists():
                return row.get("model_name", "unknown"), best_pt

    evaluation_rows = read_csv(cfg.evaluation_summary_csv)
    evaluation_rows = [row for row in evaluation_rows if row.get("mode") == cfg.mode and row.get("status") == "success"]
    evaluation_rows.sort(key=lambda item: to_float(item.get("map50_95")) or -1.0, reverse=True)
    for row in evaluation_rows:
        best_pt = Path(row.get("best_pt", ""))
        if best_pt.exists():
            return row.get("model_name", "unknown"), best_pt

    training_rows = read_csv(cfg.training_summary_csv)
    for row in training_rows:
        best_pt = Path(row.get("best_pt", ""))
        if row.get("mode") == cfg.mode and row.get("status") == "success" and best_pt.exists():
            return row.get("model_name", "unknown"), best_pt

    raise FileNotFoundError("No usable best.pt found. Train models first.")


def collect_sample_images(cfg, num_images: int) -> list[Path]:
    candidate_dirs = [
        cfg.dataset_dir / "test" / "images",
        cfg.dataset_dir / "valid" / "images",
        cfg.dataset_dir / "train" / "images",
    ]
    images: list[Path] = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
            if len(images) >= num_images:
                return images
    return images


def predict_samples_for_mode(mode: str, num_images: int = 5, extended: bool = False, stage: str = "stage1") -> Path:
    cfg = get_mode_config(mode, stage=stage, extended=extended)
    ensure_mode_directories(cfg)
    ensure_ultralytics_env(cfg.project_root)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Ultralytics. Install with: pip install -U ultralytics"
        ) from exc

    model_name, best_pt = pick_best_model(cfg)
    sample_images = collect_sample_images(cfg, max(1, num_images))
    if not sample_images:
        raise FileNotFoundError("No images found for sample prediction.")

    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    model = YOLO(str(best_pt))
    results = model.predict(
        source=[str(path) for path in sample_images],
        imgsz=cfg.imgsz,
        device=cfg.device,
        conf=0.25,
        verbose=False,
        save=False,
    )

    for image_path, result in zip(sample_images, results):
        output_path = cfg.predictions_dir / f"pred_{image_path.name}"
        result.save(filename=str(output_path))
        manifest_rows.append(
            {
                "mode": cfg.mode,
                "model_name": model_name,
                "best_pt": str(best_pt),
                "source_image": str(image_path),
                "prediction_image": str(output_path),
                "status": "success",
                "error_message": "",
            }
        )

    manifest_csv = cfg.predictions_dir / "prediction_manifest.csv"
    write_csv(
        manifest_csv,
        manifest_rows,
        fieldnames=[
            "mode",
            "model_name",
            "best_pt",
            "source_image",
            "prediction_image",
            "status",
            "error_message",
        ],
    )

    print("Sample predictions complete.")
    print(f"Model used: {model_name}")
    print(f"best.pt: {best_pt}")
    print(f"Predictions folder: {cfg.predictions_dir}")
    print(f"Manifest CSV: {manifest_csv}")
    return manifest_csv


def main() -> None:
    args = parse_args()
    predict_samples_for_mode(
        mode=args.mode,
        num_images=args.num_images,
        extended=args.extended,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
