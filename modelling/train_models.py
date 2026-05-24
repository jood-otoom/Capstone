from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    ExperimentConfig,
    ModeConfig,
    build_absolute_data_yaml,
    bytes_to_mb,
    ensure_mode_directories,
    ensure_ultralytics_env,
    get_mode_config,
    patch_ultralytics_threadpool_for_windows_sandbox,
    read_csv,
    resolve_model_source,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO models for smoke test or full experiment.")
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke_test",
        choices=["smoke_test", "full_experiment"],
        help="Choose 'smoke_test' for YOLOv8n quick run or 'full_experiment' for all models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining of already successful experiments.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include heavy models in the full experiment grid.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1",
        choices=["stage1", "stage2"],
        help="Experiment stage.",
    )
    return parser.parse_args()


def short_model_name(model_name: str) -> str:
    return model_name.replace(".pt", "").replace(".yaml", "")


def get_artifact_path(path: Path) -> str:
    return str(path.resolve()) if path.exists() else ""


def locate_artifacts(run_dir: Path) -> dict[str, str]:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    results_png = run_dir / "results.png"
    confusion_matrix = run_dir / "confusion_matrix.png"
    labels_jpg = run_dir / "labels.jpg"

    return {
        "best_pt": get_artifact_path(best),
        "best_pt_size_mb": f"{bytes_to_mb(best.stat().st_size):.4f}" if best.exists() else "",
        "last_pt": get_artifact_path(last),
        "results_csv": get_artifact_path(results_csv),
        "results_png": get_artifact_path(results_png),
        "confusion_matrix_png": get_artifact_path(confusion_matrix),
        "labels_jpg": get_artifact_path(labels_jpg),
    }


def copy_latest_summary(cfg: ModeConfig) -> None:
    latest_target = cfg.compare_root / "training_summary.csv"
    if cfg.training_summary_csv.exists():
        shutil.copy2(cfg.training_summary_csv, latest_target)


def _train_with_source(
    YOLO: Any,
    cfg: ModeConfig,
    exp: ExperimentConfig,
    absolute_yaml: Path,
    pretrained: bool | None = None,
) -> Path:
    model_source = resolve_model_source(cfg.project_root, exp.model_weight_or_config)
    model = YOLO(str(model_source))
    train_kwargs: dict[str, Any] = {
        "data": str(absolute_yaml),
        "epochs": exp.epochs,
        "imgsz": exp.imgsz,
        "batch": exp.batch,
        "device": exp.device,
        "lr0": exp.learning_rate,
        "optimizer": exp.optimizer,
        "seed": exp.seed,
        "deterministic": exp.deterministic,
        "patience": exp.patience,
        "workers": cfg.workers,
        "project": str(cfg.train_runs_root),
        "name": exp.experiment_name,
        "exist_ok": True,
        "verbose": True,
        "plots": True,
    }
    if pretrained is not None:
        train_kwargs["pretrained"] = pretrained
    if cfg.fraction is not None:
        train_kwargs["fraction"] = cfg.fraction

    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", cfg.train_runs_root / exp.experiment_name))
    return save_dir


def train_models_for_mode(mode: str, force: bool = False, extended: bool = False, stage: str = "stage1") -> Path:
    cfg = get_mode_config(mode, stage=stage, extended=extended)
    ensure_mode_directories(cfg)
    ensure_ultralytics_env(cfg.project_root)
    absolute_yaml = build_absolute_data_yaml(cfg.dataset_dir, cfg.absolute_yaml)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Ultralytics. Install with: pip install -U ultralytics"
        ) from exc
    patch_ultralytics_threadpool_for_windows_sandbox()

    import torch

    # Load existing training summary to support resume capabilities
    existing_runs = {}
    if cfg.training_summary_csv.exists():
        try:
            for row in read_csv(cfg.training_summary_csv):
                if row.get("status") == "success" and row.get("experiment_name"):
                    existing_runs[row["experiment_name"]] = row
        except Exception as e:
            print(f"Warning: Failed to read existing training summary: {e}")

    rows: list[dict[str, str]] = []
    fieldnames = [
        "mode",
        "stage",
        "experiment_name",
        "model_name",
        "model_requested",
        "model_source_used",
        "run_dir",
        "best_pt",
        "last_pt",
        "results_csv",
        "results_png",
        "confusion_matrix_png",
        "labels_jpg",
        "best_pt_size_mb",
        "planned_epochs",
        "actual_epochs_completed",
        "imgsz",
        "batch",
        "learning_rate",
        "optimizer",
        "fraction",
        "device",
        "gpu_memory_reserved_mb",
        "gpu_memory_allocated_mb",
        "peak_gpu_memory_mb",
        "training_start_time",
        "training_end_time",
        "training_time_seconds",
        "training_time_minutes",
        "time_per_epoch_seconds",
        "status",
        "error_message",
        "notes",
    ]

    print(f"Starting training mode: {cfg.mode} (Stage: {cfg.stage})")
    print(f"Project root: {cfg.project_root}")
    print(f"Dataset: {cfg.dataset_dir}")
    print(f"Absolute YAML: {absolute_yaml}")
    print(f"Device selected: {cfg.device}")
    print(f"Total experiments: {len(cfg.experiments)}")
    print("-" * 80)

    for index, exp in enumerate(cfg.experiments, start=1):
        experiment_name = exp.experiment_name
        model_name = exp.model_name
        model_weight_or_config = exp.model_weight_or_config

        # Resume logic: skip if successfully completed
        if experiment_name in existing_runs and not force:
            print(f"[{index}/{len(cfg.experiments)}] Skipping completed experiment: {experiment_name}")
            rows.append(existing_runs[experiment_name])
            # Save incrementally
            write_csv(cfg.training_summary_csv, rows, fieldnames)
            copy_latest_summary(cfg)
            continue

        expected_run_dir = cfg.train_runs_root / experiment_name
        model_source = resolve_model_source(cfg.project_root, model_weight_or_config)
        training_start_time = datetime.now().isoformat(timespec="seconds")
        start_time = time.time()

        status = "failed"
        error_message = ""
        notes = ""
        actual_run_dir = expected_run_dir
        used_source = str(model_source)

        gpu_memory_reserved_mb = "0.00"
        gpu_memory_allocated_mb = "0.00"
        peak_gpu_memory_mb = "0.00"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"[{index}/{len(cfg.experiments)}] Training: {experiment_name} from: {model_source}")
        try:
            actual_run_dir = _train_with_source(
                YOLO=YOLO,
                cfg=cfg,
                exp=exp,
                absolute_yaml=absolute_yaml,
            )
            status = "success"
        except Exception as first_exc:
            # Smoke-test fallback: if yolov8s.pt fails, try scratch .yaml
            if cfg.mode == "smoke_test" and model_weight_or_config == "yolov8s.pt":
                try:
                    fallback_exp = ExperimentConfig(
                        experiment_name=f"yolov8s_yaml_fallback",
                        model_name="yolov8s",
                        model_weight_or_config="yolov8s.yaml",
                        epochs=exp.epochs,
                        imgsz=exp.imgsz,
                        batch=exp.batch,
                        learning_rate=exp.learning_rate,
                        optimizer=exp.optimizer,
                        device=exp.device,
                    )
                    actual_run_dir = _train_with_source(
                        YOLO=YOLO,
                        cfg=cfg,
                        exp=fallback_exp,
                        absolute_yaml=absolute_yaml,
                        pretrained=False,
                    )
                    used_source = "yolov8s.yaml"
                    status = "success"
                    notes = "Fallback to yolov8s.yaml scratch succeeded."
                    print("Smoke-test fallback succeeded.")
                except Exception as fallback_exc:
                    error_message = f"Primary error: {first_exc} | Fallback error: {fallback_exc}"
            else:
                error_message = str(first_exc)
                if model_weight_or_config.startswith("yolo26"):
                    error_message += " | YOLO26 model may require a newer compatible Ultralytics version."

        training_end_time = datetime.now().isoformat(timespec="seconds")
        training_time_seconds_val = time.time() - start_time
        training_time_seconds = f"{training_time_seconds_val:.2f}"
        training_time_minutes = f"{training_time_seconds_val / 60.0:.4f}"
        artifacts = locate_artifacts(actual_run_dir)

        # Retrieve actual epochs completed from results.csv
        actual_epochs_completed = exp.epochs
        if status == "success":
            results_csv_path = Path(artifacts["results_csv"])
            if results_csv_path.exists():
                try:
                    with results_csv_path.open("r", encoding="utf-8") as f:
                        # Subtract header
                        actual_epochs_completed = sum(1 for _ in f) - 1
                except Exception as e:
                    print(f"Warning: Failed to read actual epochs completed: {e}")

        # Compute average time per epoch
        if actual_epochs_completed > 0:
            time_per_epoch_seconds = f"{training_time_seconds_val / actual_epochs_completed:.2f}"
        else:
            time_per_epoch_seconds = "0.00"

        # Retrieve GPU memory telemetry
        if torch.cuda.is_available():
            gpu_memory_reserved_mb = f"{torch.cuda.memory_reserved() / (1024.0 * 1024.0):.2f}"
            gpu_memory_allocated_mb = f"{torch.cuda.memory_allocated() / (1024.0 * 1024.0):.2f}"
            peak_gpu_memory_mb = f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.2f}"

        row = {
            "mode": cfg.mode,
            "stage": cfg.stage,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "model_requested": model_weight_or_config,
            "model_source_used": used_source,
            "run_dir": str(actual_run_dir.resolve()) if actual_run_dir.exists() else str(actual_run_dir),
            "best_pt": artifacts["best_pt"],
            "last_pt": artifacts["last_pt"],
            "results_csv": artifacts["results_csv"],
            "results_png": artifacts["results_png"],
            "confusion_matrix_png": artifacts["confusion_matrix_png"],
            "labels_jpg": artifacts["labels_jpg"],
            "best_pt_size_mb": artifacts["best_pt_size_mb"],
            "planned_epochs": str(exp.epochs),
            "actual_epochs_completed": str(actual_epochs_completed),
            "imgsz": str(exp.imgsz),
            "batch": str(exp.batch),
            "learning_rate": str(exp.learning_rate),
            "optimizer": exp.optimizer,
            "fraction": "" if cfg.fraction is None else str(cfg.fraction),
            "device": exp.device,
            "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
            "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "training_start_time": training_start_time,
            "training_end_time": training_end_time,
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_minutes,
            "time_per_epoch_seconds": time_per_epoch_seconds,
            "status": status,
            "error_message": error_message,
            "notes": notes,
        }
        rows.append(row)

        # Incremental CSV write
        write_csv(cfg.training_summary_csv, rows, fieldnames)
        copy_latest_summary(cfg)

        if status == "success":
            print(f"Completed: {experiment_name} (Epochs: {actual_epochs_completed}/{exp.epochs}) in {training_time_seconds}s")
        else:
            print(f"Failed: {experiment_name}")
            print(f"Error: {error_message}")
        print("-" * 80)

    successes = sum(1 for row in rows if row["status"] == "success")
    failures = len(rows) - successes
    print("Training run finished.")
    print(f"Summary CSV: {cfg.training_summary_csv}")
    print(f"Successful experiments: {successes}")
    print(f"Failed experiments: {failures}")
    return cfg.training_summary_csv


def main() -> None:
    args = parse_args()
    train_models_for_mode(
        mode=args.mode,
        force=args.force,
        extended=args.extended,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()

