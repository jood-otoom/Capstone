from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    build_absolute_data_yaml,
    bytes_to_mb,
    ensure_mode_directories,
    ensure_ultralytics_env,
    get_mode_config,
    patch_ultralytics_threadpool_for_windows_sandbox,
    read_csv,
    to_float,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO models on the hold-out test split only.")
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke_test",
        choices=["smoke_test", "full_experiment"],
        help="Which experiment mode to evaluate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of already successful evaluations.",
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


def metric_from_attr(obj: Any, attr_names: list[str]) -> float | None:
    for attr in attr_names:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            numeric = to_float(value)
            if numeric is not None:
                return numeric
    return None


def parse_metrics(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    results_dict = getattr(metrics, "results_dict", {}) or {}

    precision = metric_from_attr(box, ["mp", "p"]) if box is not None else None
    recall = metric_from_attr(box, ["mr", "r"]) if box is not None else None
    map50 = metric_from_attr(box, ["map50"]) if box is not None else None
    map50_95 = metric_from_attr(box, ["map"]) if box is not None else None

    if precision is None:
        precision = to_float(results_dict.get("metrics/precision(B)"))
    if recall is None:
        recall = to_float(results_dict.get("metrics/recall(B)"))
    if map50 is None:
        map50 = to_float(results_dict.get("metrics/mAP50(B)"))
    if map50_95 is None:
        map50_95 = to_float(results_dict.get("metrics/mAP50-95(B)"))

    fitness = to_float(getattr(metrics, "fitness", None))
    if fitness is None:
        fitness = to_float(results_dict.get("fitness"))

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "fitness": fitness,
        "inference_ms": to_float(speed.get("inference")),
        "preprocess_ms": to_float(speed.get("preprocess")),
        "postprocess_ms": to_float(speed.get("postprocess")),
    }


def load_training_rows(training_csv: Path, mode: str, fallback_csv: Path) -> list[dict[str, str]]:
    rows = read_csv(training_csv)
    if rows:
        return [row for row in rows if row.get("mode") == mode]
    fallback_rows = read_csv(fallback_csv)
    return [row for row in fallback_rows if row.get("mode") == mode]


def evaluate_mode(mode: str, force: bool = False, extended: bool = False, stage: str = "stage1") -> Path:
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

    training_rows = load_training_rows(
        cfg.training_summary_csv,
        cfg.mode,
        cfg.compare_root / "training_summary.csv",
    )
    # Ensure we filter by stage if present in training rows
    if training_rows and "stage" in training_rows[0]:
        training_rows = [row for row in training_rows if row.get("stage") == stage]

    if not training_rows:
        raise FileNotFoundError(
            f"No training summary rows found for mode '{cfg.mode}' stage '{stage}'. Run train_models.py first."
        )

    # Load existing evaluation summary to support resume capabilities
    existing_evals = {}
    if cfg.evaluation_summary_csv.exists():
        try:
            for row in read_csv(cfg.evaluation_summary_csv):
                if row.get("status") == "success" and row.get("experiment_name"):
                    existing_evals[row["experiment_name"]] = row
        except Exception as e:
            print(f"Warning: Failed to read existing evaluation summary: {e}")

    eval_rows: list[dict[str, str]] = []
    successful = [row for row in training_rows if row.get("status") == "success" and row.get("best_pt")]

    fieldnames = [
        "mode",
        "stage",
        "experiment_name",
        "model_name",
        "best_pt",
        "run_dir",
        "status",
        "error_message",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "fitness",
        "preprocess_ms_per_image",
        "inference_ms_per_image",
        "postprocess_ms_per_image",
        "total_latency_ms_per_image",
        "fps",
        "param_count",
        "best_pt_size_mb",
        "gpu_memory_reserved_mb",
        "gpu_memory_allocated_mb",
        "peak_gpu_memory_mb",
        "deployment_suitability",
        "planned_epochs",
        "actual_epochs_completed",
        "imgsz",
        "batch",
        "learning_rate",
        "optimizer",
        "device",
        "evaluation_start_time",
        "evaluation_end_time",
        "evaluation_time_seconds",
        "evaluation_time_minutes",
    ]

    print(f"Starting test split evaluation mode: {cfg.mode} (Stage: {cfg.stage})")
    print(f"Test split YAML: {absolute_yaml}")
    print(f"Models queued for test split evaluation: {len(successful)}")
    print("-" * 80)

    for index, row in enumerate(successful, start=1):
        experiment_name = row.get("experiment_name", row.get("model_name", ""))
        model_name = row.get("model_name", "")
        best_pt = Path(row["best_pt"])
        status = "failed"
        error_message = ""
        evaluation_start_time = datetime.now().isoformat(timespec="seconds")
        start_time = time.time()

        # Resume logic: skip if successfully completed
        if experiment_name in existing_evals and not force:
            print(f"[{index}/{len(successful)}] Skipping completed test split evaluation for: {experiment_name}")
            eval_rows.append(existing_evals[experiment_name])
            write_csv(cfg.evaluation_summary_csv, eval_rows, fieldnames)
            shutil.copy2(cfg.evaluation_summary_csv, cfg.compare_root / "evaluation_summary.csv")
            continue

        metrics_data = {
            "precision": "",
            "recall": "",
            "map50": "",
            "map50_95": "",
            "fitness": "",
            "preprocess_ms_per_image": "",
            "inference_ms_per_image": "",
            "postprocess_ms_per_image": "",
            "total_latency_ms_per_image": "",
            "fps": "",
            "param_count": "",
            "best_pt_size_mb": "",
        }

        print(f"[{index}/{len(successful)}] Evaluating {experiment_name} on hold-out test split using {best_pt}")
        deployment_suitability = "Cloud Only"
        try:
            if not best_pt.exists():
                raise FileNotFoundError(f"best.pt not found: {best_pt}")

            model = YOLO(str(best_pt))
            metrics = model.val(
                data=str(absolute_yaml),
                split="test",
                imgsz=int(row.get("imgsz", cfg.imgsz)),
                batch=int(row.get("batch", cfg.batch)),
                device=row.get("device", cfg.device),
                workers=cfg.workers,
                verbose=False,
                plots=False,
            )
            parsed = parse_metrics(metrics)

            param_count = ""
            try:
                param_count = str(int(sum(param.numel() for param in model.model.parameters())))
            except Exception:
                param_count = ""

            best_pt_size_mb = ""
            best_pt_size_mb_val = 0.0
            if best_pt.exists():
                best_pt_size_mb_val = bytes_to_mb(best_pt.stat().st_size)
                best_pt_size_mb = f"{best_pt_size_mb_val:.4f}"

            # Calculate components per image
            prep_ms = parsed["preprocess_ms"] or 0.0
            inf_ms = parsed["inference_ms"] or 0.0
            post_ms = parsed["postprocess_ms"] or 0.0
            tot_lat = prep_ms + inf_ms + post_ms
            fps_val = (1000.0 / tot_lat) if tot_lat > 0 else 0.0

            # Deployment classification
            peak_gpu = to_float(row.get("peak_gpu_memory_mb")) or 0.0
            if tot_lat <= 30.0 and fps_val >= 30.0 and peak_gpu <= 1000.0 and best_pt_size_mb_val <= 30.0:
                deployment_suitability = "Edge Device Suitable"
            elif tot_lat <= 100.0 and fps_val >= 10.0 and peak_gpu <= 4000.0 and best_pt_size_mb_val <= 100.0:
                deployment_suitability = "Mid-Range GPU Suitable"
            else:
                deployment_suitability = "Cloud Only"

            metrics_data = {
                "precision": "" if parsed["precision"] is None else f"{parsed['precision']:.6f}",
                "recall": "" if parsed["recall"] is None else f"{parsed['recall']:.6f}",
                "map50": "" if parsed["map50"] is None else f"{parsed['map50']:.6f}",
                "map50_95": "" if parsed["map50_95"] is None else f"{parsed['map50_95']:.6f}",
                "fitness": "" if parsed["fitness"] is None else f"{parsed['fitness']:.6f}",
                "preprocess_ms_per_image": f"{prep_ms:.4f}",
                "inference_ms_per_image": f"{inf_ms:.4f}",
                "postprocess_ms_per_image": f"{post_ms:.4f}",
                "total_latency_ms_per_image": f"{tot_lat:.4f}",
                "fps": f"{fps_val:.4f}",
                "param_count": param_count,
                "best_pt_size_mb": best_pt_size_mb,
            }
            status = "success"

            # Confusion matrix copy
            confusion_matrices_dir = cfg.mode_root / "confusion_matrices"
            confusion_matrices_dir.mkdir(parents=True, exist_ok=True)
            train_confusion_matrix = Path(row.get("confusion_matrix_png", ""))
            if train_confusion_matrix.exists():
                try:
                    dest = confusion_matrices_dir / f"{experiment_name}_confusion_matrix.png"
                    shutil.copy2(train_confusion_matrix, dest)
                except Exception as e:
                    print(f"Warning: Failed to copy confusion matrix: {e}")

            print(f"Completed test split evaluation for {experiment_name}")
        except Exception as exc:
            error_message = str(exc)
            print(f"Test split evaluation failed for {experiment_name}: {error_message}")

        evaluation_end_time = datetime.now().isoformat(timespec="seconds")
        evaluation_time_seconds_val = time.time() - start_time
        evaluation_time_seconds = f"{evaluation_time_seconds_val:.2f}"
        evaluation_time_minutes = f"{evaluation_time_seconds_val / 60.0:.4f}"

        eval_rows.append(
            {
                "mode": cfg.mode,
                "stage": cfg.stage,
                "experiment_name": experiment_name,
                "model_name": model_name,
                "best_pt": str(best_pt),
                "run_dir": row.get("run_dir", ""),
                "status": status,
                "error_message": error_message,
                "precision": metrics_data["precision"],
                "recall": metrics_data["recall"],
                "map50": metrics_data["map50"],
                "map50_95": metrics_data["map50_95"],
                "fitness": metrics_data["fitness"],
                "preprocess_ms_per_image": metrics_data["preprocess_ms_per_image"],
                "inference_ms_per_image": metrics_data["inference_ms_per_image"],
                "postprocess_ms_per_image": metrics_data["postprocess_ms_per_image"],
                "total_latency_ms_per_image": metrics_data["total_latency_ms_per_image"],
                "fps": metrics_data["fps"],
                "param_count": metrics_data["param_count"],
                "best_pt_size_mb": metrics_data["best_pt_size_mb"],
                "gpu_memory_reserved_mb": row.get("gpu_memory_reserved_mb", "0.00"),
                "gpu_memory_allocated_mb": row.get("gpu_memory_allocated_mb", "0.00"),
                "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb", "0.00"),
                "deployment_suitability": deployment_suitability,
                "planned_epochs": row.get("planned_epochs", ""),
                "actual_epochs_completed": row.get("actual_epochs_completed", ""),
                "imgsz": row.get("imgsz", ""),
                "batch": row.get("batch", ""),
                "learning_rate": row.get("learning_rate", ""),
                "optimizer": row.get("optimizer", ""),
                "device": row.get("device", ""),
                "evaluation_start_time": evaluation_start_time,
                "evaluation_end_time": evaluation_end_time,
                "evaluation_time_seconds": evaluation_time_seconds,
                "evaluation_time_minutes": evaluation_time_minutes,
            }
        )

        # Incremental CSV write
        write_csv(cfg.evaluation_summary_csv, eval_rows, fieldnames)
        shutil.copy2(cfg.evaluation_summary_csv, cfg.compare_root / "evaluation_summary.csv")

    successful_count = sum(1 for row in eval_rows if row["status"] == "success")
    failed_count = len(eval_rows) - successful_count
    print("Evaluation finished.")
    print(f"Evaluation summary: {cfg.evaluation_summary_csv}")
    print(f"Successful evaluations: {successful_count}")
    print(f"Failed evaluations: {failed_count}")
    return cfg.evaluation_summary_csv


def main() -> None:
    args = parse_args()
    evaluate_mode(
        mode=args.mode,
        force=args.force,
        extended=args.extended,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()

