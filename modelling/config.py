from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASELINE_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
YOLO26_MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
SMOKE_TEST_MODELS = ["yolov8n.pt"]
FULL_EXPERIMENT_MODELS = BASELINE_MODELS + YOLO26_MODELS

VALID_MODES = {"smoke_test", "full_experiment"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ExperimentConfig:
    experiment_name: str
    model_name: str
    model_weight_or_config: str
    epochs: int
    imgsz: int
    batch: int
    learning_rate: float
    optimizer: str
    device: str
    seed: int = 42
    deterministic: bool = True
    patience: int = 15


@dataclass
class ModeConfig:
    mode: str
    project_root: Path
    modelling_root: Path
    compare_root: Path
    mode_root: Path
    train_runs_root: Path
    dataset_dir: Path
    dataset_yaml: Path
    absolute_yaml: Path
    experiments: list[ExperimentConfig]
    stage: str
    workers: int
    fraction: float | None
    training_summary_csv: Path
    evaluation_summary_csv: Path
    comparison_csv: Path
    plots_dir: Path
    predictions_dir: Path
    top_models_dir: Path
    report_md: Path
    # Legacy fields kept for backward compatibility
    models: list[str]
    epochs: int
    imgsz: int
    batch: int
    device: str


def detect_project_root(start: Path | None = None) -> Path:
    env_root = os.getenv("CAPSTONE_PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if env_path.exists():
            return env_path

    if start is None:
        start = Path(__file__).resolve().parent

    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "data_processed").exists():
            return candidate
    return start.parent


def detect_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_ultralytics_env(project_root: Path) -> Path:
    yolo_config_dir = project_root / ".ultralytics"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    return yolo_config_dir


def patch_ultralytics_threadpool_for_windows_sandbox() -> None:
    """
    Ultralytics label caching uses multiprocessing ThreadPool, which can fail
    with WinError 5 in restricted Windows sandboxes. This patch replaces that
    pool with a simple sequential implementation.
    """
    try:
        import ultralytics.data.dataset as yolo_dataset
    except Exception:
        return

    class _SequentialThreadPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_SequentialThreadPool":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def imap(self, func: Any, iterable: Any):
            for item in iterable:
                yield func(item)

    yolo_dataset.ThreadPool = _SequentialThreadPool


def safe_mode_name(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {valid}")
    return normalized


def get_mode_config(mode: str, stage: str = "stage1", extended: bool = False) -> ModeConfig:
    mode = safe_mode_name(mode)
    modelling_root = Path(__file__).resolve().parent
    project_root = detect_project_root(modelling_root)
    compare_root = project_root / "runs" / "compare_yolov8_vs_yolo26"
    mode_root = compare_root / mode
    train_runs_root = mode_root / "train_runs"
    top_models_dir = mode_root / "top_models"
    report_md = mode_root / "final_experiment_report.md"

    device = detect_device()
    dataset_dir = project_root / "data_processed"
    workers = 4
    fraction = None

    experiments: list[ExperimentConfig] = []

    if mode == "smoke_test":
        workers = 0
        fraction = 0.10
        # 2 lightweight smoke test runs using yolov8s only
        smoke_combos = [
            ("yolov8s_lr001_sgd", "yolov8s.pt", 1, 320, 4, 0.01, "SGD"),
            ("yolov8s_lr0001_adamw", "yolov8s.pt", 2, 320, 4, 0.001, "AdamW"),
        ]
        for exp_name, weight, ep, img, b, lr, opt in smoke_combos:
            experiments.append(
                ExperimentConfig(
                    experiment_name=exp_name,
                    model_name="yolov8s",
                    model_weight_or_config=weight,
                    epochs=ep,
                    imgsz=img,
                    batch=b,
                    learning_rate=lr,
                    optimizer=opt,
                    device=device,
                )
            )
    else:
        if stage == "stage1":
            base_models = [
                ("yolov8s", "yolov8s.pt"),
                ("yolov8m", "yolov8m.pt"),
                ("yolov8l", "yolov8l.pt"),
                ("yolov8x", "yolov8x.pt"),
                ("yolo26s", "yolo26s.pt"),
                ("yolo26m", "yolo26m.pt"),
                ("yolo26l", "yolo26l.pt"),
                ("yolo26x", "yolo26x.pt"),
            ]
            
            # Use only 2 hyperparameters with 2 values each:
            # Learning rate: 0.01, 0.001
            # Optimizer: SGD, AdamW
            learning_rates = [0.01, 0.001]
            optimizers = ["SGD", "AdamW"]
            
            for model_name, weight in base_models:
                for lr in learning_rates:
                    for opt in optimizers:
                        # Required experiment naming format: {model_name}_lr{learning_rate}_{optimizer}
                        # 0.01 -> lr001, 0.001 -> lr0001
                        if abs(lr - 0.01) < 1e-5:
                            lr_str = "001"
                        elif abs(lr - 0.001) < 1e-5:
                            lr_str = "0001"
                        else:
                            lr_str = str(lr).replace('.', '')
                        
                        exp_name = f"{model_name}_lr{lr_str}_{opt.lower()}"
                        experiments.append(
                            ExperimentConfig(
                                experiment_name=exp_name,
                                model_name=model_name,
                                model_weight_or_config=weight,
                                epochs=50,
                                imgsz=640,
                                batch=8,
                                learning_rate=lr,
                                optimizer=opt,
                                device=device,
                            )
                        )
        elif stage == "stage2":
            # Future-proof Stage 2 fine-tuning placeholder.
            base_models = [
                ("yolov8s", "yolov8s.pt"),
                ("yolo26s", "yolo26s.pt"),
            ]
            configs = [
                ("FT1", 30, 640, 8, 0.002, "AdamW"),
            ]
            for model_name, weight in base_models:
                for suffix, ep, img, b, lr, opt in configs:
                    if abs(lr - 0.002) < 1e-5:
                        lr_str = "0002"
                    else:
                        lr_str = str(lr).replace('.', '')
                    exp_name = f"{model_name}_stage2_lr{lr_str}_{opt.lower()}"
                    experiments.append(
                        ExperimentConfig(
                            experiment_name=exp_name,
                            model_name=model_name,
                            model_weight_or_config=weight,
                            epochs=ep,
                            imgsz=img,
                            batch=b,
                            learning_rate=lr,
                            optimizer=opt,
                            device=device,
                        )
                    )

    if device == "cpu":
        workers = 0
        for exp in experiments:
            exp.device = "cpu"
            # CPU safety: keep batch small
            exp.batch = min(exp.batch, 2)

    dataset_yaml = dataset_dir / "data.yaml"
    absolute_yaml = mode_root / "data_absolute.yaml"
    training_summary_csv = mode_root / "training_summary.csv"
    evaluation_summary_csv = mode_root / "evaluation_summary.csv"
    comparison_csv = mode_root / "comparison_summary.csv"
    plots_dir = mode_root / "comparison_plots"
    predictions_dir = mode_root / "sample_predictions"

    # Legacy fields fallback setup
    legacy_models = list({exp.model_weight_or_config for exp in experiments})
    legacy_epochs = experiments[0].epochs if experiments else 40
    legacy_imgsz = experiments[0].imgsz if experiments else 640
    legacy_batch = experiments[0].batch if experiments else 8

    return ModeConfig(
        mode=mode,
        project_root=project_root,
        modelling_root=modelling_root,
        compare_root=compare_root,
        mode_root=mode_root,
        train_runs_root=train_runs_root,
        dataset_dir=dataset_dir,
        dataset_yaml=dataset_yaml,
        absolute_yaml=absolute_yaml,
        experiments=experiments,
        stage=stage,
        workers=workers,
        fraction=fraction,
        training_summary_csv=training_summary_csv,
        evaluation_summary_csv=evaluation_summary_csv,
        comparison_csv=comparison_csv,
        plots_dir=plots_dir,
        predictions_dir=predictions_dir,
        top_models_dir=top_models_dir,
        report_md=report_md,
        models=legacy_models,
        epochs=legacy_epochs,
        imgsz=legacy_imgsz,
        batch=legacy_batch,
        device=device,
    )


def ensure_mode_directories(cfg: ModeConfig) -> None:
    cfg.compare_root.mkdir(parents=True, exist_ok=True)
    cfg.mode_root.mkdir(parents=True, exist_ok=True)
    cfg.train_runs_root.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)
    cfg.top_models_dir.mkdir(parents=True, exist_ok=True)



def read_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return data


def _normalize_names(raw_names: Any, nc: int | None = None) -> list[str]:
    if isinstance(raw_names, dict):
        ordered = [raw_names[key] for key in sorted(raw_names.keys(), key=lambda x: int(x))]
        return [str(name) for name in ordered]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, str):
        cleaned = raw_names.strip().strip("[]")
        if not cleaned:
            return []
        return [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
    if isinstance(nc, int) and nc > 0:
        return [str(i) for i in range(nc)]
    return []


def build_absolute_data_yaml(dataset_dir: Path, output_yaml: Path) -> Path:
    source_yaml = dataset_dir / "data.yaml"
    raw = read_yaml_file(source_yaml)

    nc_raw = raw.get("nc")
    nc = int(nc_raw) if nc_raw is not None else None
    names = _normalize_names(raw.get("names"), nc=nc)
    if nc is None and names:
        nc = len(names)
    if nc is None:
        raise ValueError(f"Could not infer class count from {source_yaml}")
    if not names:
        names = [str(i) for i in range(nc)]

    absolute = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(absolute, handle, sort_keys=False, allow_unicode=False)
    return output_yaml


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "" or cleaned.lower() == "none":
            return None
        value = cleaned
    try:
        return float(value)
    except Exception:
        pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        numeric = [to_float(item) for item in value]
        numeric = [item for item in numeric if item is not None]
        if not numeric:
            return None
        return float(sum(numeric) / len(numeric))
    if hasattr(value, "mean"):
        try:
            return float(value.mean())  # numpy arrays/tensors
        except Exception:
            return None
    return None


def resolve_model_source(project_root: Path, model_name: str) -> Path | str:
    local_candidate = project_root / "pretrained_weights" / model_name
    if local_candidate.exists():
        return local_candidate.resolve()

    local_candidate_alt = Path(model_name)
    if local_candidate_alt.exists():
        return local_candidate_alt.resolve()
    return model_name


def bytes_to_mb(size_bytes: int) -> float:
    return float(size_bytes) / (1024.0 * 1024.0)
