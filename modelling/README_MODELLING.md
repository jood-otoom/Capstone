# YOLO Modelling Pipeline (YOLOv8 vs YOLO26)

This directory contains a complete, robust, and research-grade object-detection workflow designed for comparing the **YOLOv8** and **YOLO26** model families. It supports scientific reproducibility, incremental crash recovery, early stopping, and automated final report generation.

---

## 1. Pipeline Highlights

- **Scientific Reproducibility**: Mandatory `seed=42` and `deterministic=True` applied across all training sessions.
- **Ultralytics Early Stopping**: Automatically terminates training if hold-out test split metrics do not improve for `patience=15` epochs, recording both planned and actual completed epochs.
- **Controlled Comparison Grid**:
  - **Stage 1**: Controlled search across 32 experiment combinations (8 core models × 2 learning rates × 2 optimizers).
- **Fail-Safe Resume Support**: Keeps track of successful experiments/evaluations via `training_summary.csv` and `evaluation_summary.csv`. If training is interrupted, re-running the command skips successful runs automatically unless `--force` is passed.
- **Incremental Progress Saving**: Writes summaries to disk after every training, evaluation, and prediction step to prevent data loss from system crashes.
- **Balanced Score Normalization**: Prioritizes **$70\%$ mAP@50-95, $20\%$ hold-out test split latency, and $10\%$ weight file size** to determine the optimal candidates.
- **Deployment Suitability Classification**: Categorizes models into:
  - *Edge Device Suitable* (Latency $\le 30$ms, FPS $\ge 30$, peak VRAM $\le 1000$MB, size $\le 30$MB)
  - *Mid-Range GPU Suitable* (Latency $\le 100$ms, FPS $\ge 10$, peak VRAM $\le 4000$MB, size $\le 100$MB)
  - *Cloud Only* (Fails to meet edge/mid-range thresholds)
- **Top Models Directory**: Automatically extracts and copies the best weight files for the highest accuracy, fastest inference, and best balanced score models into the `top_models/` folder.
- **Automated Research Report Generator**: Automatically generates a presentation-ready `final_experiment_report.md` detailing pareto-optimal trade-offs, deployment suitability, and VRAM telemetry.

---

## 2. Model Grids & Configurations

### Core Grid (32 Runs)
Compares 8 core model configurations across two hyperparameter settings:
- **Models**: `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x` (nano models `yolov8n` and `yolo26n` are excluded).
- **Hyperparameters**:
  - **Learning rate**: `0.01` and `0.001`
  - **Optimizer**: `SGD` and `AdamW`
- **Fixed Parameters (for Fairness)**:
  - `epochs = 50`
  - `imgsz = 640`
  - `batch = 8` (CPU environment falls back to `2` automatically)
  - `patience = 15`
  - `seed = 42`
  - `deterministic = True`

Required naming format: `{model_name}_lr{learning_rate}_{optimizer}` (e.g. `yolov8s_lr001_sgd`, `yolo26x_lr0001_adamw`).

### Lightweight Smoke Test (2 Runs)
To verify end-to-end code changes quickly and safely on CPU using `yolov8s` only:
- **Run 1 (`yolov8s_lr001_sgd`)**: 1 epoch, imgsz=320, batch=4, SGD, lr=0.01
- **Run 2 (`yolov8s_lr0001_adamw`)**: 2 epochs, imgsz=320, batch=4, AdamW, lr=0.001

---

## 3. CLI Command Usage

All commands should be executed from the `C:\Capstone` directory:

### Run Dataset Health Check
Validate pathing, class annotations, image counts, and generate health reports:
```powershell
python modelling/check_dataset.py
```

### Run End-to-End Smoke Test
Run dataset checking, training, evaluation, comparison, and prediction sample generation sequentially:
```powershell
python modelling/run_smoke_test.py
```

### Run Training Pipeline
```powershell
# Run smoke test training (2 quick runs)
python modelling/train_models.py --mode smoke_test

# Run full Stage 1 training (32 combinations)
python modelling/train_models.py --mode full_experiment --stage stage1

# Force retrain all configurations even if previously completed successfully
python modelling/train_models.py --mode full_experiment --stage stage1 --force
```

### Run Evaluation Pipeline
Evaluates completed models strictly on the hold-out test split (`split="test"`):
```powershell
# Evaluate smoke test runs
python modelling/evaluate_models.py --mode smoke_test

# Evaluate full Stage 1 runs
python modelling/evaluate_models.py --mode full_experiment --stage stage1

# Re-run all evaluations from scratch
python modelling/evaluate_models.py --mode full_experiment --stage stage1 --force
```

### Run Comparison & Report Generation
Generates accuracy/latency/fps/balanced rankings, extracts top models, copies confusion matrices, saves 6 custom plots, and compiles `final_experiment_report.md`:
```powershell
# Compare smoke test runs
python modelling/compare_results.py --mode smoke_test

# Compare full Stage 1 runs
python modelling/compare_results.py --mode full_experiment --stage stage1
```

### Predict Samples
Executes inference using the highest-scoring model on random hold-out test split samples:
```powershell
# Predict using smoke test's best model
python modelling/predict_samples.py --mode smoke_test --num-images 5

# Predict using Stage 1's best model
python modelling/predict_samples.py --mode full_experiment --stage stage1 --num-images 10
```

---

## 4. Directory Structure & Key Deliverables

All outputs are saved under `runs/compare_yolov8_vs_yolo26/<mode>/` (e.g. `runs/compare_yolov8_vs_yolo26/full_experiment/`):

- **`train_runs/`**: Raw Ultralytics run folders per experiment containing training curves (`results.png`), training matrices, and logs.
- **`confusion_matrices/`**: Isolated confusion matrices for every successful experiment.
- **`top_models/`**: Extracted weights (`.pt`) for the Pareto frontier:
  - `best_accuracy_model.pt`
  - `fastest_model.pt`
  - `best_balanced_model.pt`
- **`comparison_plots/`**: Six custom publication-ready graphs:
  1. `map50_95_by_experiment.png` (Bar)
  2. `latency_by_experiment.png` (Bar)
  3. `fps_by_experiment.png` (Bar)
  4. `balanced_score_by_experiment.png` (Bar)
  5. `model_size_vs_map.png` (Scatter)
  6. `latency_vs_map.png` (Scatter)
- **`sample_predictions/`**: Sample visual test outputs with overlay bounding boxes.
- **`training_summary.csv`**: Incremental training telemetry (VRAM, training start/end times, training duration, average time per epoch, actual epochs).
- **`evaluation_summary.csv`**: Hold-out test split metrics, latency/speed breakdown per image, FPS, VRAM usage, and deployment suitability labels.
- **`comparison_summary.csv`**: Combined ranks, 70-20-10 balanced scores, and parameters list.
- **`final_experiment_report.md`**: Presentation-ready markdown summary of accident-detection performance on the hold-out test split.
