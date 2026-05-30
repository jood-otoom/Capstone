# Accident Detection Module

## Overview
This module implements the accident detection stage of the project using Ultralytics YOLO models. In the current workspace, detection training logic lives under `modelling/`, the deployed detector checkpoint lives under `full_runs/train_runs/`, and the Gradio UI calls the detector through `modelling/ui_app/combined_pipeline.py`.

## Purpose
The purpose of this module is to determine whether an uploaded road image or video contains an accident. The detector produces accident-localization boxes and confidence values, then passes the result forward to the downstream severity classifier and UI only when an accident is detected.

## Project Structure
The files and folders below are the detection-related pieces that are present in the current workspace:

| Path | Role |
|---|---|
| `modelling/config.py` | Central experiment configuration for dataset paths, model lists, output roots, and training settings. |
| `modelling/check_dataset.py` | Validates the `data_processed` dataset structure and label/image split health. |
| `modelling/train_models.py` | Trains YOLO experiments for `smoke_test` or `full_experiment` modes. |
| `modelling/evaluate_models.py` | Evaluates trained `best.pt` checkpoints on the `test` split. |
| `modelling/test_ui.py` | Gradio UI entry point. It wires file upload events to the detection pipeline. |
| `modelling/ui_app/config.py` | Defines `PROJECT_ROOT` and the hardcoded detector/classifier model directories used by the UI. |
| `modelling/ui_app/model_loader.py` | Resolves `best.pt` paths for the detector and classifier. |
| `modelling/ui_app/detection_pipeline.py` | Converts YOLO result objects into a boolean accident / no-accident decision. |
| `modelling/ui_app/combined_pipeline.py` | Main image and video inference flow used by the UI. |
| `modelling/incident_logger.py` | Logs confirmed image incidents and can send them to a webhook. |
| `data_processed/data.yaml` | Source dataset definition used by the training code. |
| `full_runs/data_absolute.yaml` | Existing absolute-path dataset YAML from a previous run environment. |
| `full_runs/train_runs/` | Historical YOLO run folders with checkpoints and plots for 32 experiment variants. |
| `full_runs/training_summary.csv` | Historical training summary for the full experiment set. |
| `full_runs/comparison_rankings/full_experiment_ranked_runs.csv` | Existing ranked summary of full experiment runs. |
| `full_runs/comparison_rankings/rank_full_experiment_runs.py` | Script that ranks the existing `full_runs/train_runs` results. |

## Model Used
Two model families appear in the current training code:

- `modelling/config.py` defines the full experiment grid across `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`, `yolo26s`, `yolo26m`, `yolo26l`, and `yolo26x`.
- The Gradio UI is currently configured to use the `yolo26m_lr0001_sgd` run as its detector source through `modelling/ui_app/config.py`.

Current detector path used by the UI:

```text
C:\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd
```

Resolved checkpoint path:

```text
C:\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd\weights\best.pt
```

`modelling/ui_app/model_loader.py` resolves weights in this order:

1. `<model_dir>/weights/best.pt`
2. `<model_dir>/best.pt`
3. A `best/` directory inside `model_dir`
4. Recursive search for `best.pt`

## Dataset
The live dataset definition is in `data_processed/data.yaml`:

```yaml
path: C:/Capstone/data_processed
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['accident']
```

Confirmed dataset facts from the current workspace:

| Split | Image files | `.txt` label files |
|---|---:|---:|
| `train` | 6410 | 6410 |
| `valid` | 254 | 254 |
| `test` | 712 | 712 |

The detector is therefore trained as a single-class object detector with one class: `accident`.

An additional generated YAML file also exists at `full_runs/data_absolute.yaml`, but it currently points to a previous Linux path:

```text
/root/Capstone/Capstone_Files/data_processed
```

That file is not directly reusable on this Windows workspace without regeneration or manual editing.

## Training
### Current training script
The active training entry point is:

```text
modelling/train_models.py
```

### Training modes defined in code
`modelling/config.py` defines:

- `smoke_test`
- `full_experiment`

For the current `full_experiment` stage 1 configuration, the code builds 32 runs:

- 8 model variants
- 2 learning rates: `0.01` and `0.001`
- 2 optimizers: `SGD` and `AdamW`

Fixed training settings for stage 1:

| Setting | Value |
|---|---:|
| Epochs | 50 |
| Image size | 640 |
| Batch size | 8 |
| Patience | 15 |
| Seed | 42 |
| Deterministic | `True` |

The training code calls `ultralytics.YOLO(...).train(...)` with:

- `data=<absolute_yaml>`
- `project=<train_runs_root>`
- `name=<experiment_name>`
- `plots=True`

### Actual training command in this codebase
```powershell
python modelling/train_models.py --mode full_experiment --stage stage1
```

Smoke test command:

```powershell
python modelling/train_models.py --mode smoke_test
```

### Important path note
The current training code writes to:

```text
C:\Capstone\runs\compare_yolov8_vs_yolo26\<mode>\train_runs\
```

That `runs/` directory is not present in the current workspace. The detector checkpoint actually used by the UI comes from the older `full_runs/` artifact tree instead.

## Inference
There is no standalone CLI-only prediction script present in the current workspace. The implemented inference entry points are the UI functions inside `modelling/ui_app/combined_pipeline.py`.

### Image inference flow
Main function:

```text
predict_accident_gui(input_image, conf_threshold)
```

Flow:

1. `predict_accident_gui()` calls `pipeline.process_image(...)`.
2. `AccidentSeverityPipeline.detect_accident()` runs the YOLO detector with `self.detector(frame, conf=conf_threshold, verbose=False)`.
3. `detect_accident_from_collection()` decides whether the result counts as an accident.
4. The first YOLO result is rendered with `results[0].plot()` for UI display.
5. If an accident is detected, the highest-confidence box is cropped and passed to the severity classifier. If crop classification fails, the full frame is used as a fallback.

### Video inference flow
Main function:

```text
predict_accident_video_gui(video_path, conf_threshold)
```

Flow:

1. `pipeline.detector.predict(...)` is called on the video path.
2. The code saves predictions to `PROJECT_ROOT / "runs" / "detect"`.
3. The function scans frame-level results and counts frames where an accident is detected.
4. The highest-confidence crop from the best frame is passed to the severity classifier.
5. If the saved output is AVI, `modelling/ui_app/media_utils.py` converts it to MP4 for preview.

The video detector is called with these explicit parameters:

- `save=True`
- `imgsz=640`
- `vid_stride=3`

### Automatic UI thresholds
In `run_model_inference_flow()`:

- Images are sent to `predict_accident_gui(..., conf_threshold=0.60)`
- Videos are sent to `predict_accident_video_gui(..., conf_threshold=0.70)`

### Accident decision rule
`modelling/ui_app/detection_pipeline.py` marks a result as an accident when:

- a detected class name contains `"accident"`, or
- a detected class id is `0`, or
- any detection boxes exist at all

For this single-class dataset, that effectively means any detected box is treated as an accident.

## Outputs
### Training run outputs
Each historical run folder under `full_runs/train_runs/<run_name>/` contains real Ultralytics artifacts such as:

- `weights/best.pt`
- `weights/last.pt`
- `args.yaml`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `BoxF1_curve.png`
- `BoxPR_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `labels.jpg`
- `train_batch*.jpg`
- `val_batch*_pred.jpg`
- `val_batch*_labels.jpg`

### Image inference outputs
Image inference returns UI objects rather than saving a detector image to disk by default:

- annotated image array
- status HTML
- alert banner HTML
- alert signal markup

If an accident is detected in the image flow, `modelling/incident_logger.py` can also:

- save the BGR frame to `C:\Capstone\incident_logs\frames\`
- append metadata to `C:\Capstone\incident_logs\incidents.csv`
- append JSON lines to `C:\Capstone\incident_logs\incidents.jsonl`
- post the incident to the configured webhook

### Video inference outputs
Video inference writes the rendered prediction video under:

```text
C:\Capstone\runs\detect\video_predictions_<timestamp>\
```

The UI then returns the preview path of the saved video, converting AVI output to MP4 when needed.

## Integration With UI
The Gradio UI entry point is `modelling/test_ui.py`.

Integration path:

1. `test_ui.py` binds `evidence_upload.upload(...)` to `run_model_inference_flow`.
2. `run_model_inference_flow()` determines whether the uploaded file is an image or a video.
3. It calls `predict_accident_gui()` or `predict_accident_video_gui()`.
4. The pipeline stores the last detection confidence, best frame, and best crop in `AccidentSeverityPipeline`.
5. `run_agent_analysis_flow()` is triggered after upload and only proceeds with downstream AI analysis when an accident was detected.

Model loading behavior:

- `modelling/ui_app/config.py` hardcodes the detection model directory.
- `AccidentSeverityPipeline.load_models()` loads the detector first.
- If the severity classifier fails to load, detection still remains active.

## Best Model / Checkpoints
### Deployed detector checkpoint
The detector currently wired into the UI is:

```text
C:\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd\weights\best.pt
```

The paired `last.pt` is stored at:

```text
C:\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd\weights\last.pt
```

### Historical checkpoints
All 32 historical run directories under `full_runs/train_runs/` contain `weights/best.pt` and `weights/last.pt`.

### Existing ranking artifact
The current ranking file `full_runs/comparison_rankings/full_experiment_ranked_runs.csv` places `yolo26m_lr0001_sgd` at rank 1.

## Evaluation Results
The current workspace does not contain a generated `evaluation_summary.csv` from `modelling/evaluate_models.py`, so separate test-split evaluation metrics could not be quoted from disk.

The metrics below are therefore taken from the files that do exist in `full_runs/`.

### Top ranked runs from `full_runs/comparison_rankings/full_experiment_ranked_runs.csv`

| Rank | Run | Best epoch | Best mAP50-95 | Best mAP50 | Precision | Recall | Training time (min) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `yolo26m_lr0001_sgd` | 1 | 0.615 | 0.955 | 1.00000 | 0.95513 | 34.6434 |
| 2 | `yolo26x_lr0001_sgd` | 1 | 0.614 | 0.965 | 1.00000 | 0.96731 | 83.7857 |
| 3 | `yolo26l_lr0001_sgd` | 1 | 0.609 | 0.955 | 1.00000 | 0.95256 | 57.0917 |

### Deployed run details from `full_runs/train_runs/yolo26m_lr0001_sgd/results.csv`

- Best recorded validation `mAP50-95(B)`: `0.615`
- Best recorded validation `mAP50(B)`: `0.955`
- Best recorded validation precision: `1.0`
- Best recorded validation recall: `0.95513`
- Final recorded epoch: `16`
- Final epoch `mAP50-95(B)`: `0.548`

These are validation metrics recorded during training, not a separate exported test-only summary.

## Requirements
No dedicated detection requirements file is present in the current workspace. The dependencies below are inferred from the active detection code and ranking scripts.

### Core detector and training
- `ultralytics`
- `torch`
- `pyyaml`

### UI inference
- `gradio`
- `opencv-python`
- `numpy`

### Ranking and offline analysis
- `pandas`
- `matplotlib`
- `openpyxl`

### Incident logging
- `requests`

## How to Run
### 1. Install the libraries used by the active detection code
```powershell
pip install ultralytics torch pyyaml gradio opencv-python numpy requests pandas matplotlib openpyxl
```

### 2. Verify the dataset path
The live dataset config is:

```text
C:\Capstone\data_processed\data.yaml
```

Optional dataset check:

```powershell
python modelling/check_dataset.py
```

### 3. Train detection models
Full experiment:

```powershell
python modelling/train_models.py --mode full_experiment --stage stage1
```

Smoke test:

```powershell
python modelling/train_models.py --mode smoke_test
```

### 4. Evaluate trained checkpoints
```powershell
python modelling/evaluate_models.py --mode full_experiment --stage stage1
```

### 5. Run the detection UI
```powershell
python modelling/test_ui.py
```

The UI launches on `127.0.0.1:7860` when that port is available.

## Notes
- `modelling/ui_app/config.py` hardcodes the detector directory to `C:\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd`. If you want to switch checkpoints, update that file.
- `full_runs/data_absolute.yaml`, `full_runs/training_summary.csv`, and the per-run `args.yaml` files contain old `/root/Capstone/Capstone_Files/...` absolute paths from a previous environment. They should be regenerated or edited before reuse on this machine.
- The active training code is configured to write new experiments under `C:\Capstone\runs\compare_yolov8_vs_yolo26\...`, but that `runs/` directory is not currently present in this workspace.
- `check_dataset.py` currently treats `test` labels as not required, while `evaluate_models.py` evaluates with `split="test"`. The current dataset does contain `test/labels/*.txt`, so those two assumptions are not fully aligned.
- `train_models.py` still advertises a YOLOv8n smoke test in its CLI help text, but `modelling/config.py` currently defines smoke-test runs using `yolov8s`.
- `modelling/config.py` still defines comparison and prediction output paths, but no `compare_results.py` or standalone prediction script is present in `modelling` now.
- There is no standalone prediction script currently present under `modelling/`; inference is currently exposed through the Gradio UI pipeline.
- `incident_logger.py` currently has `ENABLE_INCIDENT_WEBHOOK = True`, so image detections may trigger outbound webhook calls in addition to local logging.
