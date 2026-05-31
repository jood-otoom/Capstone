# AcciEye | Severity Classification Module

This directory contains the **Severity Classification Module** of the AcciEye system. The purpose of this module is to perform a downstream evaluation of localized accident bounding boxes, classifying the crash severity as `no-accident`, `moderate-accident`, or `severe-accident`.

---

## 📖 Module Overview

While the primary detector (`YOLOv26m`) localizes the presence of an accident, it does not assess its severity. To prevent overloading emergency dispatch teams with minor incidents, the visual crop of the accident is routed to this classification module. 

The module evaluates the cropped visual patch and classifies it. If the classification result contains `severe` or `high` severity, the telemetry pipeline triggers an immediate escalation workflow.

---

## 📂 Dataset Details

The classification model is trained on a categorized severity dataset located in `Classification/data-v1/`.

* **Dataset Splits**:
  * `train/`: Training split containing class-specific folders.
  * `valid/`: Validation split for checkpoint evaluation.
  * `test/`: Testing split for final model comparison.
* **Target Classes**:
  1. `no-accident`: Images where no collision or vehicle hazard is present.
  2. `moderate-accident`: Collisions with low-to-moderate vehicle body deformation or minor road blockages.
  3. `severe-accident`: High-impact collisions with severe cabin intrusion, fires, rollovers, or major blockages.

---

## ⚙️ Model Selection & Ranking Criteria

The training and evaluation pipeline evaluates multiple YOLO classification architectures. It systematically trains three model variants on the dataset:
1. `yolov8n-cls` (Nano Classification Model)
2. `yolov8s-cls` (Small Classification Model)
3. `yolov8m-cls` (Medium Classification Model)

### Ranking Priority Order
To select the best model for production deployment, the evaluation script (`yolo-classification.py`) ranks the models according to the following metric priorities:
1. **Macro F1 Score** (Priority 1): Focuses on treating all classes equally, protecting accuracy on imbalanced categories.
2. **Weighted F1 Score** (Priority 2): Accounts for class distribution density.
3. **Severe Accident Recall** (Priority 3): Ensures that severe accidents are never missed (highest clinical safety priority).
4. **Accuracy** (Priority 4): Measures overall prediction accuracy.

The best-performing model is loaded and copied to the production weights directory:

```text
Classification/best_severity_classifier/best.pt
```

---

## 🛠️ Code Structure

* `yolo-classification.py`: The central execution script containing dataset validation, training parameters, early stopping, performance ranking, classification reports, confusion matrix plotting, and image prediction helpers.
* `yolo-classification.ipynb`: The Jupyter Notebook variant used during initial experimentation and model plotting.
* `best_severity_classifier/best.pt`: The finalized PyTorch weights deployed in the production Gradio application.

---

## 🚀 How to Train and Run

### 1. Requirements Setup
Ensure all necessary libraries are installed in your virtual environment:

```powershell
pip install ultralytics torch pandas matplotlib seaborn scikit-learn pillow
```

### 2. Run Model Training and Grid Sweep
Execute the training script to perform dataset validation, train all three variants for 50 epochs, evaluate metrics on the test split, and export the best checkpoint to the production directory:

```powershell
python Classification/yolo-classification.py
```

### 3. Inline Prediction API
To load the deployed model and predict an individual image patch from another Python script, use the `predict_image` API:

```python
from ultralytics import YOLO
import os

CLASS_NAMES = ["moderate-accident", "no-accident", "severe-accident"]
model = YOLO("Classification/best_severity_classifier/best.pt")

def predict_image(image_path, model=model):
    result = model(image_path, verbose=False)[0]
    pred_idx = int(result.probs.top1)
    confidence = float(result.probs.top1conf.item())
    
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence
    }

prediction = predict_image("path/to/cropped_patch.jpg")
print("Severity classification result:", prediction)
```

---

## 📊 Integration in the Inference Pipeline

In the production UI flow (`modelling/ui_app/combined_pipeline.py`):
1. **Detection**: `YOLOv26m` evaluates the frame and outputs accident bounding boxes.
2. **Cropping**: The highest-confidence bounding box is cropped from the image matrix.
3. **Classification**: The cropped patch is passed to the severity model (`best_severity_classifier/best.pt`).
4. **Logging**: The classification results are forwarded to `incident_logger.py` to dictate escalation flags and webhook payloads.
