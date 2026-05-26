# Safe auto-install of ultralytics (works in notebook and converted script)
try:
    import ultralytics
except ImportError:
    import sys
    import subprocess
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])

import os
import time
import shutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_fscore_support
)

device = 0 if torch.cuda.is_available() else 'cpu'
print(f'Using device: {"GPU" if device == 0 else "CPU"}')

import os
from pathlib import Path

# 1. Get project root, allowing override via CLASSIFICATION_ROOT env var
if "CLASSIFICATION_ROOT" in os.environ:
    PROJECT_ROOT = Path(os.environ["CLASSIFICATION_ROOT"]).resolve()
else:
    try:
        # __file__ is available when run as a python script
        PROJECT_ROOT = Path(__file__).resolve().parent
    except NameError:
        # Default to current working directory when run in notebook
        PROJECT_ROOT = Path.cwd().resolve()

# 2. Define paths, allowing overrides via environment variables
DATASET_PATH = Path(os.getenv("DATASET_PATH", PROJECT_ROOT / "data-v1")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "runs" / "yolo_classifier")).resolve()
FINAL_MODEL_PATH = Path(os.getenv("FINAL_MODEL_PATH", PROJECT_ROOT / "outputs" / "best_severity_classifier.pt")).resolve()

TRAIN_DIR = DATASET_PATH / "train"
VAL_DIR = DATASET_PATH / "valid"
TEST_DIR = DATASET_PATH / "test"

# 3. Automatic Folder Creation
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

# 4. Dataset Validation Block
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"DATASET_PATH does not exist: {DATASET_PATH}")

for folder_name, folder_path in [("train", TRAIN_DIR), ("valid", VAL_DIR), ("test", TEST_DIR)]:
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"{folder_name.capitalize()} folder does not exist or is not a directory: {folder_path}")

# Check class subfolders in train
CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
if not CLASS_NAMES:
    raise FileNotFoundError(f"No class subfolders found in training directory: {TRAIN_DIR}")

NUM_CLASSES = len(CLASS_NAMES)

# Count images helper
def count_images(directory, classes):
    total = 0
    for c in classes:
        class_dir = directory / c
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder '{c}' is missing in directory: {directory}")
        # Count common image file extensions
        total += sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
    return total

train_count = count_images(TRAIN_DIR, CLASS_NAMES)
val_count = count_images(VAL_DIR, CLASS_NAMES)
test_count = count_images(TEST_DIR, CLASS_NAMES)

print("Dataset Validation Passed:")
print("Classes found:", CLASS_NAMES)
print(f"Train images:  {train_count}")
print(f"Val images:    {val_count}")
print(f"Test images:   {test_count}")

# 5. Resolve Test Image Path automatically
if "TEST_IMAGE_PATH" in os.environ:
    test_image_path = Path(os.environ["TEST_IMAGE_PATH"]).resolve()
else:
    test_image_path = None
    if TEST_DIR.exists():
        # Find the first image file in any class subfolder of TEST_DIR
        for class_dir in sorted(TEST_DIR.iterdir()):
            if class_dir.is_dir():
                for img_file in sorted(class_dir.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        test_image_path = img_file
                        break
            if test_image_path:
                break
    
    if not test_image_path:
        test_image_path = PROJECT_ROOT / "test_image.jpg"
        print(f"Warning: No test image found in {TEST_DIR}. Defaulting to fallback path: {test_image_path}")
    else:
        print(f"Selected test image: {test_image_path}")

test_image_path_str = str(test_image_path)

# We define this function first because we'll reuse it for every model variant.
# It returns ALL the metrics your teammate asked for, not just accuracy.

def evaluate_model(model, test_dir, class_names):
    """
    Runs the model on every test image and returns:
    - Accuracy
    - Macro F1 (treats all classes equally — best for imbalanced data)
    - Weighted F1 (weighted by how many samples each class has)
    - Per-class precision, recall, F1
    - Severe accident recall specifically (most important for safety)
    - Confusion matrix
    - Average inference time per image (milliseconds)
    """
    all_preds  = []
    all_labels = []
    times      = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)

            start  = time.time()
            result = model(img_path, verbose=False)[0]
            end    = time.time()

            times.append((end - start) * 1000)          # convert to ms
            all_preds.append(result.probs.top1)
            all_labels.append(class_idx)

    # Overall metrics
    accuracy    = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1    = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_time    = sum(times) / len(times)

    # Per-class metrics
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names)))
    )

    # Severe accident recall specifically
    # Find whichever class index corresponds to 'severe'
    severe_idx    = next((i for i, c in enumerate(class_names) if 'severe' in c.lower()), None)
    severe_recall = recall[severe_idx] if severe_idx is not None else None

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy':      accuracy,
        'macro_f1':      macro_f1,
        'weighted_f1':   weighted_f1,
        'severe_recall': severe_recall,
        'avg_time_ms':   avg_time,
        'precision':     precision,
        'recall':        recall,
        'f1_per_class':  f1_per_class,
        'confusion':     cm,
        'all_preds':     all_preds,
        'all_labels':    all_labels,
    }

print('Evaluation function ready.')

# We train nano, small, and medium, evaluate each one fully,
# then pick the best based on your teammate's priority order:
# 1st → macro F1,  2nd → weighted F1,  3rd → severe recall,  4th → accuracy

VARIANTS   = ['yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt']
RUN_NAMES  = ['nano',           'small',           'medium']

# NOTE: OUTPUT_DIR, DATASET_PATH, TEST_DIR, and CLASS_NAMES are defined globally

all_results = {}   # stores metrics for every variant

for variant, run_name in zip(VARIANTS, RUN_NAMES):
    print(f'\n{"="*50}')
    print(f'  Training: {run_name} ({variant})')
    print(f'{"="*50}')

    # --- Train ---
    m = YOLO(variant)
    m.train(
        data     = str(DATASET_PATH),
        epochs   = 50,
        imgsz    = 224,
        batch    = 32,
        device   = device,
        patience = 10,          # early stopping
        save     = True,
        project  = str(OUTPUT_DIR),
        name     = run_name,
        exist_ok = True,
    )

    # --- Load the best checkpoint saved during training ---
    best_pt = os.path.join(str(OUTPUT_DIR), run_name, 'weights', 'best.pt')
    best_m  = YOLO(best_pt)

    # --- Evaluate using all metrics ---
    print(f'\nEvaluating {run_name} on test set...')
    metrics = evaluate_model(best_m, str(TEST_DIR), CLASS_NAMES)
    metrics['model_path'] = best_pt
    all_results[run_name] = metrics

    print(f'  Accuracy      : {metrics["accuracy"]:.2f}%')
    print(f'  Macro F1      : {metrics["macro_f1"]:.4f}')
    print(f'  Weighted F1   : {metrics["weighted_f1"]:.4f}')
    print(f'  Severe Recall : {metrics["severe_recall"]:.4f}')
    print(f'  Avg Infer Time: {metrics["avg_time_ms"]:.1f} ms')

print('\n All variants trained and evaluated.')

# Build a comparison table
rows = []
for run_name, m in all_results.items():
    rows.append({
        'Model':          run_name,
        'Accuracy (%)':   round(m['accuracy'],      2),
        'Macro F1':       round(m['macro_f1'],       4),
        'Weighted F1':    round(m['weighted_f1'],    4),
        'Severe Recall':  round(m['severe_recall'],  4),
        'Avg Time (ms)':  round(m['avg_time_ms'],    1),
    })

df_compare = pd.DataFrame(rows).set_index('Model')
print('\n=== Model Comparison ===')
print(df_compare.to_string())

# ── Pick the best using your teammate's priority order ──
# Sort by: macro_f1 DESC, weighted_f1 DESC, severe_recall DESC, accuracy DESC
best_run = max(
    all_results.keys(),
    key=lambda r: (
        all_results[r]['macro_f1'],
        all_results[r]['weighted_f1'],
        all_results[r]['severe_recall'],
        all_results[r]['accuracy'],
    )
)

print(f'\n✅ Best model selected: {best_run}')
print(f'   Macro F1      : {all_results[best_run]["macro_f1"]:.4f}')
print(f'   Weighted F1   : {all_results[best_run]["weighted_f1"]:.4f}')
print(f'   Severe Recall : {all_results[best_run]["severe_recall"]:.4f}')
print(f'   Accuracy      : {all_results[best_run]["accuracy"]:.2f}%')

# ── Load the actual best model ──
best_model = YOLO(all_results[best_run]['model_path'])
print(f'\n✅ Best model loaded from: {all_results[best_run]["model_path"]}')

metrics = all_results[best_run]

print(f'\n{"="*55}')
print(f'  Final Report — Best Model: {best_run}')
print(f'{"="*55}')
print(f'  Accuracy         : {metrics["accuracy"]:.2f}%')
print(f'  Macro F1         : {metrics["macro_f1"]:.4f}')
print(f'  Weighted F1      : {metrics["weighted_f1"]:.4f}')
print(f'  Avg Inference    : {metrics["avg_time_ms"]:.1f} ms per image')
print(f'\n  Per-class breakdown:')
print(f'  {"Class":<22} {"Precision":>10} {"Recall":>10} {"F1":>10}')
print(f'  {"-"*54}')
for i, cls in enumerate(CLASS_NAMES):
    print(
        f'  {cls:<22}'
        f' {metrics["precision"][i]:>10.4f}'
        f' {metrics["recall"][i]:>10.4f}'
        f' {metrics["f1_per_class"][i]:>10.4f}'
    )

# Full sklearn report
print(f'\n  Full Classification Report:')
print(classification_report(
    metrics['all_labels'],
    metrics['all_preds'],
    target_names=CLASS_NAMES
))

cm            = metrics['confusion']
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1)
ax1.set_title(f'Confusion Matrix — Raw Counts ({best_run})')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2)
ax2.set_title(f'Confusion Matrix — Normalized ({best_run})')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# This is the file you share with your teammate.
# It contains everything she needs to load the model and run predictions.

# Copy to the globally configured FINAL_MODEL_PATH
shutil.copy(all_results[best_run]['model_path'], str(FINAL_MODEL_PATH))

print("==================================================")
print("              FINAL MODEL EXECUTION SUMMARY       ")
print("==================================================")
print(f"Best Model Name      : {best_run}")
print(f"Accuracy             : {all_results[best_run]['accuracy']:.2f}%")
print(f"Macro F1             : {all_results[best_run]['macro_f1']:.4f}")
print(f"Weighted F1          : {all_results[best_run]['weighted_f1']:.4f}")
if all_results[best_run].get('severe_recall') is not None:
    print(f"Severe Recall        : {all_results[best_run]['severe_recall']:.4f}")
else:
    print("Severe Recall        : Not Available")
print(f"Final Saved Model Path: {FINAL_MODEL_PATH}")
print("==================================================")

def predict_image(image_path, model=best_model):
    """
    Predict severity class for one image.
    Returns clean values for later integration.
    """

    result = model(image_path, verbose=False)[0]

    pred_idx = int(result.probs.top1)
    confidence = float(result.probs.top1conf.item())
    probabilities = result.probs.data.tolist()

    output = {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probabilities": {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, probabilities)
        }
    }

    return output



prediction = predict_image(test_image_path_str)
print(prediction)
