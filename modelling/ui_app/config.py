# modelling/ui_app/config.py
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_DIR = PROJECT_ROOT / "accident_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# ── Model Paths — Docker-aware ───────────────────────────────
# In Docker, model_weights/ is mounted at /app/model_weights
# On Windows, falls back to the absolute C:\Capstone paths

_IN_DOCKER = Path("/app").exists() and Path("/app/model_weights").exists()

if _IN_DOCKER:
    DETECTION_MODEL_DIR    = Path("/app/model_weights/detection")
    CLASSIFICATION_MODEL_DIR = Path("/app/model_weights/classification")
else:
    DETECTION_MODEL_DIR    = Path(r"C:\Desktop\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd")
    CLASSIFICATION_MODEL_DIR = Path(r"C:\Desktop\Capstone\Classification\best_severity_classifier")