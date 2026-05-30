import sys
from pathlib import Path

# Resolve dynamic project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(f"Dynamic Project Root: {PROJECT_ROOT}")

# Dynamically add accident_agent to sys.path
AGENT_DIR = PROJECT_ROOT / "accident_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Model Directory Path Constants
DETECTION_MODEL_DIR = Path(r"C:\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd")
CLASSIFICATION_MODEL_DIR = Path(r"C:\Capstone\Classification\best_severity_classifier")
