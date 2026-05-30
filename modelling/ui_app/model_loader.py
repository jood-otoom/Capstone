import csv
import zipfile
import os
from pathlib import Path
from ui_app.config import PROJECT_ROOT, DETECTION_MODEL_DIR, CLASSIFICATION_MODEL_DIR

def find_best_model() -> Path:
    """
    Look for the absolute highest-ranked model according to the mAP@50-95 metric.
    Prioritizes full experiment yolo26m_lr0001_sgd, then fallback rankings/directories.
    """
    project_root = PROJECT_ROOT
    
    # 0. Prioritize the user-designated best model from the full experiment runs
    full_runs_best = project_root / "full_runs" / "train_runs" / "yolo26m_lr0001_sgd" / "weights" / "best.pt"
    if full_runs_best.exists():
        return full_runs_best
        
    compare_root = project_root / "runs" / "compare_yolov8_vs_yolo26"

    def get_best_from_csv(csv_path: Path):
        if not csv_path.exists():
            return None
        best_pt = None
        best_map = -1.0
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    map_val = row.get("map50_95", "")
                    pt = row.get("best_pt", "")
                    if map_val and pt:
                        try:
                            map_val_f = float(map_val)
                            pt_path = Path(pt)
                            if map_val_f > best_map and pt_path.exists():
                                best_map = map_val_f
                                best_pt = pt_path
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
        return best_pt

    # 1. Full Experiment
    for summary_name in ["comparison_summary.csv", "evaluation_summary.csv"]:
        res = get_best_from_csv(compare_root / "full_experiment" / summary_name)
        if res:
            return res

    # 2. Smoke Test
    for summary_name in ["comparison_summary.csv", "evaluation_summary.csv"]:
        res = get_best_from_csv(compare_root / "smoke_test" / summary_name)
        if res:
            return res

    # In case they are at the root of compare_root
    for summary_name in ["comparison_summary.csv", "evaluation_summary.csv"]:
        res = get_best_from_csv(compare_root / summary_name)
        if res:
            return res

    # 3. Standalone best weights
    detect_runs = project_root / "runs" / "detect"
    if detect_runs.exists():
        best_pts = list(detect_runs.rglob("best.pt"))
        if best_pts:
            best_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return best_pts[0]

    # 4. Fallback native weights inside pretrained_weights/
    pretrained_weights = project_root / "pretrained_weights"
    if pretrained_weights.exists():
        for fm in ["yolov8s.pt", "yolov8n.pt", "yolo26n.pt"]:
            fm_path = pretrained_weights / fm
            if fm_path.exists():
                return fm_path

    return None

def find_best_pt(model_dir: Path) -> Path:
    """
    Finds the correct YOLO weights in model_dir.
    Checks model_dir / "weights" / "best.pt", then model_dir / "best.pt",
    then checks if a "best" folder exists to zip it back to "best.pt",
    then recursively searches for best.pt. Raises FileNotFoundError if not found.
    """
    model_dir = Path(model_dir)
    
    # 1. Prefer model_dir / "weights" / "best.pt"
    path1 = model_dir / "weights" / "best.pt"
    if path1.exists():
        return path1
        
    # 2. Check model_dir / "best.pt"
    path2 = model_dir / "best.pt"
    if path2.exists():
        return path2

    # 3. Check if an unzipped "best" directory exists and needs to be zipped to "best.pt"
    best_dir = model_dir / "best"
    if best_dir.exists() and best_dir.is_dir():
        import zipfile
        import os
        print(f"[Pipeline] Found unzipped weights folder at {best_dir}. Automatically zipping to {path2}...")
        try:
            with zipfile.ZipFile(path2, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(best_dir):
                    for file in files:
                        file_path = Path(root) / file
                        archive_name = file_path.relative_to(best_dir.parent)
                        zipf.write(file_path, archive_name)
            print(f"[Pipeline] Successfully zipped weights to {path2}")
            return path2
        except Exception as e:
            print(f"[Pipeline] Error zipping weights: {e}")
            
    # 4. Recursively search for best.pt
    matches = list(model_dir.rglob("best.pt"))
    if matches:
        return matches[0]
        
    raise FileNotFoundError(f"Could not find best.pt weights in {model_dir}")
