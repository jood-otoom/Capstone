import subprocess
import sys
import os
from pathlib import Path
import csv
from html import escape
import time

# Fallback lightweight installation check
try:
    import gradio as gr
except ImportError:
    print("Gradio not found. Installing gradio...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

import numpy as np

# Resolve dynamic project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Dynamic Project Root: {PROJECT_ROOT}")

# Dynamically add accident_agent to sys.path
AGENT_DIR = PROJECT_ROOT / "accident_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Ensure required libraries are installed
for lib_name, import_name in [
    ("langchain", "langchain"),
    ("sentence-transformers", "sentence_transformers"),
    ("langchain-huggingface", "langchain_huggingface"),
    ("langchain-openai", "langchain_openai"),
    ("langchain-community", "langchain_community"),
    ("networkx", "networkx"),
    ("faiss-cpu", "faiss"),
    ("pypdf", "pypdf"),
    ("pdfplumber", "pdfplumber"),
    ("python-dotenv", "dotenv")
]:
    try:
        __import__(import_name)
    except ImportError:
        print(f"[Self-Healing] Missing dependency '{lib_name}'. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name])
        except Exception as e:
            print(f"[Self-Healing WARNING] Failed to auto-install {lib_name}: {e}")

try:
    import dotenv
    dotenv.load_dotenv(dotenv_path=str(AGENT_DIR / ".env"))
except Exception as e:
    print(f"Error loading agent env: {e}")

class APIKeyManager:
    def __init__(self):
        self.keys = [
            "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",
            "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"
        ]
        self.current_index = 0

    def get_current_key(self) -> str:
        return self.keys[self.current_index]

    def rotate_key(self) -> str:
        self.current_index = (self.current_index + 1) % len(self.keys)
        os.environ["OPENROUTER_API_KEY"] = self.keys[self.current_index]
        try:
            from app.core.config import settings
            settings.OPENROUTER_API_KEY = self.keys[self.current_index]
        except Exception:
            pass
        print(f"[APIKeyManager] Credit exhausted or API key failed. Rotated to key index: {self.current_index}")
        return self.keys[self.current_index]

api_key_manager = APIKeyManager()
# Initialize environment keys
os.environ["OPENROUTER_API_KEY"] = api_key_manager.get_current_key()
os.environ["HF_TOKEN"] = "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"

def safe_agent_call(agent, method_name, *args, **kwargs):
    last_err = None
    for attempt in range(len(api_key_manager.keys)):
        current_key = api_key_manager.get_current_key()
        os.environ["OPENROUTER_API_KEY"] = current_key
        try:
            from app.core.config import settings
            settings.OPENROUTER_API_KEY = current_key
        except Exception:
            pass
            
        # Update LLM dynamically if it exists
        if agent and hasattr(agent, "llm") and agent.llm:
            if hasattr(agent.llm, "openai_api_key"):
                agent.llm.openai_api_key = current_key
            if hasattr(agent.llm, "api_key"):
                agent.llm.api_key = current_key

        try:
            method = getattr(agent, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            print(f"[safe_agent_call] Attempt {attempt+1} using key index {api_key_manager.current_index} failed: {e}")
            
            # Rotate key on credit or auth failures
            is_credit_or_auth = any(word in err_msg for word in [
                "credit", "balance", "insufficient", "payment", "402", "unauthorized", "api_key", "401", "403"
            ])
            if is_credit_or_auth:
                api_key_manager.rotate_key()
            else:
                raise e
    raise RuntimeError(f"All API keys failed or exhausted credit. Last error: {last_err}")

def get_accident_agent():
    """
    Safely load and instantiate the AccidentAgent.
    Returns the agent instance or raises descriptive errors.
    """
    agent_path = PROJECT_ROOT / "accident_agent"
    if not agent_path.exists():
        raise FileNotFoundError(f"Accident Agent folder is missing at: {agent_path}")

    required_files = ["app/services/agent_service.py", "app/core/graph_logic.py", "app/core/prompts.py"]
    for rf in required_files:
        p = agent_path / rf
        if not p.exists():
            raise FileNotFoundError(f"Accident Agent is missing crucial file: {rf}")

    os.environ["OPENROUTER_API_KEY"] = api_key_manager.get_current_key()
    
    from app.services.agent_service import AccidentAgent
    return AccidentAgent()



def get_media_label(source: str) -> str:
    return "Video Analysis" if source == "video" else "Image Analysis"


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


# Model Directory Path Constants
DETECTION_MODEL_DIR = Path(r"C:\Capstone\full_runs\train_runs\yolo26m_lr0001_sgd")
CLASSIFICATION_MODEL_DIR = Path(r"C:\Capstone\Classification\best_severity_classifier")


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


def render_final_summary_html(
    accident_detected: bool | None,
    detection_conf: float,
    severity_label: str | None,
    severity_conf: float,
    escalation_status: str | None = None,
    assistant_summary: str = ""
) -> str:
    if accident_detected is None:
        return """
        <div style="background: #ffffff; border: 1px solid rgba(147, 197, 253, 0.42); border-radius: 18px; padding: 20px 24px; box-shadow: 0 4px 18px rgba(29, 78, 216, 0.03); width: 100%;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="font-size: 1.1rem; color: #1e40af;">&#128203;</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #0f172a;">Step 5: Final Detection Summary</h3>
            </div>
            <p style="margin: 0 0 16px; font-size: 0.88rem; color: #64748b;">Review the final accident detection, severity, escalation, and AI assistant summary.</p>
            <div style="text-align: center; color: #64748b; padding: 30px; font-size: 0.92rem; background: rgba(248, 250, 252, 0.6); border: 1px dashed rgba(147, 197, 253, 0.35); border-radius: 12px;">
                &#9993; <b>Awaiting Sequential Pipeline execution...</b><br>
                Upload an image or video above and click Run Inference to trigger the sequential model pipeline. The final unified report will populate here.
            </div>
        </div>
        """

    det_val = "Accident" if accident_detected else "No Accident"
    det_conf_str = f"{detection_conf * 100:.1f}%" if accident_detected else "-"
    
    display_sev = severity_display_label(severity_label) if (severity_label and severity_label not in ["Model not found", "Not Applied", "Classification Failed"]) else "Pending"
    if not accident_detected:
        display_sev = "Low Severity" # Match severity display rules for no accident

    if accident_detected and severity_label not in ["Model not found", "Not Applied", "Classification Failed", None]:
        sev_conf_str = f"{severity_conf * 100:.1f}%"
    else:
        sev_conf_str = "-"

    if escalation_status is None:
        if severity_label and "severe" in str(severity_label).lower():
            esc_val = "Escalated"
        else:
            esc_val = "Not Escalated"
    else:
        esc_val = escalation_status

    det_badge_style = "background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;" if accident_detected else "background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;"
    
    if "severe" in display_sev.lower():
        sev_badge_style = "background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;"
    elif "moderate" in display_sev.lower():
        sev_badge_style = "background: #fffbeb; border: 1px solid #fef3c7; color: #b45309;"
    else:
        sev_badge_style = "background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8;"
        
    esc_badge_style = "background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; font-weight: 800;" if esc_val == "Escalated" else "background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;"

    if assistant_summary:
        preview_text = escape(assistant_summary)
        parts = preview_text.split('**')
        new_text = ""
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                new_text += f"<b>{part}</b>"
            else:
                new_text += part
        preview_text = new_text.replace('\\n', '<br>').replace('\n', '<br>')
        if len(preview_text) > 400:
            preview_text = preview_text[:397] + "..."
    else:
        preview_text = "Pending AI analysis report. Once accident detection finishes, the liability report preview will appear here."

    return f"""
    <div style="background: #ffffff; border: 1px solid rgba(147, 197, 253, 0.42); border-radius: 18px; padding: 22px 24px; box-shadow: 0 4px 18px rgba(29, 78, 216, 0.03); width: 100%;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="font-size: 1.1rem; color: #2563eb;">&#128203;</span>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #0f172a;">Step 5: Final Detection Summary</h3>
        </div>
        <p style="margin: 0 0 18px; font-size: 0.88rem; color: #64748b;">Review the final accident detection, severity, escalation, and AI assistant summary.</p>
        
        <div style="display: flex; flex-direction: column; gap: 16px; width: 100%;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; width: 100%;">
                <div style="{det_badge_style} padding: 12px 14px; border-radius: 12px; display: flex; flex-direction: column; gap: 3px;">
                    <span style="font-size: 0.76rem; text-transform: uppercase; font-weight: 800; opacity: 0.8; letter-spacing: 0.03em;">Accident Detection</span>
                    <span style="font-size: 1.1rem; font-weight: 900;">{det_val}</span>
                    <span style="font-size: 0.82rem; font-weight: 700; opacity: 0.9;">Confidence: {det_conf_str}</span>
                </div>
                
                <div style="{sev_badge_style} padding: 12px 14px; border-radius: 12px; display: flex; flex-direction: column; gap: 3px;">
                    <span style="font-size: 0.76rem; text-transform: uppercase; font-weight: 800; opacity: 0.8; letter-spacing: 0.03em;">Severity Classification</span>
                    <span style="font-size: 1.1rem; font-weight: 900;">{display_sev}</span>
                    <span style="font-size: 0.82rem; font-weight: 700; opacity: 0.9;">Confidence: {sev_conf_str}</span>
                </div>
                
                <div style="{esc_badge_style} padding: 12px 14px; border-radius: 12px; display: flex; flex-direction: column; gap: 3px; justify-content: center;">
                    <span style="font-size: 0.76rem; text-transform: uppercase; font-weight: 800; opacity: 0.8; letter-spacing: 0.03em;">Escalation Status</span>
                    <span style="font-size: 1.1rem; font-weight: 900;">{esc_val}</span>
                </div>
            </div>
            
            <div style="background: #f8fafc; border: 1px solid rgba(147, 197, 253, 0.25); border-radius: 12px; padding: 14px 18px; width: 100%;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #1e3a8a; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                    <span>&#9993;</span> AI Assistant Report Preview
                </div>
                <div style="font-size: 0.88rem; line-height: 1.55; color: #334155; word-break: break-word; white-space: pre-wrap; max-height: 160px; overflow-y: auto; padding-right: 6px;">{preview_text}</div>
            </div>
        </div>
    </div>
    """


def severity_display_label(raw_label):
    if raw_label is None:
        return ""
    normalized = str(raw_label).strip().lower().replace("_", " ").replace("-", " ")
    if normalized == "no accident":
        return "Low Severity"
    if normalized == "severe":
        return "Severe"
    if normalized == "moderate":
        return "Moderate"
    return str(raw_label).strip().title()


def format_severity_label(label: str) -> str:
    """Formats raw severity labels into clean title-cased strings for the UI."""
    if not label:
        return "N/A"
    if label in ["Model not found", "Not Applied", "Classification Failed"]:
        return label
    return severity_display_label(label)


def build_pipeline_status_banner(
    accident_detected: bool,
    detection_conf: float,
    severity_label: str,
    severity_conf: float,
    num_detections: int = 0,
    source: str = "image",
    processed_video_path: str = None
) -> str:
    """Generates a premium, highly detailed HTML dashboard banner for sequential pipeline results."""
    tone = "alert" if accident_detected else "safe"
    icon = "&#9888;" if accident_detected else "&#10003;"
    
    title = "Accident Detected" if accident_detected else "No Accident Detected"
    
    det_result = "Accident" if accident_detected else "No Accident"
    det_conf_str = f"{detection_conf * 100:.1f}%" if accident_detected else "-"
    
    formatted_sev = format_severity_label(severity_label)
    sev_conf_str = f"{severity_conf * 100:.1f}%" if (accident_detected and severity_label not in ["Model not found", "Not Applied", "Classification Failed"]) else "-"
    
    # Extra statistics text
    if accident_detected:
        if source == "video":
            msg = f"Sequential pipeline completed: accident detected in {num_detections} frame(s)."
        else:
            msg = f"Sequential pipeline completed: {num_detections} accident-related object(s) detected."
    else:
        msg = "Inference completed and no accident class was detected above the selected confidence threshold."

    video_stats_html = ""
    if source == "video" and processed_video_path:
        video_stats_html = f"""
        <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(0, 0, 0, 0.08); font-size: 0.9rem; opacity: 0.95;">
            &#128253; <b>Video Stats:</b> Detected in <b>{num_detections}</b> frame(s). 
            Processed video saved to: <code style="font-size: 0.82rem; word-break: break-all; background: rgba(255, 255, 255, 0.5); padding: 2px 6px; border-radius: 4px; font-family: monospace;">{escape(processed_video_path)}</code>
        </div>
        """

    return f"""
    <div class="status-banner tone-{tone}" role="status" aria-live="polite" style="display: flex; flex-direction: column; gap: 12px; width: 100%;">
        <div style="display: flex; align-items: center; gap: 14px; width: 100%;">
            <div class="status-icon">{icon}</div>
            <div class="status-copy" style="flex: 1;">
                <p class="status-label" style="font-size: 1.15rem; font-weight: 800; margin: 0;">{title}</p>
                <p class="status-message" style="margin: 4px 0 0; font-size: 0.92rem; opacity: 0.9;">{escape(msg)}</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; width: 100%; margin-top: 4px;">
            <div style="background: rgba(255, 255, 255, 0.82); color: #1e3a8a !important; padding: 12px 16px; border-radius: 14px; border: 1px solid rgba(30, 58, 138, 0.15); display: flex; flex-direction: column; gap: 2px;">
                <span style="font-size: 0.78rem; text-transform: uppercase; font-weight: 800; opacity: 0.75; letter-spacing: 0.04em; color: #1e3a8a !important;">Accident Detection Result</span>
                <span style="font-size: 1.15rem; font-weight: 800; color: #1e3a8a !important;">{escape(det_result)}</span>
                <span style="font-size: 0.88rem; font-weight: 600; opacity: 0.9; margin-top: 4px; color: #1e3a8a !important;">Confidence: <b style="color: #1e3a8a !important;">{det_conf_str}</b></span>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.82); color: #1e3a8a !important; padding: 12px 16px; border-radius: 14px; border: 1px solid rgba(30, 58, 138, 0.15); display: flex; flex-direction: column; gap: 2px;">
                <span style="font-size: 0.78rem; text-transform: uppercase; font-weight: 800; opacity: 0.75; letter-spacing: 0.04em; color: #1e3a8a !important;">Severity Classification Result</span>
                <span style="font-size: 1.15rem; font-weight: 800; color: #1e3a8a !important;">{escape(formatted_sev)}</span>
                <span style="font-size: 0.88rem; font-weight: 600; opacity: 0.9; margin-top: 4px; color: #1e3a8a !important;">Confidence: <b style="color: #1e3a8a !important;">{sev_conf_str}</b></span>
            </div>
        </div>
        
        {video_stats_html}
    </div>
    """


class AccidentSeverityPipeline:
    """
    Unified accident detection and severity classification sequential pipeline.
    Ensures safe error handling, automatic weight resolving, zipping and fallbacks.
    """
    def __init__(self):
        self.detector = None
        self.classifier = None
        self.detector_path = None
        self.classifier_path = None
        self.classifier_missing = False
        self.last_det_conf = 0.0
        self.last_sev_label = None
        self.last_sev_conf = 0.0

    def load_models(self):
        # 1. Resolve and Load Detection Model
        print("[Pipeline] Resolving accident detection weights...")
        try:
            self.detector_path = find_best_pt(DETECTION_MODEL_DIR)
            print(f"[Pipeline] Resolved detection weights to: {self.detector_path}")
            self.detector = YOLO(str(self.detector_path))
            print("[Pipeline] Accident detection model loaded successfully!")
        except Exception as e:
            print(f"[Pipeline] ERROR loading detection model: {e}")
            self.detector = None
            raise e

        # 2. Resolve and Load Severity Classification Model
        print("[Pipeline] Resolving accident severity classification weights...")
        try:
            self.classifier_path = find_best_pt(CLASSIFICATION_MODEL_DIR)
            print(f"[Pipeline] Resolved classification weights to: {self.classifier_path}")
            self.classifier = YOLO(str(self.classifier_path))
            print("[Pipeline] Severity classification model loaded successfully!")
            self.classifier_missing = False
        except Exception as e:
            print(f"[Pipeline] WARNING: Classification model could not be loaded: {e}. Accident detection remains active.")
            self.classifier = None
            self.classifier_missing = True

    def detect_accident(self, frame, conf_threshold):
        if self.detector is None:
            raise RuntimeError("Detection model is not loaded.")
        results = self.detector(frame, conf=conf_threshold, verbose=False)
        accident_detected = detect_accident_from_collection(results)
        
        max_conf = 0.0
        if results and len(results) > 0:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None and len(boxes) > 0:
                max_conf = float(boxes.conf.max().item())
                
        return results, accident_detected, max_conf

    def classify_severity(self, frame_or_crop):
        if self.classifier_missing or self.classifier is None:
            return "Model not found", 0.0
            
        try:
            if frame_or_crop is None or frame_or_crop.size == 0 or frame_or_crop.shape[0] < 5 or frame_or_crop.shape[1] < 5:
                raise ValueError("Crop is invalid or too small for classification.")
                
            results = self.classifier(frame_or_crop, verbose=False)
            if results and len(results) > 0:
                probs = results[0].probs
                pred_idx = int(probs.top1)
                confidence = float(probs.top1conf.item())
                
                class_names = results[0].names
                raw_label = class_names[pred_idx]
                return raw_label, confidence
            else:
                raise ValueError("Classifier returned empty results.")
        except Exception as crop_error:
            print(f"[Pipeline] Classification on crop failed: {crop_error}.")
            return None, 0.0

    def process_image(self, image_path_or_array, conf_threshold):
        """Sequential inference for image uploads."""
        results, accident_detected, max_conf = self.detect_accident(image_path_or_array, conf_threshold)
        
        severity_label = "Not Applied"
        severity_conf = 0.0
        annotated_image = results[0].plot() if results else image_path_or_array
        
        if accident_detected:
            best_crop = None
            highest_conf = 0.0
            
            if results and len(results) > 0:
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf[0].item())
                        if conf > highest_conf:
                            highest_conf = conf
                            try:
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                orig_img = results[0].orig_img
                                if orig_img is not None:
                                    best_crop = orig_img[y1:y2, x1:x2]
                            except Exception as e:
                                print(f"[Pipeline] Error cropping detection: {e}")
            
            # Classify crop
            label = None
            conf_val = 0.0
            if best_crop is not None:
                label, conf_val = self.classify_severity(best_crop)
                
            # Fallback to full frame
            if label is None:
                print("[Pipeline] Falling back to classifying full frame...")
                label, conf_val = self.classify_severity(image_path_or_array)
                
            severity_label = label if label else "Classification Failed"
            severity_conf = conf_val
            
        return {
            "accident_detected": accident_detected,
            "detection_confidence": max_conf,
            "severity_label": severity_label,
            "severity_confidence": severity_conf,
            "annotated_image": annotated_image,
            "num_detections": len(results[0].boxes) if (results and getattr(results[0], "boxes", None) is not None) else 0
        }


# Initialize Unified Pipeline Globally
pipeline = AccidentSeverityPipeline()
try:
    pipeline.load_models()
    model = pipeline.detector
    BEST_MODEL_PATH = pipeline.detector_path
except Exception as init_err:
    print(f"CRITICAL: Failed to load pipeline models: {init_err}")
    model = None
    BEST_MODEL_PATH = None



def build_status_banner(title: str, message: str, tone: str, icon: str) -> str:
    safe_title = escape(title)
    safe_message = escape(message)
    return f"""
    <div class="status-banner tone-{tone}" role="status" aria-live="polite">
        <div class="status-icon">{icon}</div>
        <div class="status-copy">
            <p class="status-label">{safe_title}</p>
            <p class="status-message">{safe_message}</p>
        </div>
    </div>
    """


def build_model_badge(model_path: Path | None) -> str:
    resolved_path = str(model_path) if model_path else "No compatible model found"
    return f"""
    <div class="model-badge">
        <span class="model-badge-label">Loaded Model</span>
        <code class="model-badge-path">{escape(resolved_path)}</code>
    </div>
    """


def build_threshold_readout(conf_threshold: float) -> str:
    return f"""
    <div class="threshold-head">
        <span class="threshold-label">Confidence Threshold</span>
        <span class="threshold-value">{conf_threshold:.2f}</span>
    </div>
    """


def detect_accident_from_result(result) -> bool:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False

    names = getattr(result, "names", {}) or {}
    for cls_idx in boxes.cls:
        class_id = int(cls_idx)
        class_name = ""
        try:
            if isinstance(names, dict):
                class_name = str(names.get(class_id, ""))
            else:
                class_name = str(names[class_id])
        except Exception:
            class_name = ""

        if "accident" in class_name.lower():
            return True

    for cls_idx in boxes.cls:
        if int(cls_idx) == 0:
            return True

    return len(boxes) > 0


def detect_accident_from_collection(results) -> bool:
    if results is None:
        return False
    return any(detect_accident_from_result(result) for result in results)


def build_alert_banner(state: str = "standby", source: str = "image") -> str:
    return ""


def build_alert_controls() -> str:
    return ""


def build_alert_signal(alert_active: bool, status: str, source: str) -> str:
    import uuid
    alert_id = str(uuid.uuid4())
    title = "Accident Detected" if alert_active else "Alert Standby"
    message = "The uploaded media was classified as an accident." if alert_active else ""
    return f"""
    <div
        class="alert-signal-data"
        data-alert-id="{alert_id}"
        data-alert-active="{str(alert_active).lower()}"
        data-alert-status="{escape(status)}"
        data-alert-source="{escape(source)}"
        data-alert-title="{escape(title)}"
        data-alert-message="{escape(message)}"
    ></div>
    """


def build_alert_controller_head() -> str:
    return """
    <script>
    (() => {
      if (window.__accidentAlertControllerLoaded) return;
      window.__accidentAlertControllerLoaded = true;

      const controller = {
        state: {
          id: "",
          active: false,
          status: "idle",
          source: "image",
          title: "Accident Detected",
          message: "The uploaded media was classified as an accident.",
        },
        audioSupported: Boolean(window.AudioContext || window.webkitAudioContext),
        notificationSupported: typeof window.Notification !== "undefined",
        soundEnabled: true,
        hasUserInteracted: false,
        audioContext: null,
        alarmTimer: null,
        alarmTimeout: null,
        signalObserver: null,
        signalObserverHost: null,
        attachTimer: null,
        handlersBound: false,
        lastNotificationKey: "",

        init() {
          this.bindGlobalHandlers();
          this.attachSignalObserver();
          this.syncFromSignal();
        },

        enableSoundAndNotifications() {
          this.soundEnabled = true;
          this.registerInteraction();
          this.ensureAudioContext();

          if (this.notificationSupported && Notification.permission === "default") {
            Notification.requestPermission()
              .then((permission) => {
                if (permission === "granted") {
                  this.lastNotificationKey = "";
                  this.notify();
                }
              })
              .catch(() => {});
          }
        },

        bindGlobalHandlers() {
          if (this.handlersBound) return;
          this.handlersBound = true;

          document.addEventListener("pointerdown", () => {
            this.registerInteraction();
          }, true);

          document.addEventListener("keydown", () => {
            this.registerInteraction();
          }, true);

          document.addEventListener("click", (event) => {
            const button = event.target.closest("button");
            if (!button) return;

            if (button.id === "run-image-inference-btn" || button.id === "run-video-inference-btn") {
              this.enableSoundAndNotifications();
            }
          }, true);

          window.addEventListener("beforeunload", () => this.destroy(), { once: true });
          window.addEventListener("pagehide", () => this.destroy(), { once: true });
        },

        registerInteraction() {
          this.hasUserInteracted = true;
          if (this.soundEnabled) {
            this.ensureAudioContext();
          }
        },

        ensureAudioContext() {
          if (!this.audioSupported) return null;
          const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
          if (!AudioContextCtor) return null;

          try {
            if (!this.audioContext) {
              this.audioContext = new AudioContextCtor();
            }
            if (this.audioContext.state === "suspended") {
              this.audioContext.resume().catch(() => {});
            }
            return this.audioContext;
          } catch (error) {
            return null;
          }
        },

        attachSignalObserver() {
          const host = document.getElementById("alert-signal-region");
          if (!host) {
            window.clearTimeout(this.attachTimer);
            this.attachTimer = window.setTimeout(() => this.attachSignalObserver(), 250);
            return;
          }

          if (this.signalObserverHost === host) return;

          if (this.signalObserver) {
            this.signalObserver.disconnect();
          }

          this.signalObserverHost = host;
          this.signalObserver = new MutationObserver(() => this.syncFromSignal());
          this.signalObserver.observe(host, {
            subtree: true,
            childList: true,
            characterData: true,
            attributes: true,
          });
        },

        readSignalPayload() {
          const host = document.getElementById("alert-signal-region");
          const node = host ? host.querySelector(".alert-signal-data") : null;
          if (!node) {
            return {
              id: "",
              active: false,
              status: "idle",
              source: "image",
              title: "Accident Detected",
              message: "The uploaded media was classified as an accident.",
            };
          }

          return {
            id: node.dataset.alertId || "",
            active: node.dataset.alertActive === "true",
            status: node.dataset.alertStatus || "idle",
            source: node.dataset.alertSource || "image",
            title: node.dataset.alertTitle || "Accident Detected",
            message: node.dataset.alertMessage || "The uploaded media was classified as an accident.",
          };
        },

        syncFromSignal() {
          const next = this.readSignalPayload();
          const wasActive = this.state.active;
          const previousStatus = this.state.status;
          const previousId = this.state.id;
          this.state = next;

          if (!next.active) {
            if (wasActive || previousStatus !== next.status || previousId !== next.id) {
              this.resetRuntime();
            } else {
              this.applyClasses();
            }
            return;
          }

          if (!wasActive || previousId !== next.id) {
            this.stopAlarm();
            this.activateAlert();
            return;
          }

          this.applyClasses();
        },

        activateAlert() {
          this.applyClasses();
          this.triggerVibration();

          if (this.soundEnabled && this.hasUserInteracted) {
            this.startAlarm();
          }

          this.notify();
        },

        resetRuntime() {
          this.stopAlarm();
          this.stopVibration();
          this.lastNotificationKey = "";
          this.applyClasses();
        },

        startAlarm() {
          if (!this.state.active || !this.soundEnabled || !this.hasUserInteracted) return;
          if (!this.ensureAudioContext()) return;
          if (this.alarmTimer) return;

          this.playAlarmPattern();
          this.alarmTimer = window.setInterval(() => {
            this.playAlarmPattern();
          }, 1900);

          if (this.alarmTimeout) {
            window.clearTimeout(this.alarmTimeout);
          }
          this.alarmTimeout = window.setTimeout(() => {
            this.stopAlarm();
          }, 10000);
        },

        stopAlarm() {
          if (this.alarmTimer) {
            window.clearInterval(this.alarmTimer);
            this.alarmTimer = null;
          }
          if (this.alarmTimeout) {
            window.clearTimeout(this.alarmTimeout);
            this.alarmTimeout = null;
          }
        },

        playAlarmPattern() {
          const ctx = this.ensureAudioContext();
          if (!ctx) return;

          const pulses = [
            { at: 0.0, freq: 740, duration: 0.16 },
            { at: 0.26, freq: 620, duration: 0.16 },
            { at: 0.56, freq: 760, duration: 0.24 },
          ];

          pulses.forEach((pulse) => {
            try {
              const oscillator = ctx.createOscillator();
              const gainNode = ctx.createGain();
              const startAt = ctx.currentTime + pulse.at;
              const stopAt = startAt + pulse.duration;

              oscillator.type = "square";
              oscillator.frequency.setValueAtTime(pulse.freq, startAt);
              gainNode.gain.setValueAtTime(0.0001, startAt);
              gainNode.gain.exponentialRampToValueAtTime(0.06, startAt + 0.02);
              gainNode.gain.exponentialRampToValueAtTime(0.0001, stopAt);

              oscillator.connect(gainNode);
              gainNode.connect(ctx.destination);
              oscillator.start(startAt);
              oscillator.stop(stopAt + 0.03);
              oscillator.onended = () => {
                try {
                  oscillator.disconnect();
                  gainNode.disconnect();
                } catch (error) {
                }
              };
            } catch (error) {
            }
          });
        },

        triggerVibration() {
          if (typeof navigator !== "undefined" && typeof navigator.vibrate === "function") {
            try {
              navigator.vibrate([300, 150, 300, 150, 500]);
            } catch (error) {
            }
          }
        },

        stopVibration() {
          if (typeof navigator !== "undefined" && typeof navigator.vibrate === "function") {
            try {
              navigator.vibrate(0);
            } catch (error) {
            }
          }
        },

        notify() {
          if (!this.notificationSupported) return;
          if (Notification.permission !== "granted") return;

          const notificationKey = `${this.state.source}:${this.state.status}`;
          if (this.lastNotificationKey === notificationKey) return;
          this.lastNotificationKey = notificationKey;

          try {
            new Notification(this.state.title || "Accident Detected", {
              body: this.state.message || "The uploaded media was classified as an accident.",
              tag: "capstone-accident-alert",
              renotify: true,
            });
          } catch (error) {
          }
        },

        enableBrowserAlerts() {
          this.registerInteraction();
          if (!this.notificationSupported) {
            return;
          }

          if (Notification.permission === "granted") {
            this.notify();
            return;
          }

          Notification.requestPermission()
            .then((permission) => {
              if (permission === "granted") {
                this.lastNotificationKey = "";
                this.notify();
              }
            })
            .catch(() => {});
        },

        applyClasses() {
          const body = document.body;
          if (!body) return;

          body.classList.toggle("accident-alert-active", this.state.active);
          body.classList.toggle("accident-alert-source-image", this.state.active && this.state.source === "image");
          body.classList.toggle("accident-alert-source-video", this.state.active && this.state.source === "video");
        },

        setPill(id, text, tone) {
          const pill = document.getElementById(id);
          if (!pill) return;
          pill.textContent = text;
          pill.className = `alert-pill ${tone}`;
        },

        destroy() {
          this.stopAlarm();
          this.stopVibration();
          if (this.signalObserver) {
            this.signalObserver.disconnect();
          }
          window.clearTimeout(this.attachTimer);
          try {
            if (this.audioContext && typeof this.audioContext.close === "function" && this.audioContext.state !== "closed") {
              this.audioContext.close().catch(() => {});
            }
          } catch (error) {
            console.warn("AudioContext close failed:", error);
          }
        },
      };

      window.__accidentAlertController = controller;

      const boot = () => controller.init();
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
      } else {
        boot();
      }
    })();
    </script>
    """
def predict_accident_gui(input_image: np.ndarray, conf_threshold: float):
    if pipeline.detector is None:
        return input_image, build_status_banner(
            title="Model unavailable",
            message="No object detection model was loaded. Please check the configured model paths.",
            tone="alert",
            icon="&#10005;",
        ), build_alert_banner("error", "image"), build_alert_signal(False, "error", "image")

    if input_image is None:
        return None, build_status_banner(
            title="Image required",
            message="Upload a road image to run accident detection inference.",
            tone="neutral",
            icon="&#8682;",
        ), build_alert_banner("standby", "image"), build_alert_signal(False, "idle", "image")

    try:
        # Run sequential pipeline
        res = pipeline.process_image(input_image, conf_threshold)
        
        accident_detected = res["accident_detected"]
        det_conf = res["detection_confidence"]
        sev_label = res["severity_label"]
        sev_conf = res["severity_confidence"]
        annotated_image = res["annotated_image"]
        num_detections = res["num_detections"]
        
        status_html = build_pipeline_status_banner(
            accident_detected=accident_detected,
            detection_conf=det_conf,
            severity_label=sev_label,
            severity_conf=sev_conf,
            num_detections=num_detections,
            source="image"
        )
        
        pipeline.last_det_conf = det_conf
        pipeline.last_sev_label = sev_label
        pipeline.last_sev_conf = sev_conf

        if accident_detected:
            alert_html = build_alert_banner("active", "image")
            alert_signal = build_alert_signal(True, "accident", "image")
        else:
            alert_html = build_alert_banner("clear", "image")
            alert_signal = build_alert_signal(False, "clear", "image")
            
        return annotated_image, status_html, alert_html, alert_signal
        
    except Exception as e:
        print(f"[IMAGE] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return input_image, build_status_banner(
            title="Inference failed",
            message=f"Error occurred during image inference: {str(e)}",
            tone="alert",
            icon="&#10005;",
        ), build_alert_banner("error", "image"), build_alert_signal(False, "error", "image")


def handle_image_upload(image: np.ndarray | None):
    if image is None:
        return (
            build_status_banner(
                title="No image uploaded",
                message="Please upload a valid road image.",
                tone="neutral",
                icon="&#8682;",
            ),
            None,
            build_alert_banner("standby", "image"),
            build_alert_signal(False, "idle", "image"),
            render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", "")
        )
    return (
        build_status_banner(
            title="Image uploaded successfully",
            message="Review the preview and click Run Inference to detect accidents.",
            tone="safe",
            icon="&#10003;",
        ),
        None,
        build_alert_banner("armed", "image"),
        build_alert_signal(False, "armed", "image"),
        render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
    )


def handle_video_upload(video_path: str | None):
    if not video_path:
        return (
            build_status_banner(
                title="No video uploaded",
                message="Please upload a road video for testing.",
                tone="neutral",
                icon="&#8682;",
            ),
            None,
            gr.update(visible=True),
            build_alert_banner("standby", "video"),
            build_alert_signal(False, "idle", "video"),
            render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", "")
        )
    
    # Validate file extension
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(video_path).suffix.lower()
    if ext not in allowed_exts:
        return (
            build_status_banner(
                title="Unsupported video format",
                message=f"The uploaded format {ext} is not supported. Please use MP4, AVI, MOV, or MKV.",
                tone="alert",
                icon="&#10005;",
            ),
            None,
            gr.update(visible=True),
            build_alert_banner("error", "video"),
            build_alert_signal(False, "error", "video"),
            render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", "")
        )
    
    return (
        build_status_banner(
            title="Video uploaded successfully",
            message="Review the preview and click Run Video Inference to process.",
            tone="safe",
            icon="&#10003;",
        ),
        None,
        gr.update(visible=True),
        build_alert_banner("armed", "video"),
        build_alert_signal(False, "armed", "video"),
        render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
    )


def handle_image_clear():
    return (
        build_status_banner(
            title="No image uploaded",
            message="Please upload a valid road image.",
            tone="neutral",
            icon="&#8682;",
        ),
        None,
        build_alert_banner("standby", "image"),
        build_alert_signal(False, "idle", "image"),
        render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", "")
    )


def handle_video_clear():
    return (
        build_status_banner(
            title="No video uploaded",
            message="Please upload a road video for testing.",
            tone="neutral",
            icon="&#8682;",
        ),
        None,
        gr.update(visible=True),
        build_alert_banner("standby", "video"),
        build_alert_signal(False, "idle", "video"),
        render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", "")
    )


def run_image_inference(input_image: np.ndarray, conf_threshold: float, chat_history, agent):
    """
    Unified image pipeline + AI agent generator flow.
    """
    import cv2
    
    # Step 1: Run YOLO pipeline
    try:
        annotated_image, status_html, alert_html, alert_signal = predict_accident_gui(input_image, conf_threshold)
    except Exception as e:
        print(f"[Image Inference] YOLO pipeline failed: {e}")
        chat_history = chat_history or []
        yield (
            input_image,
            build_status_banner("Inference Failed", f"YOLO error: {e}", "alert", "&#10005;"),
            build_alert_banner("error", "image"),
            build_alert_signal(False, "error", "image"),
            render_chat_html(chat_history),
            gr.update(interactive=False, placeholder="Inference failed."),
            gr.update(interactive=False, value="Locked"),
            agent,
            chat_history,
            render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
        )
        return

    # Check if accident was detected
    accident_detected = 'data-alert-active="true"' in alert_signal

    # If no accident is detected
    if not accident_detected:
        chat_history = chat_history or []
        yield (
            annotated_image,
            status_html,
            alert_html,
            alert_signal,
            render_chat_html(chat_history),
            gr.update(interactive=True, placeholder="Ask the legal assistant general questions..."),
            gr.update(interactive=True, value="Send"),
            agent,
            chat_history,
            render_final_summary_html(False, pipeline.last_det_conf, "Not Applied", 0.0, "Not Escalated", "")
        )
        return

    # If accident is detected, we perform AI analysis sequentially
    chat_history = chat_history or []
    yield (
        annotated_image,
        build_status_banner(
            title="Analyzing accident with AI agent...",
            message="Accident detected! Sending scene metadata and crop to Traffic Liability Engine...",
            tone="neutral",
            icon="&#9711;",
        ),
        alert_html,
        alert_signal,
        render_chat_html(chat_history),
        gr.update(interactive=False, placeholder="Analyzing accident with AI agent..."),
        gr.update(interactive=False, value="Analyzing..."),
        agent,
        chat_history,
        render_final_summary_html(True, pipeline.last_det_conf, pipeline.last_sev_label, pipeline.last_sev_conf, None, "AI liability analysis in progress...")
    )

    # Step 2: Initialize Agent safely
    if not agent:
        try:
            print("[Agent] Initializing AccidentAgent...")
            agent = get_accident_agent()
        except Exception as e:
            print(f"[Agent] Initialization failed: {e}")
            chat_history = chat_history or []
            chat_history.append(("Assistant", f"❌ Failed to load AI Agent folder or configs: {str(e)}"))
            yield (
                annotated_image,
                status_html,
                alert_html,
                alert_signal,
                render_chat_html(chat_history),
                gr.update(interactive=False, placeholder="Agent offline. Config missing."),
                gr.update(interactive=False, value="Locked"),
                agent,
                chat_history,
                render_final_summary_html(True, pipeline.last_det_conf, pipeline.last_sev_label, pipeline.last_sev_conf, None, f"Agent offline. Error: {str(e)}")
            )
            return

    # Save current frame to temporary file for Vision Model analysis
    temp_path = os.path.join(str(PROJECT_ROOT), "accident_agent", "temp_input.jpg")
    try:
        bgr_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_path, bgr_img)
    except Exception as e:
        print(f"[Agent] Failed to save temporary image frame: {e}")
        temp_path = os.path.join(str(PROJECT_ROOT), "accident_agent", "temp_input.jpg")

    # Step 3: Run AI agent analysis using safe API wrapper
    try:
        print(f"[Agent] Calling generate_initial_analysis on: {temp_path}")
        analysis_report = safe_agent_call(agent, "generate_initial_analysis", [temp_path])
        
        chat_history = chat_history or []
        chat_history.append(("Assistant", analysis_report))
        
        yield (
            annotated_image,
            status_html,
            alert_html,
            alert_signal,
            render_chat_html(chat_history),
            gr.update(interactive=True, placeholder="Ask the legal assistant..."),
            gr.update(interactive=True, value="Send"),
            agent,
            chat_history,
            render_final_summary_html(True, pipeline.last_det_conf, pipeline.last_sev_label, pipeline.last_sev_conf, None, analysis_report)
        )
    except Exception as e:
        print(f"[Agent] Analysis invocation failed: {e}")
        chat_history = chat_history or []
        chat_history.append(("Assistant", f"❌ Agent analysis failed: {str(e)}\n\nCheck terminal logs for debugging details."))
        yield (
            annotated_image,
            status_html,
            alert_html,
            alert_signal,
            render_chat_html(chat_history),
            gr.update(interactive=True, placeholder="Ask the legal assistant..."),
            gr.update(interactive=True, value="Send"),
            agent,
            chat_history,
            render_final_summary_html(True, pipeline.last_det_conf, pipeline.last_sev_label, pipeline.last_sev_conf, None, f"Analysis failed: {str(e)}")
        )


def convert_avi_to_mp4(input_path: str) -> str | None:
    """
    Converts an AVI video file to MP4 format using OpenCV.
    """
    from pathlib import Path
    import cv2

    input_file = Path(input_path)
    if not input_file.exists():
        return None

    output_file = input_file.with_suffix(".mp4")

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()

    if output_file.exists() and output_file.stat().st_size > 0:
        return str(output_file)

    return None


def run_video_inference(video_path: str | None, confidence_threshold: float, chat_history, agent):
    """
    Handles video inference input from the UI.
    This function should not train or modify models.
    It passes the uploaded video to the pipeline detector and runs
    sequential severity classification on the highest-confidence crop.
    It then invokes the AI accident agent sequentially.
    """
    import traceback
    import cv2

    print("[VIDEO] run_video_inference started")
    print(f"[VIDEO] received video_path: {video_path}")
    print(f"[VIDEO] confidence_threshold: {confidence_threshold}")

    if pipeline.detector is None:
        print("[VIDEO] detector model is None")
        chat_history = chat_history or []
        yield (
            None,
            build_status_banner(
                title="Model unavailable",
                message="No object detection model was loaded. Please check the configured paths.",
                tone="alert",
                icon="&#10005;",
            ),
            gr.update(visible=True),
            build_alert_banner("error", "video"),
            build_alert_signal(False, "error", "video"),
            render_chat_html(chat_history),
            gr.update(interactive=False, placeholder="Model unavailable."),
            gr.update(interactive=False, value="Locked"),
            agent,
            chat_history,
            render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
        )
        return

    if not video_path:
        print("[VIDEO] no video_path provided")
        chat_history = chat_history or []
        yield (
            None,
            build_status_banner(
                title="No video uploaded",
                message="Upload a road video to run accident detection inference.",
                tone="neutral",
                icon="&#8682;",
            ),
            gr.update(visible=True),
            build_alert_banner("standby", "video"),
            build_alert_signal(False, "idle", "video"),
            render_chat_html(chat_history),
            gr.update(interactive=False, placeholder="Upload video first."),
            gr.update(interactive=False, value="Locked"),
            agent,
            chat_history,
            render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
        )
        return

    video_file = Path(video_path)
    print(f"[VIDEO] video_path exists: {video_file.exists()}")
    print(f"[VIDEO] video_path suffix: {video_file.suffix}")

    # Validate video format
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if video_file.suffix.lower() not in allowed_exts:
        print(f"[VIDEO] unsupported format: {video_file.suffix}")
        chat_history = chat_history or []
        yield (
            None,
            build_status_banner(
                title="Unsupported video format",
                message=f"The uploaded format {video_file.suffix} is not supported. Please use MP4, AVI, MOV, or MKV.",
                tone="alert",
                icon="&#10005;",
            ),
            gr.update(visible=True),
            build_alert_banner("error", "video"),
            build_alert_signal(False, "error", "video"),
            render_chat_html(chat_history),
            gr.update(interactive=False, placeholder="Unsupported video."),
            gr.update(interactive=False, value="Locked"),
            agent,
            chat_history,
            render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
        )
        return

    # Yield processing status
    chat_history = chat_history or []
    yield (
        None,
        build_status_banner(
            title="Processing video...",
            message="Processing video frame by frame. This may take time depending on video length and device. Please wait...",
            tone="neutral",
            icon="&#9711;",
        ),
        gr.update(visible=True),
        build_alert_banner("processing", "video"),
        build_alert_signal(False, "processing", "video"),
        render_chat_html(chat_history),
        gr.update(interactive=False, placeholder="Processing video... Chatbot locked."),
        gr.update(interactive=False, value="Analyzing..."),
        agent,
        chat_history,
        render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
    )

    try:
        run_id = str(int(time.time()))
        project_dir = PROJECT_ROOT / "runs" / "detect"
        name_dir = f"video_predictions_{run_id}"
        out_dir = project_dir / name_dir

        print(f"[VIDEO] output directory path: {out_dir}")
        print("[VIDEO] calling pipeline.detector.predict() now...")

        # Run inference using the natively supported YOLO predict function with stride
        results = pipeline.detector.predict(
            source=video_path,
            conf=confidence_threshold,
            save=True,
            project=str(project_dir),
            name=name_dir,
            exist_ok=True,
            imgsz=640,
            vid_stride=3,
        )

        print("[VIDEO] pipeline.detector.predict() finished")
        
        # Scan prediction results for accident frames
        highest_det_conf = 0.0
        best_crop = None
        best_frame = None
        number_of_detected_frames = 0

        print("[VIDEO] Scanning prediction results for accident frames...")
        for res_frame in results:
            if detect_accident_from_result(res_frame):
                number_of_detected_frames += 1
                boxes = getattr(res_frame, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf[0].item())
                        if conf > highest_det_conf:
                            highest_det_conf = conf
                            try:
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                orig_img = res_frame.orig_img
                                if orig_img is not None:
                                    best_crop = orig_img[y1:y2, x1:x2]
                                    best_frame = orig_img
                            except Exception as e:
                                print(f"[VIDEO] Cropping failed: {e}")

        accident_detected = (number_of_detected_frames > 0)
        severity_label = "Not Applied"
        severity_conf = 0.0
        
        if accident_detected:
            label = None
            conf_val = 0.0
            if best_crop is not None:
                print(f"[VIDEO] Accident detected! Running severity classification on crop from highest confidence frame (conf={highest_det_conf:.4f})...")
                label, conf_val = pipeline.classify_severity(best_crop)
                
            # Fallback to full frame
            if label is None:
                if best_frame is not None:
                    print("[VIDEO] Classification on crop failed. Falling back to full frame of best detection...")
                    label, conf_val = pipeline.classify_severity(best_frame)
                    
            severity_label = label if label else "Classification Failed"
            severity_conf = conf_val
            print(f"[VIDEO] Severity Classification result: {severity_label} ({severity_conf:.4f})")
        else:
            print("[VIDEO] No accident detected in video.")

        # Locate the saved video file recursively using rglob
        output_video_path = None
        if out_dir.exists():
            print("[VIDEO] listing all files found recursively inside the output folder:")
            for file in out_dir.rglob("*"):
                if file.is_file():
                    print(f"  - {file}")
                    if file.suffix.lower() in allowed_exts and not output_video_path:
                        output_video_path = str(file)

        print(f"[VIDEO] output_video_path found: {output_video_path}")
        preview_path = None
        conversion_failed = False

        if output_video_path and Path(output_video_path).exists():
            output_file = Path(output_video_path)
            if output_file.suffix.lower() == ".mp4":
                preview_path = str(output_file)
            elif output_file.suffix.lower() == ".avi":
                print(f"[VIDEO] Found AVI output. Converting to MP4: {output_video_path}")
                converted_path = convert_avi_to_mp4(str(output_file))
                if converted_path:
                    print(f"[VIDEO] Conversion successful: {converted_path}")
                    preview_path = converted_path
                else:
                    print("[VIDEO] Conversion failed")
                    conversion_failed = True
            else:
                preview_path = str(output_file)

        print(f"[VIDEO] final preview path returned to Gradio: {preview_path}")

        # Build beautifully styled dashboard status banner for video
        status_html = build_pipeline_status_banner(
            accident_detected=accident_detected,
            detection_conf=highest_det_conf,
            severity_label=severity_label,
            severity_conf=severity_conf,
            num_detections=number_of_detected_frames,
            source="video",
            processed_video_path=preview_path if preview_path else output_video_path
        )

        alert_banner = build_alert_banner("active" if accident_detected else "clear", "video")
        alert_sig = build_alert_signal(accident_detected, "accident" if accident_detected else "clear", "video")

        if accident_detected:
            chat_history = chat_history or []
            # Yield intermediate results first, then run sequential KAG analysis
            yield (
                preview_path if preview_path else None,
                build_status_banner(
                    title="Analyzing accident with AI agent...",
                    message="Accident detected! Extracting crash frames and initiating liability reasoning...",
                    tone="neutral",
                    icon="&#9711;",
                ),
                gr.update(visible=False) if preview_path else gr.update(visible=True),
                alert_banner,
                alert_sig,
                render_chat_html(chat_history),
                gr.update(interactive=False, placeholder="Analyzing accident with AI agent..."),
                gr.update(interactive=False, value="Analyzing..."),
                agent,
                chat_history,
                render_final_summary_html(True, highest_det_conf, severity_label, severity_conf, None, "AI liability analysis in progress...")
            )

            # Initialize Agent
            if not agent:
                try:
                    print("[Video Agent] Initializing AccidentAgent...")
                    agent = get_accident_agent()
                except Exception as e:
                    print(f"[Video Agent] Initialization failed: {e}")
                    chat_history = chat_history or []
                    chat_history.append(("Accident Analysis Trigger", f"❌ Failed to load AI Agent folder or configs: {str(e)}"))
                    yield (
                        preview_path if preview_path else None,
                        status_html,
                        gr.update(visible=False) if preview_path else gr.update(visible=True),
                        alert_banner,
                        alert_sig,
                        render_chat_html(chat_history),
                        gr.update(interactive=False, placeholder="Agent offline. Config missing."),
                        gr.update(interactive=False, value="Locked"),
                        agent,
                        chat_history,
                        render_final_summary_html(True, highest_det_conf, severity_label, severity_conf, None, f"Agent offline. Error: {str(e)}")
                    )
                    return

            # Save keyframes (best frame and best crop)
            temp_paths = []
            if best_frame is not None:
                tf = os.path.join(str(PROJECT_ROOT), "accident_agent", "temp_video_frame.jpg")
                try:
                    cv2.imwrite(tf, cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR))
                    temp_paths.append(tf)
                except Exception as e:
                    print(f"[Video Agent] Error saving frame: {e}")
            if best_crop is not None:
                tc = os.path.join(str(PROJECT_ROOT), "accident_agent", "temp_video_crop.jpg")
                try:
                    cv2.imwrite(tc, cv2.cvtColor(best_crop, cv2.COLOR_RGB2BGR))
                    temp_paths.append(tc)
                except Exception as e:
                    print(f"[Video Agent] Error saving crop: {e}")

            # Invoke agent visual KAG report
            try:
                print(f"[Video Agent] Calling generate_initial_analysis on: {temp_paths}")
                analysis_report = safe_agent_call(agent, "generate_initial_analysis", temp_paths)
                
                chat_history = chat_history or []
                chat_history.append(("Auto Accident Analysis", analysis_report))
                
                yield (
                    preview_path if preview_path else None,
                    status_html,
                    gr.update(visible=False) if preview_path else gr.update(visible=True),
                    alert_banner,
                    alert_sig,
                    render_chat_html(chat_history),
                    gr.update(interactive=True, placeholder="Ask the legal assistant..."),
                    gr.update(interactive=True, value="Send"),
                    agent,
                    chat_history,
                    render_final_summary_html(True, highest_det_conf, severity_label, severity_conf, None, analysis_report)
                )
            except Exception as e:
                print(f"[Video Agent] KAG analysis failed: {e}")
                chat_history = chat_history or []
                chat_history.append(("Auto Accident Analysis", f"❌ Video analysis failed: {str(e)}\n\nCheck terminal logs for traceback."))
                yield (
                    preview_path if preview_path else None,
                    status_html,
                    gr.update(visible=False) if preview_path else gr.update(visible=True),
                    alert_banner,
                    alert_sig,
                    render_chat_html(chat_history),
                    gr.update(interactive=True, placeholder="Ask the legal assistant..."),
                    gr.update(interactive=True, value="Send"),
                    agent,
                    chat_history,
                    render_final_summary_html(True, highest_det_conf, severity_label, severity_conf, None, f"Analysis failed: {str(e)}")
                )
        else:
            chat_history = chat_history or []
            yield (
                preview_path if preview_path else None,
                status_html,
                gr.update(visible=False) if preview_path else gr.update(visible=True),
                alert_banner,
                alert_sig,
                render_chat_html(chat_history),
                gr.update(interactive=True, placeholder="Ask the legal assistant general questions..."),
                gr.update(interactive=True, value="Send"),
                agent,
                chat_history,
                render_final_summary_html(False, highest_det_conf, "Not Applied", 0.0, "Not Escalated", "")
            )

    except Exception as e:
        print(f"[VIDEO] ERROR: {str(e)}")
        traceback.print_exc()
        chat_history = chat_history or []
        yield (
            None,
            build_status_banner(
                title="Video processing failed",
                message=f"Error: {str(e)}. Check the terminal logs for full traceback.",
                tone="alert",
                icon="&#10005;",
            ),
            gr.update(visible=True),
            build_alert_banner("error", "video"),
            build_alert_signal(False, "error", "video"),
            render_chat_html(chat_history),
            gr.update(interactive=False, placeholder="Processing error."),
            gr.update(interactive=False, value="Locked"),
            agent,
            chat_history,
            render_final_summary_html(False, 0.0, None, 0.0, "Not Escalated", "")
        )


def render_chat_html(chat_history):
    if not chat_history:
        return """
        <div style="text-align: center; color: var(--text-muted); padding: 40px 20px; font-size: 0.98rem; line-height: 1.6; background: rgba(248, 250, 252, 0.6); border: 1px dashed var(--card-border); border-radius: 20px; margin: 18px 0; min-height: 480px; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 10px;">
            <div style="font-size: 2.2rem;">&#129302;</div>
            <div style="font-weight: 800; color: var(--primary-dark); font-size: 1.15rem;">Accident Law Assistant Chat</div>
            <div style="max-width: 420px; opacity: 0.85;">
                Awaiting inference result. Upload a road scene or video and run detection to automatically initiate the AI Accident Liability Analysis.
            </div>
        </div>
        """
    
    html = '<div class="chat-preview" style="height: 480px; max-height: 480px; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; margin: 18px 0; background: rgba(248, 250, 252, 0.3); border-radius: 16px; border: 1px solid var(--card-border);">'
    for role, text in chat_history:
        if role == "User":
            html += f"""
            <div class="chat-bubble chat-bubble-user" style="align-self: flex-end; background: #2563eb !important; border: 1px solid #1e40af; color: #ffffff !important; max-width: 85%; padding: 12px 16px; border-radius: 18px 18px 2px 18px; font-size: 0.92rem; line-height: 1.5; text-align: left; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);">
                <span class="chat-role" style="display: block; margin-bottom: 4px; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: #dbeafe !important;">User</span>
                <div style="word-break: break-word; white-space: pre-wrap; color: #ffffff !important;">{escape(text)}</div>
            </div>
            """
        else:
            # Safely render the text and preserve formatting / line breaks
            formatted_text = escape(text).replace('\\n', '<br>').replace('\n', '<br>')
            # Support basic formatting like bold or bullet points if any
            # Format bold blocks safely
            parts = formatted_text.split('**')
            new_text = ""
            for idx, part in enumerate(parts):
                if idx % 2 == 1:
                    new_text += f"<b>{part}</b>"
                else:
                    new_text += part
            formatted_text = new_text
            
            # Replaces * bullet points with bullet symbol
            formatted_text = formatted_text.replace('<br>* ', '<br>&bull; ')
            formatted_text = formatted_text.replace('<br>- ', '<br>&bull; ')
            
            html += f"""
            <div class="chat-bubble chat-bubble-assistant" style="align-self: flex-start; background: #ffffff; border: 1px solid rgba(147, 197, 253, 0.45); color: #0f172a; max-width: 85%; padding: 12px 16px; border-radius: 18px 18px 18px 2px; font-size: 0.92rem; line-height: 1.5; box-shadow: 0 4px 12px rgba(29, 78, 216, 0.04); text-align: left;">
                <span class="chat-role" style="display: block; margin-bottom: 4px; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: #1e40af;">Assistant</span>
                <div style="word-break: break-word; white-space: pre-wrap; color: #0f172a !important;">{formatted_text}</div>
            </div>
            """
    html += '</div>'
    return html


def build_app():
    custom_css = """
    :root, .dark {
        --page-bg-top: #eef6ff;
        --page-bg-mid: #f8fbff;
        --page-bg-bottom: #dfefff;
        --card-bg: rgba(255, 255, 255, 0.97);
        --card-border: rgba(147, 197, 253, 0.42);
        --card-shadow: 0 22px 60px rgba(29, 78, 216, 0.10);
        --text-primary: #0f172a;
        --text-secondary: #1d4ed8;
        --text-muted: #64748b;
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --primary-soft: #dbeafe;
        --accent-cyan: #38bdf8;
        --success: #16a34a;
        --warning: #f59e0b;
        --danger: #dc2626;
        --alert-bg: #fef2f2;
        --alert-border: #fecaca;
        --alert-text: #991b1b;
        --safe-bg: #f0fdf4;
        --safe-border: #bbf7d0;
        --safe-text: #166534;
        --neutral-bg: #eff6ff;
        --neutral-border: #bfdbfe;
        --neutral-text: #1d4ed8;
        
        /* Force light mode variables globally in both light and dark client themes */
        --body-background-fill: #ffffff !important;
        --block-background-fill: #ffffff !important;
        --block-border-color: rgba(147, 197, 253, 0.42) !important;
        --block-title-text-color: #0f172a !important;
        --block-label-text-color: #1e40af !important;
        --input-background-fill: #ffffff !important;
        --input-border-color: rgba(147, 197, 253, 0.42) !important;
        --input-text-color: #0f172a !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f8fafc !important;
        --border-color-primary: rgba(147, 197, 253, 0.42) !important;
        --border-color-secondary: rgba(147, 197, 253, 0.3) !important;
        --panel-background-fill: #ffffff !important;
        --panel-border-color: rgba(147, 197, 253, 0.42) !important;
    }

    /* Force light theme elements on Gradio containers to override dark mode */
    .gradio-container,
    .gradio-container .block,
    .gradio-container .form,
    .gradio-container .fieldset,
    .gradio-container .compact,
    .gradio-container .padded,
    .gradio-container .panel,
    .gradio-container .box,
    .gradio-container input,
    .gradio-container select,
    .gradio-container textarea,
    .gradio-container .gr-input,
    .gradio-container .gr-box,
    .gradio-container .gr-form,
    .gradio-container .gr-block {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #0f172a !important;
        border-color: rgba(147, 197, 253, 0.42) !important;
    }

    body, body.dark {
        margin: 0;
        min-height: 100vh;
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 24%),
            linear-gradient(140deg, var(--page-bg-top) 0%, var(--page-bg-mid) 52%, var(--page-bg-bottom) 100%) !important;
        color: var(--text-primary) !important;
        font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    /* Full-width landscape container */
    .gradio-container {
        max-width: none !important;
        width: 100% !important;
        padding: 24px 32px !important;
        background: transparent !important;
    }

    #page-shell {
        gap: 26px;
    }

    .hero-card,
    .panel-card,
    .workflow-card,
    .workflow-strip-card,
    .footer-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 28px !important;
        box-shadow: var(--card-shadow) !important;
    }

    /* Strict Readability Safeguards */
    .panel-card,
    .workflow-strip-card,
    .threshold-card,
    .assistant-compose {
        color: #0f172a !important;
    }

    .panel-card *,
    .threshold-card *,
    .assistant-compose *,
    .workflow-strip-card * {
        --text-primary: #0f172a !important;
        --body-text-color: #0f172a !important;
        --block-title-text-color: #1e3a8a !important;
        --block-label-text-color: #1e3a8a !important;
        --input-text-color: #0f172a !important;
    }

    .panel-card p,
    .panel-card span,
    .panel-card h1,
    .panel-card h2,
    .panel-card h3,
    .panel-card h4,
    .panel-card div,
    .panel-card label,
    .threshold-card p,
    .threshold-card span,
    .threshold-card div,
    .assistant-compose div,
    .assistant-compose span,
    .workflow-strip-card div,
    .workflow-strip-card span {
        color: #0f172a;
    }

    .gradio-container textarea,
    .gradio-container input[type="text"] {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1px solid rgba(147, 197, 253, 0.42) !important;
    }

    .gradio-container textarea::placeholder,
    .gradio-container input[type="text"]::placeholder {
        color: #64748b !important;
        opacity: 0.8 !important;
    }

    .model-badge {
        margin-top: 14px;
        padding: 14px 18px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(147, 197, 253, 0.38);
        border-radius: 18px;
    }

    .model-badge-label {
        display: inline-block;
        margin-bottom: 4px;
        color: var(--primary-dark);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .model-badge-path {
        display: block;
        margin: 0;
        color: var(--text-primary);
        font-size: 0.9rem;
        font-family: monospace;
    }

    /* Select Input Mode label & pill buttons */
    .mode-selector {
        background: #ffffff !important;
        border: 1px solid rgba(147, 197, 253, 0.42) !important;
        border-radius: 18px !important;
        padding: 16px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.03) !important;
    }

    .mode-selector .block-label,
    .mode-selector span.block-label,
    .mode-selector span {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #2563eb !important; /* Medium blue text */
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        padding: 0 !important;
        margin-bottom: 12px !important;
        display: block !important;
        position: static !important;
        width: auto !important;
        height: auto !important;
    }

    .mode-selector .wrap,
    .mode-selector fieldset,
    .mode-selector .form {
        display: flex !important;
        flex-direction: row !important;
        gap: 12px !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .mode-selector label,
    .mode-selector label.gr-radio {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 8px 16px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        background: #ffffff !important;
        color: #475569 !important; /* Navy/gray */
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
    }

    .mode-selector label:hover,
    .mode-selector label.gr-radio:hover {
        border-color: rgba(37, 99, 235, 0.3) !important;
        background: #f0f7ff !important;
    }

    .mode-selector label.selected,
    .mode-selector label.gr-radio-selected,
    .mode-selector input[type="radio"]:checked + span,
    .mode-selector label:has(input[type="radio"]:checked) {
        background: #eff6ff !important; /* Very light blue */
        border: 1px solid #2563eb !important; /* Blue border */
        color: #1e3a8a !important; /* Navy text */
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.08) !important;
    }

    .mode-selector input[type="radio"] {
        display: none !important;
    }

    .content-grid {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 20px !important;
        width: 100% !important;
    }

    .panel-card {
        flex: 1 1 18% !important; /* Landscape row beside each other */
        min-width: 260px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        background: #ffffff !important;
        padding: 24px !important;
    }

    .panel-intro {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 16px;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        flex: 0 0 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent-cyan) 100%);
        color: #ffffff !important;
        font-size: 0.8rem;
        font-weight: 800;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.12);
    }

    .panel-copy {
        min-width: 0;
    }

    .panel-kicker {
        margin: 0 0 4px;
        color: var(--primary-dark) !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .panel-title {
        margin: 0;
        color: var(--text-primary) !important;
        font-size: 1.24rem;
        font-weight: 850;
    }

    .panel-description {
        margin: 8px 0 0;
        color: var(--text-muted) !important;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .image-shell {
        overflow: hidden;
        border: 1px solid var(--card-border) !important;
        border-radius: 16px !important;
        background: #f8fafc !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    #alert-signal-region {
        display: none;
    }

    .emergency-alert-card {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 16px;
        padding: 16px 18px;
        border-radius: 18px;
        border: 1px solid transparent;
    }

    .emergency-alert-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        flex: 0 0 36px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.78);
        font-size: 1rem;
        font-weight: 900;
    }

    .emergency-alert-copy {
        min-width: 0;
    }

    .emergency-alert-topline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
    }

    .emergency-alert-eyebrow {
        color: inherit;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .emergency-alert-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }

    .emergency-alert-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 26px;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .badge-danger {
        background: rgba(185, 28, 28, 0.12);
        border-color: rgba(185, 28, 28, 0.18);
        color: #991b1b !important;
    }

    .badge-clear {
        background: rgba(22, 163, 74, 0.10);
        border-color: rgba(22, 163, 74, 0.18);
        color: var(--safe-text) !important;
    }

    .badge-standby,
    .badge-source {
        background: rgba(37, 99, 235, 0.08);
        border-color: rgba(147, 197, 253, 0.42);
        color: var(--primary-dark) !important;
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.25);
        color: #b45309 !important;
    }

    .emergency-alert-title {
        margin: 8px 0 0;
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: -0.02em;
    }

    .emergency-alert-message {
        margin: 8px 0 0;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .state-active {
        background: #fee2e2 !important;
        border: 2px solid #ef4444 !important;
        color: #7f1d1d !important;
    }

    .state-active .emergency-alert-title,
    .state-active .emergency-alert-message,
    .state-active .emergency-alert-eyebrow {
        color: #7f1d1d !important;
    }

    .state-active .emergency-alert-icon {
        background: #fca5a5 !important;
        color: #7f1d1d !important;
    }



    .state-clear {
        background: #d1fae5 !important;
        border: 1px solid #10b981 !important;
        color: #064e3b !important;
    }
    .state-clear .emergency-alert-title,
    .state-clear .emergency-alert-message,
    .state-clear .emergency-alert-eyebrow {
        color: #064e3b !important;
    }
    .state-clear .emergency-alert-icon {
        background: #a7f3d0 !important;
        color: #064e3b !important;
    }

    .state-standby,
    .state-processing {
        background: #dbeafe !important;
        border: 1px solid #93c5fd !important;
        color: #1e3a8a !important;
    }
    .state-standby .emergency-alert-title,
    .state-standby .emergency-alert-message,
    .state-standby .emergency-alert-eyebrow,
    .state-processing .emergency-alert-title,
    .state-processing .emergency-alert-message,
    .state-processing .emergency-alert-eyebrow {
        color: #1e3a8a !important;
    }
    .state-standby .emergency-alert-icon,
    .state-processing .emergency-alert-icon {
        background: #bfdbfe !important;
        color: #1e3a8a !important;
    }

    .state-error {
        background: #ffedd5 !important;
        border: 1px solid #f97316 !important;
        color: #7c2d12 !important;
    }

    /* Clean White Confidence threshold card styling */
    .threshold-card {
        background: #ffffff !important;
        border: 1px solid rgba(147, 197, 253, 0.42) !important;
        border-radius: 18px !important;
        padding: 16px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.03) !important;
    }
    
    .threshold-card .block,
    .threshold-card .form {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    .threshold-slider {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .threshold-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 10px;
        margin-bottom: 8px;
        padding: 0 4px;
    }

    .threshold-label {
        color: #1e3a8a !important; /* Navy */
        font-size: 0.92rem !important;
        font-weight: 750 !important;
    }

    .threshold-value {
        color: #2563eb !important; /* Blue */
        font-size: 0.98rem !important;
        font-weight: 800 !important;
        background: rgba(37, 99, 235, 0.08);
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(147, 197, 253, 0.5);
    }

    .primary-action,
    .primary-action button {
        width: 100%;
        min-height: 48px;
        border: none !important;
        border-radius: 14px !important;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: #ffffff !important;
        font-size: 0.98rem !important;
        font-weight: 800 !important;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    .primary-action:hover,
    .primary-action button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 16px 28px rgba(37, 99, 235, 0.22);
    }

    .placeholder-note {
        margin-top: 14px;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(219, 234, 254, 0.55);
        border: 1px solid rgba(147, 197, 253, 0.35);
        color: var(--text-muted) !important;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .severity-stack {
        display: flex;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
    }

    .severity-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 800;
        width: fit-content;
    }

    .severity-low {
        background: rgba(22, 163, 74, 0.10) !important;
        border: 1px solid rgba(22, 163, 74, 0.22) !important;
        color: var(--success) !important;
    }

    .severity-high {
        background: rgba(220, 38, 38, 0.10) !important;
        border: 1px solid rgba(220, 38, 38, 0.22) !important;
        color: var(--danger) !important;
    }

    .explanation-input textarea {
        min-height: 240px !important;
        padding: 12px 14px !important;
        line-height: 1.55 !important;
        font-size: 0.95rem !important;
        color: #0f172a !important;
        background-color: #ffffff !important;
    }

    .assistant-chip {
        display: inline-flex;
        align-items: center;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.10);
        color: var(--primary-dark) !important;
        font-size: 0.88rem;
        font-weight: 700;
    }

    .chat-preview {
        display: grid;
        gap: 10px;
        margin: 14px 0;
    }

    .chat-bubble {
        max-width: 92%;
        padding: 12px 14px;
        border-radius: 18px;
        font-size: 0.92rem;
        line-height: 1.55;
        border: 1px solid transparent;
    }

    .chat-bubble-user {
        justify-self: end;
        background: rgba(37, 99, 235, 0.10) !important;
        border-color: rgba(37, 99, 235, 0.18) !important;
        color: var(--primary-dark) !important;
    }

    .chat-bubble-assistant {
        justify-self: start;
        background: rgba(248, 250, 252, 0.96) !important;
        border-color: rgba(203, 213, 225, 0.75) !important;
        color: var(--text-primary) !important;
    }

    /* Clean Step 5 follow-up input container */
    .assistant-compose {
        background: #ffffff !important;
        border: 1px solid rgba(147, 197, 253, 0.4) !important;
        border-radius: 18px !important;
        padding: 12px !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.03) !important;
        gap: 10px !important;
        margin-top: 10px !important;
    }

    .assistant-compose .block,
    .assistant-compose .form,
    .assistant-compose .row {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
    }

    .assistant-input,
    .assistant-input textarea,
    .assistant-input .container {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    .assistant-input textarea {
        min-height: 44px !important;
        padding: 10px 12px !important;
        color: #0f172a !important;
        background: #ffffff !important;
        border: 1px solid rgba(147, 197, 253, 0.4) !important;
        border-radius: 12px !important;
    }

    .assistant-button button {
        min-height: 44px;
        border-radius: 12px !important;
        font-weight: 800 !important;
        border: 1px solid transparent !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    .assistant-button button:disabled,
    .assistant-button button[disabled] {
        background: #eff6ff !important; /* light blue/gray background */
        border: 1px solid rgba(147, 197, 253, 0.4) !important;
        color: #1e3a8a !important; /* navy or muted blue text */
        cursor: not-allowed !important;
        box-shadow: none !important;
        opacity: 0.8 !important;
    }

    .assistant-button button:not(:disabled):not([disabled]) {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 16px rgba(37, 99, 235, 0.16) !important;
    }

    /* Step 6 summary workflow cards */
    .workflow-card {
        padding: 24px !important;
    }

    .workflow-header {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 16px;
    }

    .workflow-flow {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .workflow-step {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(248, 250, 252, 0.95) !important;
        border: 1px solid rgba(191, 219, 254, 0.72) !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem;
        font-weight: 700;
    }

    .workflow-step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent-cyan) 100%) !important;
        color: #ffffff !important;
        font-size: 0.7rem;
        font-weight: 800;
    }

    .workflow-arrow {
        color: var(--primary) !important;
        font-size: 1rem;
        font-weight: 800;
        text-align: center;
    }

    .status-banner {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid transparent;
    }

    .status-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        flex: 0 0 32px;
        border-radius: 50%;
        font-size: 0.9rem;
        font-weight: 800;
        background: rgba(255, 255, 255, 0.72) !important;
    }

    .status-copy {
        min-width: 0;
    }

    .status-label {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 800;
    }

    .status-message {
        margin: 4px 0 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .tone-alert {
        background: #fee2e2 !important;
        border: 2px solid #ef4444 !important;
        color: #7f1d1d !important;
    }

    .tone-alert .status-label,
    .tone-alert .status-message {
        color: #7f1d1d !important;
    }

    .tone-alert .status-icon {
        background: #fca5a5 !important;
        color: #7f1d1d !important;
    }

    .tone-safe {
        background: #d1fae5 !important;
        border: 1px solid #10b981 !important;
        color: #064e3b !important;
    }

    .tone-safe .status-label,
    .tone-safe .status-message {
        color: #064e3b !important;
    }

    .tone-safe .status-icon {
        background: #a7f3d0 !important;
        color: #064e3b !important;
    }

    .tone-neutral {
        background: #dbeafe !important;
        border: 1px solid #3b82f6 !important;
        color: #1e3a8a !important;
    }

    .tone-neutral .status-label,
    .tone-neutral .status-message {
        color: #1e3a8a !important;
    }

    .tone-neutral .status-icon {
        background: #bfdbfe !important;
        color: #1e3a8a !important;
    }

    .tone-warning {
        background: #ffedd5 !important;
        border: 1px solid #f97316 !important;
        color: #7c2d12 !important;
    }

    .tone-warning .status-label,
    .tone-warning .status-message {
        color: #7c2d12 !important;
    }

    .tone-warning .status-icon {
        background: #fed7aa !important;
        color: #7c2d12 !important;
    }

    .footer-card {
        padding: 16px 20px;
        text-align: center;
        color: var(--text-muted) !important;
        font-size: 0.92rem;
        background: rgba(255, 255, 255, 0.88);
    }

    .panel-card .gradio-container,
    .panel-card .block {
        background: transparent !important;
    }

    @media (max-width: 1300px) {
        .panel-card {
            flex: 1 1 30% !important; /* wraps nicely on medium screen */
        }
    }

    @media (max-width: 768px) {
        .content-grid {
            flex-direction: column !important;
        }
        .panel-card {
            flex: 1 1 100% !important; /* stacks vertically on mobile */
        }
        .gradio-container {
            padding: 20px 14px 28px !important;
        }

        .hero-card,
        .panel-card {
            padding: 22px;
            border-radius: 20px;
        }

        .workflow-card {
            padding: 22px;
        }

        .status-banner {
            padding: 16px;
        }

        .hero-title {
            font-size: 2.2rem;
        }

        .emergency-alert-card,
        .alert-controls-card {
            padding: 18px;
            border-radius: 20px;
        }

        .alert-controls-grid {
            grid-template-columns: 1fr;
        }

        .alert-controls-actions {
            flex-direction: column;
        }

        .alert-control-button {
            width: 100%;
        }
    }
    """
    blue_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(
        title="Total Accident Detection & Road Safety Intelligence",
    ) as demo:
        with gr.Column(elem_id="page-shell"):
            # Gradient Top Header Banner
            gr.HTML(
                """
                <div class="hero-header" style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1d4ed8 100%); padding: 40px; border-radius: 24px; color: white; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(37, 99, 235, 0.15);">
                    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 900; letter-spacing: -0.03em; color: white;">Accident Detection and Traffic Law Workflow</h1>
                    <p style="margin: 12px 0 0; font-size: 1.15rem; font-weight: 500; opacity: 0.95; line-height: 1.5; color: #dbeafe;">
                        Upload road evidence, run YOLO accident detection, classify severity, and review the Jordan Traffic Law assistant report.
                    </p>
                </div>
                """
            )

            # Workflow Strip (Steps 01-06 horizontal demo pipeline)
            gr.HTML(
                """
                <div class="workflow-strip-card" style="background: white; border: 1px solid rgba(147, 197, 253, 0.45); border-radius: 20px; padding: 20px; margin-bottom: 26px; box-shadow: 0 4px 20px rgba(29, 78, 216, 0.04);">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #2563eb; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">Workflow / Capstone Demo Pipeline</div>
                    <div class="workflow-strip-steps" style="display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;">
                        <div class="workflow-strip-step" style="flex: 1; min-width: 140px; display: flex; align-items: center; gap: 10px; padding: 12px 14px; background: #eff6ff; border-radius: 12px; border: 1px solid rgba(147, 197, 253, 0.35);">
                            <span style="font-size: 1.1rem; font-weight: 900; color: #2563eb;">01</span>
                            <span style="font-size: 0.85rem; font-weight: 700; color: #1e3a8a;">Upload Image / Video</span>
                        </div>
                        <div style="font-weight: 800; color: #2563eb; font-size: 1.2rem;">&rarr;</div>
                        <div class="workflow-strip-step" style="flex: 1; min-width: 140px; display: flex; align-items: center; gap: 10px; padding: 12px 14px; background: #eff6ff; border-radius: 12px; border: 1px solid rgba(147, 197, 253, 0.35);">
                            <span style="font-size: 1.1rem; font-weight: 900; color: #2563eb;">02</span>
                            <span style="font-size: 0.85rem; font-weight: 700; color: #1e3a8a;">YOLO Accident Detection</span>
                        </div>
                        <div style="font-weight: 800; color: #2563eb; font-size: 1.2rem;">&rarr;</div>
                        <div class="workflow-strip-step" style="flex: 1; min-width: 140px; display: flex; align-items: center; gap: 10px; padding: 12px 14px; background: #eff6ff; border-radius: 12px; border: 1px solid rgba(147, 197, 253, 0.35);">
                            <span style="font-size: 1.1rem; font-weight: 900; color: #2563eb;">03</span>
                            <span style="font-size: 0.85rem; font-weight: 700; color: #1e3a8a;">Severity Classification</span>
                        </div>
                        <div style="font-weight: 800; color: #2563eb; font-size: 1.2rem;">&rarr;</div>
                        <div class="workflow-strip-step" style="flex: 1; min-width: 140px; display: flex; align-items: center; gap: 10px; padding: 12px 14px; background: #eff6ff; border-radius: 12px; border: 1px solid rgba(147, 197, 253, 0.35);">
                            <span style="font-size: 1.1rem; font-weight: 900; color: #2563eb;">04</span>
                            <span style="font-size: 0.85rem; font-weight: 700; color: #1e3a8a;">Jordan Traffic Law Assistant</span>
                        </div>
                        <div style="font-weight: 800; color: #2563eb; font-size: 1.2rem;">&rarr;</div>
                        <div class="workflow-strip-step" style="flex: 1; min-width: 140px; display: flex; align-items: center; gap: 10px; padding: 12px 14px; background: #eff6ff; border-radius: 12px; border: 1px solid rgba(147, 197, 253, 0.35);">
                            <span style="font-size: 1.1rem; font-weight: 900; color: #2563eb;">05</span>
                            <span style="font-size: 0.85rem; font-weight: 700; color: #1e3a8a;">User Discussion / Follow-up Context</span>
                        </div>
                        <div style="font-weight: 800; color: #2563eb; font-size: 1.2rem;">&rarr;</div>
                        <div class="workflow-strip-step" style="flex: 1; min-width: 140px; display: flex; align-items: center; gap: 10px; padding: 12px 14px; background: #eff6ff; border-radius: 12px; border: 1px solid rgba(147, 197, 253, 0.35);">
                            <span style="font-size: 1.1rem; font-weight: 900; color: #2563eb;">06</span>
                            <span style="font-size: 0.85rem; font-weight: 700; color: #1e3a8a;">Final Report / Summary</span>
                        </div>
                    </div>
                </div>
                """
            )

            # Model Weights Status Badge
            gr.HTML(build_model_badge(BEST_MODEL_PATH))

            # Main Full-width Dashboard Row
            with gr.Row(equal_height=True, elem_classes="content-grid"):
                # Column 1: Step 1 (Upload Card) - scale=5
                with gr.Column(scale=5, elem_classes="panel-card upload-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">IN</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Step 1: Upload Image / Video</p>
                                <h2 class="panel-title">Accident Evidence</h2>
                                <p class="panel-description">
                                    Choose testing mode, upload your file, and set YOLO detection threshold.
                                </p>
                            </div>
                        </div>
                        """
                    )
                    
                    input_mode = gr.Radio(
                        choices=["Image", "Video"],
                        value="Image",
                        label="Select Input Mode",
                        elem_classes="mode-selector",
                    )
                    
                    # IMAGE INPUT GROUP
                    with gr.Column(visible=True) as image_input_group:
                        image_input = gr.Image(
                            type="numpy",
                            label="Upload and Preview Road Image",
                            elem_classes="image-shell",
                            elem_id="image-preview-input",
                            interactive=True,
                        )
                        with gr.Column(elem_classes="threshold-card"):
                            threshold_readout = gr.HTML(build_threshold_readout(0.25))
                            conf_slider = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                value=0.25,
                                step=0.01,
                                show_label=False,
                                elem_classes="threshold-slider",
                                elem_id="image-confidence-slider",
                            )
                        submit_btn = gr.Button(
                            "Run Inference",
                            variant="primary",
                            elem_classes="primary-action",
                            elem_id="run-image-inference-btn",
                        )
                    
                    # VIDEO INPUT GROUP
                    with gr.Column(visible=False) as video_input_group:
                        gr.HTML(
                            """
                            <div class="placeholder-note" style="margin-top: 0; margin-bottom: 14px; background: rgba(37, 99, 235, 0.08); border-color: rgba(147, 197, 253, 0.45); color: var(--primary-dark); font-weight: 600;">
                                Video Inference Testing Mode<br/>
                                <span style="font-weight: 400; font-size: 0.9em; opacity: 0.95;">
                                    For faster testing, use short MP4 clips between 3–10 seconds.
                                </span>
                            </div>
                            """
                        )
                        video_input = gr.Video(
                            label="Upload and Preview Road Video",
                            elem_classes="image-shell",
                            elem_id="video-preview-input",
                            interactive=True,
                        )
                        with gr.Column(elem_classes="threshold-card"):
                            video_threshold_readout = gr.HTML(build_threshold_readout(0.25))
                            video_conf_slider = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                value=0.25,
                                step=0.01,
                                show_label=False,
                                elem_classes="threshold-slider",
                                elem_id="video-confidence-slider",
                            )
                        video_submit_btn = gr.Button(
                            "Run Video Inference",
                            variant="primary",
                            elem_classes="primary-action",
                            elem_id="run-video-inference-btn",
                        )

                # Column 2: Steps 2-3 (Detection & Severity Output Section) - scale=9 (wider feedback card)
                with gr.Column(scale=9, elem_classes="panel-card output-card", elem_id="detection-output-panel"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">OUT</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Steps 2 &amp; 3: YOLO Accident Detection &amp; Severity Classification</p>
                                <h2 class="panel-title">Detection and severity output</h2>
                                <p class="panel-description">
                                    Monitor the alert state, review the current workflow message, and inspect the annotated detection output.
                                </p>
                            </div>
                        </div>
                        """
                    )
                    
                    alert_banner_output = gr.HTML(
                        value=build_alert_banner("standby", "image"),
                        elem_id="emergency-alert-region",
                    )
                    alert_signal_output = gr.HTML(
                        value=build_alert_signal(False, "idle", "image"),
                        elem_id="alert-signal-region",
                    )

                    # Severity classification category indicators inside output card!
                    gr.HTML(
                        """
                        <div style="margin-top: 14px; margin-bottom: 16px;">
                            <div style="font-size: 0.78rem; font-weight: 800; color: #1e3a8a; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Active Severity Categories</div>
                            <div class="severity-stack" style="display: flex; gap: 10px; flex-wrap: wrap;">
                                <span class="severity-pill severity-low">Low Severity</span>
                                <span class="severity-pill severity-high">Severe Severity</span>
                            </div>
                            <div class="placeholder-note" style="margin-top: 8px; margin-bottom: 0; padding: 10px 12px; font-size: 0.82rem; background: rgba(37, 99, 235, 0.05); border: 1px solid rgba(147, 197, 253, 0.3);">
                                Categories: moderate-accident, severe-accident, no-accident (mapped to Low Severity). Detections are classified dynamically.
                            </div>
                        </div>
                        """,
                        visible=False
                    )

                    # IMAGE OUTPUT GROUP
                    with gr.Column(visible=True, elem_id="image-output-group") as image_output_group:
                        status_output = gr.HTML(
                            value=build_status_banner(
                                title="Awaiting inference",
                                message="Upload a road image and click Run Inference to generate an accident detection result.",
                                tone="neutral",
                                icon="&#9711;",
                            ),
                            elem_id="image-status-region",
                        )
                        image_output = gr.Image(
                            type="numpy",
                            label="Rendered Output with Bounding Boxes",
                            elem_classes="image-shell",
                            elem_id="image-preview-output",
                        )
                        
                    # VIDEO OUTPUT GROUP
                    with gr.Column(visible=False, elem_id="video-output-group") as video_output_group:
                        video_status_output = gr.HTML(
                            value=build_status_banner(
                                title="No video uploaded",
                                message="Upload a road video and click Run Video Inference to generate an accident detection result.",
                                tone="neutral",
                                icon="&#8682;",
                            ),
                            elem_id="video-status-region",
                        )
                        video_output = gr.Video(
                            label="Rendered Video Output",
                            elem_classes="image-shell",
                            elem_id="video-preview-output",
                        )
                        video_placeholder = gr.HTML(
                            value="<div class='placeholder-note' style='text-align: center; margin-top: 10px;'>Video output preview will appear here after processing.</div>",
                            visible=True
                        )

                # Column 3: Step 4 (Accident Law Assistant & Follow-up Chat) - scale=6
                with gr.Column(scale=6, elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">LAW</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Step 4: Accident Law Assistant &amp; Follow-up Chat</p>
                                <h2 class="panel-title">Jordan Traffic Law Assistant Chat</h2>
                                <p class="panel-description" style="margin-top: 6px;">
                                    <span class="assistant-chip">قانون السير الأردني</span>
                                    Ask follow-up questions or discuss the AI liability analysis below.
                                </p>
                            </div>
                        </div>
                        """
                    )
                    chatbot = gr.HTML(
                        value=render_chat_html([]),
                        elem_id="law-chatbot",
                        elem_classes="chat-preview",
                    )
                    
                    with gr.Row(elem_classes="assistant-compose"):
                        law_prompt = gr.Textbox(
                            lines=1,
                            show_label=False,
                            placeholder="Upload an accident and run inference to start AI analysis...",
                            interactive=False,
                            elem_classes="assistant-input",
                            elem_id="law-assistant-prompt",
                            scale=5,
                        )
                        law_button = gr.Button(
                            "Locked",
                            interactive=False,
                            elem_classes="assistant-button",
                            elem_id="law-assistant-button",
                            scale=2,
                        )
                    agent_state = gr.State(None)
                    chat_history = gr.State([])

                # Column 4: Step 5 (Final Report / Summary) - scale=5
                with gr.Column(scale=5, elem_classes="panel-card workflow-card", visible=False):
                    gr.HTML(
                        """
                        <div class="workflow-header">
                            <span class="section-icon">FLOW</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Step 5: Final Report / Summary</p>
                                <h2 class="panel-title">Final Report Summary</h2>
                                <p class="panel-description">
                                    The full Sequential Pipeline workflow and active Capstone system roadmap.
                                </p>
                            </div>
                        </div>
                        <div class="workflow-flow" style="display: flex; flex-direction: column; gap: 8px; align-items: center; margin-top: 14px;">
                            <div class="workflow-step" style="width: 100%;"><span class="workflow-step-badge">1</span>Step 1: Upload Image / Video</div>
                            <span class="workflow-arrow" style="font-size: 1.1rem; line-height: 1;">&darr;</span>
                            <div class="workflow-step" style="width: 100%;"><span class="workflow-step-badge">2</span>Step 2: YOLO Accident Detection</div>
                            <span class="workflow-arrow" style="font-size: 1.1rem; line-height: 1;">&darr;</span>
                            <div class="workflow-step" style="width: 100%;"><span class="workflow-step-badge">3</span>Step 3: Severity Classification</div>
                            <span class="workflow-arrow" style="font-size: 1.1rem; line-height: 1;">&darr;</span>
                            <div class="workflow-step" style="width: 100%;"><span class="workflow-step-badge">4</span>Step 4: Accident Law Assistant &amp; Follow-up Chat</div>
                            <span class="workflow-arrow" style="font-size: 1.1rem; line-height: 1;">&darr;</span>
                            <div class="workflow-step" style="width: 100%;"><span class="workflow-step-badge">5</span>Step 5: Final Report / Summary</div>
                        </div>
                        """
                    )

            # Step 5: Final Detection Summary Section (Full-Width Card under the top row)
            with gr.Row():
                with gr.Column(scale=12):
                    final_summary_output = gr.HTML(
                        value=render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", ""),
                        elem_id="final-detection-summary-region"
                    )

            gr.HTML(
                """
                <div class="footer-card">
                    Designed for AI accident detection, capstone demonstration, and future smart-road safety integration.
                </div>
                """
            )

        # Mode Switching Transition Event Handler
        def on_mode_change(mode):
            if mode == "Image":
                return (
                    gr.update(visible=True),  # image_input_group
                    gr.update(visible=False), # video_input_group
                    gr.update(visible=True),  # image_output_group
                    gr.update(visible=False), # video_output_group
                    None,                     # Clear video_input
                    None,                     # Clear video_output
                    build_status_banner(
                        title="No video uploaded",
                        message="Please switch to video mode and upload a video to test.",
                        tone="neutral",
                        icon="&#9711;",
                    ),                        # Reset video status
                    None,                     # Clear image_input
                    None,                     # Clear image_output
                    build_status_banner(
                        title="Awaiting inference",
                        message="Upload a road image and click Run Inference to generate an accident detection result.",
                        tone="neutral",
                        icon="&#9711;",
                    ),                        # Reset image status
                    gr.update(visible=False), # Hide video placeholder in Image mode
                    build_alert_banner("standby", "image"),
                    build_alert_signal(False, "idle", "image"),
                    render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", ""),
                )
            else:
                return (
                    gr.update(visible=False), # image_input_group
                    gr.update(visible=True),  # video_input_group
                    gr.update(visible=False), # image_output_group
                    gr.update(visible=True),  # video_output_group
                    None,                     # Clear video_input
                    None,                     # Clear video_output
                    build_status_banner(
                        title="No video uploaded",
                        message="Upload a road video and click Run Video Inference to generate an accident detection result.",
                        tone="neutral",
                        icon="&#8682;",
                    ),                        # Reset video status
                    None,                     # Clear image_input
                    None,                     # Clear image_output
                    build_status_banner(
                        title="Awaiting inference",
                        message="Please switch to image mode and upload an image to test.",
                        tone="neutral",
                        icon="&#9711;",
                    ),                        # Reset image status
                    gr.update(visible=True),  # Show video placeholder in Video mode
                    build_alert_banner("standby", "video"),
                    build_alert_signal(False, "idle", "video"),
                    render_final_summary_html(None, 0.0, None, 0.0, "Not Escalated", ""),
                )

        input_mode.change(
            fn=on_mode_change,
            inputs=input_mode,
            outputs=[
                image_input_group,
                video_input_group,
                image_output_group,
                video_output_group,
                video_input,
                video_output,
                video_status_output,
                image_input,
                image_output,
                status_output,
                video_placeholder,
                alert_banner_output,
                alert_signal_output,
                final_summary_output,
            ],
        )

        # Upload and Clear Event handlers to dynamically update status banners
        image_input.upload(
            fn=handle_image_upload,
            inputs=image_input,
            outputs=[status_output, image_output, alert_banner_output, alert_signal_output, final_summary_output],
        )
        image_input.clear(
            fn=handle_image_clear,
            outputs=[status_output, image_output, alert_banner_output, alert_signal_output, final_summary_output],
        )

        video_input.upload(
            fn=handle_video_upload,
            inputs=video_input,
            outputs=[video_status_output, video_output, video_placeholder, alert_banner_output, alert_signal_output, final_summary_output],
        )
        video_input.clear(
            fn=handle_video_clear,
            outputs=[video_status_output, video_output, video_placeholder, alert_banner_output, alert_signal_output, final_summary_output],
        )

        # Inference Trigger Event Handlers
        submit_btn.click(
            fn=run_image_inference,
            inputs=[image_input, conf_slider, chat_history, agent_state],
            outputs=[image_output, status_output, alert_banner_output, alert_signal_output, chatbot, law_prompt, law_button, agent_state, chat_history, final_summary_output],
        )

        video_submit_btn.click(
            fn=run_video_inference,
            inputs=[video_input, video_conf_slider, chat_history, agent_state],
            outputs=[video_output, video_status_output, video_placeholder, alert_banner_output, alert_signal_output, chatbot, law_prompt, law_button, agent_state, chat_history, final_summary_output],
        )

        def initiate_chat(user_message, chat_history):
            chat_history = chat_history or []
            if not user_message or not user_message.strip():
                return "", render_chat_html(chat_history), chat_history, gr.update(), gr.update()
            chat_history.append((user_message, "⏳ Thinking..."))
            return (
                "",
                render_chat_html(chat_history),
                chat_history,
                gr.update(interactive=False, placeholder="Analyzing query with AI agent..."),
                gr.update(interactive=False, value="Sending...")
            )

        def generate_chat_reply(chat_history, agent):
            if not chat_history:
                return render_chat_html([]), [], gr.update(interactive=True, placeholder="Ask the legal assistant..."), gr.update(interactive=True, value="Send")
            
            user_message, _ = chat_history[-1]
            
            if not agent:
                chat_history[-1] = (user_message, "⚠️ AI Agent is offline. OpenRouter key or required files are missing.")
                return render_chat_html(chat_history), chat_history, gr.update(interactive=True, placeholder="Ask the legal assistant..."), gr.update(interactive=True, value="Send")

            try:
                reply = safe_agent_call(agent, "chat_with_user", user_message)
                chat_history[-1] = (user_message, reply)
            except Exception as e:
                print(f"[Chat Error]: {e}")
                chat_history[-1] = (user_message, f"❌ Agent analysis failed: {str(e)}")

            return render_chat_html(chat_history), chat_history, gr.update(interactive=True, placeholder="Ask the legal assistant..."), gr.update(interactive=True, value="Send")

        # Chatbot prompt/button event handlers
        law_prompt.submit(
            fn=initiate_chat,
            inputs=[law_prompt, chat_history],
            outputs=[law_prompt, chatbot, chat_history, law_prompt, law_button],
            queue=False
        ).then(
            fn=generate_chat_reply,
            inputs=[chat_history, agent_state],
            outputs=[chatbot, chat_history, law_prompt, law_button]
        )

        law_button.click(
            fn=initiate_chat,
            inputs=[law_prompt, chat_history],
            outputs=[law_prompt, chatbot, chat_history, law_prompt, law_button],
            queue=False
        ).then(
            fn=generate_chat_reply,
            inputs=[chat_history, agent_state],
            outputs=[chatbot, chat_history, law_prompt, law_button]
        )

        # Confidence Sliders Parameter Updates
        conf_slider.change(
            fn=build_threshold_readout,
            inputs=conf_slider,
            outputs=threshold_readout,
        )

        video_conf_slider.change(
            fn=build_threshold_readout,
            inputs=video_conf_slider,
            outputs=video_threshold_readout,
        )

    demo._deprecated_theme = blue_theme
    demo._deprecated_css = custom_css
    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue()  # Enable queuing for robust generator/yield processing!
    try:
        app.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            head=build_alert_controller_head(),
        )
    except OSError:
        print("Port 7860 is occupied. Launching on an automatically allocated free port...")
        app.launch(
            share=False,
            server_name="127.0.0.1",
            head=build_alert_controller_head(),
        )
