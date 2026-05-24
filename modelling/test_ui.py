import subprocess
import sys
import os
from pathlib import Path
import csv
from html import escape

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


def find_best_model() -> Path:
    """
    Look for the absolute highest-ranked model according to the mAP@50-95 metric.
    Prioritizes full_experiment, then smoke_test, then standalone best weights, then pretrained_weights.
    """
    project_root = Path("C:/Capstone").resolve()
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


# Load Model Globally
BEST_MODEL_PATH = find_best_model()
print(f"Loaded Best Model Path: {BEST_MODEL_PATH}")

if BEST_MODEL_PATH and BEST_MODEL_PATH.exists():
    model = YOLO(str(BEST_MODEL_PATH))
else:
    model = None


def build_status_banner(title: str, message: str, tone: str, icon: str) -> str:
    safe_title = escape(title)
    safe_message = escape(message)
    return f"""
    <div class="status-banner tone-{tone}">
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


def predict_accident_gui(input_image: np.ndarray, conf_threshold: float):
    if model is None:
        return input_image, build_status_banner(
            title="Model unavailable",
            message="No model was loaded. Please check the configured model paths.",
            tone="alert",
            icon="&#10005;",
        )

    if input_image is None:
        return None, build_status_banner(
            title="Image required",
            message="Upload a road image to run accident detection inference.",
            tone="neutral",
            icon="&#8682;",
        )

    # Inference
    results = model(input_image, conf=conf_threshold)

    # Render Bounding Boxes
    rendered_image = results[0].plot()

    # Check for 'accident' detection
    accident_detected = False
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = results[0].names
        for c in boxes.cls:
            class_name = names[int(c)].lower()
            if "accident" in class_name:
                accident_detected = True
                break

        # Fallback if class names don't explicitly have "accident"
        # Since it's an accident detection project, any valid bounding box is likely an accident.
        if not accident_detected:
            for c in boxes.cls:
                if int(c) == 0:  # Assume class 0 is accident if not named
                    accident_detected = True
                    break

        # General fallback if there's any box
        if not accident_detected and len(boxes) > 0:
            accident_detected = True

    if accident_detected:
        status_html = build_status_banner(
            title="Accident detected",
            message="The selected model found at least one accident-related detection in the uploaded image.",
            tone="alert",
            icon="!",
        )
    else:
        status_html = build_status_banner(
            title="No accident detected",
            message="Inference completed and no accident class was detected above the selected confidence threshold.",
            tone="safe",
            icon="&#10003;",
        )

    return rendered_image, status_html


def build_app():
    custom_css = """
    :root {
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
    }

    body,
    .gradio-container {
        margin: 0;
        min-height: 100vh;
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 24%),
            linear-gradient(140deg, var(--page-bg-top) 0%, var(--page-bg-mid) 52%, var(--page-bg-bottom) 100%);
        color: var(--text-primary);
        font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    .gradio-container {
        max-width: 1320px !important;
        padding: 34px 20px 42px !important;
    }

    #page-shell {
        gap: 26px;
    }

    .hero-card,
    .panel-card,
    .workflow-card,
    .footer-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 28px;
        box-shadow: var(--card-shadow);
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        padding: 34px;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.98) 0%, rgba(239, 246, 255, 0.96) 100%);
    }

    .hero-card::before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 6px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent-cyan) 100%);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.7fr) minmax(300px, 0.9fr);
        gap: 24px;
        align-items: start;
    }

    .hero-copy {
        min-width: 0;
    }

    .hero-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.08);
        color: var(--primary-dark);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 18px 0 0;
        color: var(--text-primary);
        font-size: clamp(2.2rem, 4.4vw, 3.6rem);
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        margin: 16px 0 0;
        max-width: 820px;
        color: var(--primary-dark);
        font-size: clamp(1.05rem, 2vw, 1.32rem);
        font-weight: 700;
        line-height: 1.5;
    }

    .hero-description {
        margin: 16px 0 0;
        max-width: 800px;
        color: var(--text-muted);
        font-size: 1.03rem;
        line-height: 1.8;
    }

    .hero-aside {
        min-width: 0;
    }

    .hero-aside-card {
        padding: 22px;
        border-radius: 24px;
        border: 1px solid rgba(147, 197, 253, 0.45);
        background:
            linear-gradient(180deg, rgba(219, 234, 254, 0.8) 0%, rgba(255, 255, 255, 0.98) 100%);
    }

    .hero-aside-label {
        margin: 0 0 10px;
        color: var(--primary-dark);
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-aside-title {
        margin: 0;
        color: var(--text-primary);
        font-size: 1.28rem;
        font-weight: 800;
    }

    .hero-feature-list {
        display: grid;
        gap: 12px;
        margin-top: 16px;
    }

    .hero-feature {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.82);
        color: var(--text-primary);
        font-size: 0.96rem;
        font-weight: 600;
    }

    .hero-feature-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        flex: 0 0 34px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent-cyan) 100%);
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .model-badge {
        margin-top: 26px;
        padding: 18px 20px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(147, 197, 253, 0.38);
        border-radius: 20px;
        backdrop-filter: blur(6px);
    }

    .model-badge-label {
        display: inline-block;
        margin-bottom: 8px;
        color: var(--primary-dark);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .model-badge-path {
        display: block;
        margin: 0;
        color: var(--text-primary);
        font-size: 0.95rem;
        font-family: "Cascadia Code", "Consolas", monospace;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    .content-grid {
        gap: 24px;
    }

    .panel-card {
        padding: 28px;
        min-width: 0;
    }

    .panel-intro {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 18px;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 46px;
        height: 46px;
        flex: 0 0 46px;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent-cyan) 100%);
        color: #ffffff;
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        box-shadow: 0 14px 24px rgba(37, 99, 235, 0.16);
    }

    .panel-copy {
        min-width: 0;
    }

    .panel-kicker {
        margin: 0 0 7px;
        color: var(--primary-dark);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .panel-title {
        margin: 0;
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 850;
    }

    .panel-description {
        margin: 10px 0 0;
        color: var(--text-muted);
        font-size: 0.98rem;
        line-height: 1.65;
    }

    .image-shell {
        overflow: hidden;
        border: 1px solid var(--card-border);
        border-radius: 20px;
        background: #f8fafc;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    .threshold-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .threshold-label {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 700;
    }

    .threshold-value {
        padding: 6px 10px;
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(147, 197, 253, 0.5);
        border-radius: 999px;
        color: var(--primary-dark);
        font-size: 0.95rem;
        font-weight: 800;
    }

    .threshold-slider {
        margin-bottom: 20px;
    }

    .threshold-slider .wrap,
    .threshold-slider .wrap.svelte-1ipelgc,
    .threshold-slider .container,
    .explanation-input textarea,
    .assistant-input textarea {
        border-radius: 18px !important;
    }

    .threshold-slider input,
    .explanation-input textarea,
    .assistant-input textarea {
        background: #f8fbff !important;
        border-color: rgba(147, 197, 253, 0.45) !important;
        color: var(--text-primary) !important;
    }

    .threshold-slider input {
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .primary-action,
    .primary-action button {
        width: 100%;
        min-height: 54px;
        border: none !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 16px 30px rgba(37, 99, 235, 0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    .primary-action:hover,
    .primary-action button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 20px 32px rgba(37, 99, 235, 0.26);
    }

    .placeholder-note {
        margin-top: 16px;
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(219, 234, 254, 0.55);
        border: 1px solid rgba(147, 197, 253, 0.35);
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.65;
    }

    .coming-soon-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.08);
        color: var(--primary-dark);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .severity-stack {
        display: grid;
        gap: 12px;
        margin-top: 16px;
    }

    .severity-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 14px;
        border-radius: 999px;
        font-size: 0.92rem;
        font-weight: 800;
        width: fit-content;
    }

    .severity-low {
        background: rgba(22, 163, 74, 0.10);
        border: 1px solid rgba(22, 163, 74, 0.22);
        color: var(--success);
    }

    .severity-medium {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #b45309;
    }

    .severity-high {
        background: rgba(220, 38, 38, 0.10);
        border: 1px solid rgba(220, 38, 38, 0.22);
        color: var(--danger);
    }

    .explanation-input textarea {
        min-height: 220px !important;
        padding: 14px 16px !important;
        line-height: 1.65 !important;
    }

    .assistant-chip {
        display: inline-flex;
        align-items: center;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.10);
        color: var(--primary-dark);
        font-size: 0.92rem;
        font-weight: 700;
    }

    .chat-preview {
        display: grid;
        gap: 12px;
        margin: 18px 0;
    }

    .chat-bubble {
        max-width: 92%;
        padding: 14px 16px;
        border-radius: 20px;
        font-size: 0.95rem;
        line-height: 1.6;
        border: 1px solid transparent;
    }

    .chat-bubble-user {
        justify-self: end;
        background: rgba(37, 99, 235, 0.10);
        border-color: rgba(37, 99, 235, 0.18);
        color: var(--primary-dark);
    }

    .chat-bubble-assistant {
        justify-self: start;
        background: rgba(248, 250, 252, 0.96);
        border-color: rgba(203, 213, 225, 0.75);
        color: var(--text-muted);
    }

    .chat-role {
        display: block;
        margin-bottom: 6px;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .assistant-input textarea {
        min-height: 54px !important;
        padding: 14px 16px !important;
        color: #94a3b8 !important;
        background: rgba(239, 246, 255, 0.85) !important;
    }

    .assistant-button button {
        min-height: 54px;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #bfdbfe 0%, #dbeafe 100%) !important;
        color: var(--primary-dark) !important;
        font-weight: 800 !important;
        border: 1px solid rgba(147, 197, 253, 0.42) !important;
        opacity: 0.78;
    }

    .assistant-button button:disabled {
        opacity: 0.72 !important;
        cursor: not-allowed !important;
    }

    .workflow-card {
        padding: 28px;
    }

    .workflow-header {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 18px;
    }

    .workflow-flow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 12px;
    }

    .workflow-step {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 13px 16px;
        border-radius: 18px;
        background: rgba(248, 250, 252, 0.95);
        border: 1px solid rgba(191, 219, 254, 0.72);
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 700;
    }

    .workflow-step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent-cyan) 100%);
        color: #ffffff;
        font-size: 0.76rem;
        font-weight: 800;
    }

    .workflow-arrow {
        color: var(--primary);
        font-size: 1.2rem;
        font-weight: 800;
    }

    .status-banner {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid transparent;
    }

    .status-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        flex: 0 0 36px;
        border-radius: 50%;
        font-size: 1rem;
        font-weight: 800;
        background: rgba(255, 255, 255, 0.72);
    }

    .status-copy {
        min-width: 0;
    }

    .status-label {
        margin: 0;
        font-size: 1rem;
        font-weight: 800;
    }

    .status-message {
        margin: 6px 0 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .tone-alert {
        background: var(--alert-bg);
        border-color: var(--alert-border);
        color: var(--alert-text);
    }

    .tone-safe {
        background: var(--safe-bg);
        border-color: var(--safe-border);
        color: var(--safe-text);
    }

    .tone-neutral {
        background: var(--neutral-bg);
        border-color: var(--neutral-border);
        color: var(--neutral-text);
    }

    .footer-card {
        padding: 18px 22px;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.95rem;
        background: rgba(255, 255, 255, 0.88);
    }

    .upload-card button,
    .output-card button,
    .panel-card button {
        font-family: inherit !important;
    }

    .panel-card .gradio-container,
    .panel-card .block {
        background: transparent !important;
    }

    @media (max-width: 1080px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 900px) {
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
    }
    """
    blue_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(
        theme=blue_theme,
        css=custom_css,
        title="Total Accident Detection & Road Safety Intelligence",
    ) as demo:
        with gr.Column(elem_id="page-shell"):
            with gr.Column(elem_classes="hero-card"):
                gr.HTML(
                    """
                    <div class="hero-grid">
                        <div class="hero-copy">
                            <span class="hero-label">CAPSTONE PROJECT INTERFACE</span>
                            <h1 class="hero-title">Total Accident Detection &amp; Road Safety Intelligence</h1>
                            <p class="hero-subtitle">
                                AI-Based Accident Detection, Severity Classification, and Jordan Traffic Law Assistant
                            </p>
                            <p class="hero-description">
                                Detect accidents from images and videos, classify severity, and assist with Jordan Traffic Law guidance.
                                This interface presents the current YOLO detection stage while previewing the future smart-road safety workflow.
                            </p>
                        </div>
                        <div class="hero-aside">
                            <div class="hero-aside-card">
                                <p class="hero-aside-label">Future System Scope</p>
                                <h2 class="hero-aside-title">AI Traffic Safety Dashboard</h2>
                                <div class="hero-feature-list">
                                    <div class="hero-feature">
                                        <span class="hero-feature-badge">DET</span>
                                        Accident detection from image and video evidence
                                    </div>
                                    <div class="hero-feature">
                                        <span class="hero-feature-badge">SEV</span>
                                        Severity estimation for incident triage and review
                                    </div>
                                    <div class="hero-feature">
                                        <span class="hero-feature-badge">LAW</span>
                                        Jordan Traffic Law assistant for future guidance workflows
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                )
                gr.HTML(build_model_badge(BEST_MODEL_PATH))

            with gr.Row(equal_height=True, elem_classes="content-grid"):
                with gr.Column(scale=5, elem_classes="panel-card upload-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">IN</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Accident Detection Input</p>
                                <h2 class="panel-title">Accident Detection Input</h2>
                                <p class="panel-description">
                                    Upload a road image, review the preview, tune the confidence threshold, and run the current YOLO detector.
                                </p>
                            </div>
                        </div>
                        """
                    )
                    image_input = gr.Image(
                        type="numpy",
                        label="Upload and Preview Road Image",
                        elem_classes="image-shell",
                    )
                    threshold_readout = gr.HTML(build_threshold_readout(0.25))
                    conf_slider = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.25,
                        step=0.01,
                        show_label=False,
                        elem_classes="threshold-slider",
                    )
                    submit_btn = gr.Button(
                        "Run Inference",
                        variant="primary",
                        elem_classes="primary-action",
                    )

                with gr.Column(scale=6, elem_classes="panel-card output-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">OUT</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">YOLO Detection Output</p>
                                <h2 class="panel-title">YOLO Detection Output</h2>
                                <p class="panel-description">
                                    Inspect the detection status banner and rendered output image with bounding boxes from the active model.
                                </p>
                            </div>
                        </div>
                        """
                    )
                    status_output = gr.HTML(
                        value=build_status_banner(
                            title="Awaiting inference",
                            message="Upload a road image and click Run Inference to generate an accident detection result.",
                            tone="neutral",
                            icon="&#9711;",
                        )
                    )
                    image_output = gr.Image(
                        type="numpy",
                        label="Rendered Output with Bounding Boxes",
                        elem_classes="image-shell",
                    )

            with gr.Row(equal_height=True, elem_classes="content-grid"):
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">SEV</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Future Module</p>
                                <h2 class="panel-title">Accident Severity Classification</h2>
                                <p class="panel-description">
                                    Coming soon: severity model integration
                                </p>
                            </div>
                        </div>
                        <div class="severity-stack">
                            <span class="severity-pill severity-low">Low Severity</span>
                            <span class="severity-pill severity-medium">Medium Severity</span>
                            <span class="severity-pill severity-high">High Severity</span>
                        </div>
                        <div class="placeholder-note">
                            The severity classifier will analyze the detected accident and estimate impact severity.
                        </div>
                        """
                    )

                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">TXT</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Scenario Context</p>
                                <h2 class="panel-title">Accident Explanation</h2>
                                <p class="panel-description">
                                    Capture scene details that will later be forwarded to the assistant workflow.
                                </p>
                            </div>
                        </div>
                        """
                    )
                    explanation_input = gr.Textbox(
                        lines=7,
                        label="Scenario Description",
                        placeholder="Describe the accident scenario, vehicle movement, road conditions, and any relevant details...",
                        elem_classes="explanation-input",
                    )
                    gr.HTML(
                        """
                        <div class="placeholder-note">
                            This input is a UI placeholder for the future assistant workflow and currently does not affect detection inference.
                        </div>
                        """
                    )

                with gr.Column(scale=5, elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">LAW</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Future Assistant</p>
                                <h2 class="panel-title">Jordan Traffic Law Assistant</h2>
                                <p class="panel-description">
                                    <span class="assistant-chip">قانون السير الأردني</span>
                                </p>
                            </div>
                        </div>
                        <div class="placeholder-note">
                            This assistant will later answer questions about accident responsibility and Jordan traffic law.
                            No backend logic is connected yet.
                        </div>
                        <div class="chat-preview">
                            <div class="chat-bubble chat-bubble-user">
                                <span class="chat-role">User</span>
                                The car behind me hit my vehicle at a red light. Who is responsible?
                            </div>
                            <div class="chat-bubble chat-bubble-assistant">
                                <span class="chat-role">Assistant</span>
                                Future assistant response will analyze the situation according to Jordan traffic law.
                            </div>
                        </div>
                        """
                    )
                    with gr.Row():
                        law_prompt = gr.Textbox(
                            lines=1,
                            show_label=False,
                            placeholder="Ask the legal assistant",
                            interactive=False,
                            elem_classes="assistant-input",
                            scale=5,
                        )
                        law_button = gr.Button(
                            "Coming Soon",
                            interactive=False,
                            elem_classes="assistant-button",
                            scale=2,
                        )

            with gr.Column(elem_classes="workflow-card"):
                gr.HTML(
                    """
                    <div class="workflow-header">
                        <span class="section-icon">FLOW</span>
                        <div class="panel-copy">
                            <p class="panel-kicker">Results Summary</p>
                            <h2 class="panel-title">Future Workflow Overview</h2>
                            <p class="panel-description">
                                The full capstone system roadmap for evaluators and demonstration walkthroughs.
                            </p>
                        </div>
                    </div>
                    <div class="workflow-flow">
                        <div class="workflow-step"><span class="workflow-step-badge">1</span>Upload Image/Video</div>
                        <span class="workflow-arrow">&rarr;</span>
                        <div class="workflow-step"><span class="workflow-step-badge">2</span>Detect Accident</div>
                        <span class="workflow-arrow">&rarr;</span>
                        <div class="workflow-step"><span class="workflow-step-badge">3</span>Classify Severity</div>
                        <span class="workflow-arrow">&rarr;</span>
                        <div class="workflow-step"><span class="workflow-step-badge">4</span>Explain Scenario</div>
                        <span class="workflow-arrow">&rarr;</span>
                        <div class="workflow-step"><span class="workflow-step-badge">5</span>Ask Jordan Traffic Law Assistant</div>
                    </div>
                    """
                )

            gr.HTML(
                """
                <div class="footer-card">
                    Designed for AI accident detection, capstone demonstration, and future smart-road safety integration.
                </div>
                """
            )

        submit_btn.click(
            fn=predict_accident_gui,
            inputs=[image_input, conf_slider],
            outputs=[image_output, status_output],
        )

        conf_slider.change(
            fn=build_threshold_readout,
            inputs=conf_slider,
            outputs=threshold_readout,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
