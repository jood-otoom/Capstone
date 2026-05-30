import subprocess
import sys
import os
from pathlib import Path
import csv
from html import escape
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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

from ui_app.config import PROJECT_ROOT, AGENT_DIR, DETECTION_MODEL_DIR, CLASSIFICATION_MODEL_DIR
from ui_app.model_loader import find_best_model, find_best_pt
from ui_app.media_utils import get_media_label, convert_avi_to_mp4
from ui_app.styles import render_final_summary_html, severity_display_label, format_severity_label, build_pipeline_status_banner, build_status_banner, build_model_badge, build_threshold_readout
from ui_app.alerts import build_alert_banner, build_alert_controls, build_alert_signal, build_alert_controller_head
from ui_app.agent_service import APIKeyManager, api_key_manager, safe_agent_call, get_accident_agent
from ui_app.chat_service import render_chat_html, initiate_chat, generate_chat_reply
from ui_app.detection_pipeline import detect_accident_from_result, detect_accident_from_collection
from ui_app.combined_pipeline import AccidentSeverityPipeline, pipeline, BEST_MODEL_PATH, run_image_inference, run_video_inference, handle_image_upload, handle_video_upload, handle_image_clear, handle_video_clear, run_model_inference_flow, run_agent_analysis_flow

# Resolve dynamic project root path

# Dynamically add accident_agent to sys.path

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











# Model Directory Path Constants














# Initialize Unified Pipeline Globally



















# predict_accident_gui moved to combined_pipeline.py


















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

    /* STEP 1: Upload Section Cleanup */
    #evidence-upload {
        min-height: 90px !important;
        border: 1px dashed rgba(147, 197, 253, 0.6) !important;
        border-radius: 18px !important;
        background: rgba(248, 250, 252, 0.8) !important;
        padding: 8px !important;
        margin-bottom: 0 !important;
    }
    #evidence-upload .file-preview {
        display: none !important; /* Hide file list preview inside upload component */
    }
    #evidence-upload svg,
    #evidence-upload .icon-wrap,
    #evidence-upload .file-preview-holder svg {
        width: 28px !important;
        height: 28px !important;
        margin-bottom: 0 !important;
    }
    #evidence-upload .file-drop {
        min-height: 70px !important;
        padding: 6px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
    }
    #evidence-upload .file-drop span {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #1e40af !important;
    }
    
    #clear-evidence-btn {
        margin-top: 12px !important;
        background: #f1f5f9 !important;
        color: #475569 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 8px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-height: 38px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    #clear-evidence-btn:hover {
        background: #e2e8f0 !important;
        color: #0f172a !important;
        border-color: #94a3b8 !important;
    }

    /* STEP 2: Spacing & Whitespace collapsing */
    /* Do not hide parent detection containers; only hide empty spacer placeholders. */
    #emergency-alert-region,
    #alert-signal-region {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        display: none !important;
    }

    /* STEP 4: Chat display viewport scroll */
    #law-chatbot {
        max-height: 600px !important;
        overflow-y: auto !important;
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

            # Main Full-width Dashboard Row
            with gr.Row(equal_height=True, elem_classes="content-grid"):
                # Column 1: Step 1 (Upload Card) - scale=5
                with gr.Column(scale=5, elem_classes="panel-card upload-card"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">IN</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">Step 1: Upload Evidence</p>
                                <h2 class="panel-title">Accident Evidence</h2>
                                <p class="panel-description">
                                    Upload a road image or video to automatically run accident detection.
                                </p>
                            </div>
                        </div>
                        """
                    )
                    
                    evidence_upload = gr.File(
                        label="Upload Accident Evidence (Image or Video)",
                        file_types=["image", "video"],
                        type="filepath",
                        elem_classes="mode-selector",
                        elem_id="evidence-upload"
                    )
                    
                    # IMAGE PREVIEW GROUP
                    with gr.Column(visible=True) as image_input_group:
                        image_input = gr.Image(
                            type="filepath",
                            label="Road Image Preview",
                            elem_classes="image-shell",
                            elem_id="image-preview-input",
                            interactive=False,
                        )
                    
                    # VIDEO PREVIEW GROUP
                    with gr.Column(visible=True) as video_input_group:
                        video_input = gr.Video(
                            label="Road Video Preview",
                            elem_classes="image-shell",
                            elem_id="video-preview-input",
                            interactive=False,
                        )
                    
                    clear_btn = gr.Button(
                        "Upload New Evidence / Reset",
                        visible=False,
                        elem_classes="assistant-button",
                        elem_id="clear-evidence-btn"
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
                            type="filepath",
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
                            placeholder="Upload accident evidence to start AI analysis...",
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

            gr.HTML(
                """
                <div class="footer-card">
                    Designed for AI accident detection, capstone demonstration, and future smart-road safety integration.
                </div>
                """
            )

        # Unified Upload Chained Event Handlers
        upload_event = evidence_upload.upload(
            fn=run_model_inference_flow,
            inputs=evidence_upload,
            outputs=[
                image_input,
                video_input,
                image_input_group,
                video_input_group,
                image_output_group,
                video_output_group,
                image_output,
                video_output,
                status_output,
                video_status_output,
                alert_banner_output,
                alert_signal_output,
                chatbot,
                law_prompt,
                law_button,
                chat_history,
                evidence_upload,
                clear_btn
            ]
        )
        
        upload_event.then(
            fn=run_agent_analysis_flow,
            inputs=[evidence_upload, chat_history, agent_state],
            outputs=[
                chatbot,
                law_prompt,
                law_button,
                agent_state,
                chat_history
            ]
        )

        def handle_evidence_clear():
            status_banner = build_status_banner(
                title="Awaiting inference",
                message="Upload a road image or video file to automatically start accident detection.",
                tone="neutral",
                icon="&#9711;",
            )
            video_status_banner = build_status_banner(
                title="Awaiting video inference",
                message="Upload a road video file to automatically start accident detection.",
                tone="neutral",
                icon="&#8682;",
            )
            return (
                None, None,
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False),
                None, None,
                status_banner, video_status_banner,
                build_alert_banner("standby", "image"),
                build_alert_signal(False, "idle", "image"),
                render_chat_html([]),
                gr.update(interactive=False, placeholder="Upload accident evidence to start AI analysis..."),
                gr.update(interactive=False, value="Locked"),
                [],
                gr.update(visible=True, value=None),  # Show evidence_upload and clear its value
                gr.update(visible=False)              # Hide clear_btn
            )

        evidence_upload.clear(
            fn=handle_evidence_clear,
            outputs=[
                image_input,
                video_input,
                image_input_group,
                video_input_group,
                image_output_group,
                video_output_group,
                image_output,
                video_output,
                status_output,
                video_status_output,
                alert_banner_output,
                alert_signal_output,
                chatbot,
                law_prompt,
                law_button,
                chat_history,
                evidence_upload,
                clear_btn
            ]
        )

        clear_btn.click(
            fn=handle_evidence_clear,
            outputs=[
                image_input,
                video_input,
                image_input_group,
                video_input_group,
                image_output_group,
                video_output_group,
                image_output,
                video_output,
                status_output,
                video_status_output,
                alert_banner_output,
                alert_signal_output,
                chatbot,
                law_prompt,
                law_button,
                chat_history,
                evidence_upload,
                clear_btn
            ]
        )

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

    demo._deprecated_theme = blue_theme
    demo._deprecated_css = custom_css
    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue()  # Enable queuing for robust generator/yield processing!
    try:
        app.launch(
            share=True,
            server_name="127.0.0.1",
            server_port=7860,
            head=build_alert_controller_head(),
        )
    except OSError:
        print("Port 7860 is occupied. Launching on an automatically allocated free port...")
        app.launch(
            share=True,
            server_name="127.0.0.1",
            head=build_alert_controller_head(),
        )