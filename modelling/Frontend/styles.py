from pathlib import Path
from html import escape

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
