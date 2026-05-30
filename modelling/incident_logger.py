import os
import csv
import json
import uuid
import cv2
import requests
from datetime import datetime
from pathlib import Path



# Webhook configuration
ENABLE_INCIDENT_WEBHOOK = True
INCIDENT_WEBHOOK_URL = "https://ounawa.app.n8n.cloud/webhook-test/accident-alert"
 
def generate_incident_id() -> str:
    """Generates a unique incident identifier."""
    return f"INC-{uuid.uuid4().hex[:8].upper()}"
 
def get_fake_location() -> tuple[str, str]:
    """Returns static fake GPS coordinates and PoC label for Amman, Jordan."""
    fake_location = "https://maps.google.com/?q=31.9539,35.9106"
    location_label = "Amman, Jordan - Fake PoC Location"
    return fake_location, location_label
 
def determine_escalation_status(severity: str) -> str:
    """
    Decides escalation status based on severity label.
    If severity contains severe, high, or critical, it is escalated.
    """
    if not severity:
        return "Not Escalated"
    normalized = str(severity).lower()
    if any(word in normalized for word in ["severe", "high", "critical"]):
        return "Escalated"
    return "Not Escalated"
 
def save_accident_frame(frame, incident_id: str) -> str:
    """
    Saves the best accident frame to C:\\Capstone\\incident_logs\\frames\\.
    Assumes frame is a BGR numpy array.
    """
    if frame is None:
        return ""
    try:
        output_dir = Path(r"C:\Capstone\incident_logs\frames")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"incident_{incident_id}.jpg"
       
        cv2.imwrite(str(file_path), frame)
        return str(file_path.resolve())
    except Exception as e:
        print(f"[Incident Logger] Error saving accident frame: {e}")
        return ""
 
def build_incident_payload(
    incident_id: str,
    severity: str,
    detection_confidence: float,
    classification_confidence: float | None,
    media_type: str,
    saved_accident_frame: str,
    webhook_status: str = "disabled"
) -> dict:
    """Builds the structured incident log payload dictionary."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    fake_loc, loc_label = get_fake_location()
    escalation = determine_escalation_status(severity)
   
    return {
        "incident_id": incident_id,
        "date": date_str,
        "time": time_str,
        "location": fake_loc,
        "location_label": loc_label,
        "severity": severity,
        "escalation_status": escalation,
        "detection_confidence": float(detection_confidence) if detection_confidence is not None else 0.0,
        "classification_confidence": float(classification_confidence) if classification_confidence is not None else None,
        "media_type": media_type,
        "saved_accident_frame": saved_accident_frame,
        "webhook_status": webhook_status
    }
 
def save_incident_log(payload: dict) -> None:
    """Saves the incident payload locally to JSONL and CSV files."""
    try:
        logs_dir = Path(r"C:\Capstone\incident_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
       
        # 1. Append to incidents.jsonl
        jsonl_path = logs_dir / "incidents.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
           
        # 2. Append to incidents.csv
        csv_path = logs_dir / "incidents.csv"
        fieldnames = [
            "incident_id", "date", "time", "location", "location_label",
            "severity", "escalation_status", "detection_confidence",
            "classification_confidence", "media_type", "saved_accident_frame",
            "webhook_status"
        ]
       
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(payload)
           
    except Exception as e:
        print(f"[Incident Logger] Error saving incident log locally: {e}")
 
def send_incident_webhook(payload: dict, image_path: str = None) -> str:
    """
    Safely sends form-data along with the image file (if available) to the webhook.
    Catches exceptions to ensure the pipeline never crashes if n8n is offline.
    """
    if not ENABLE_INCIDENT_WEBHOOK:
        return "disabled"
       
    try:
        # Convert all payload fields to string for standard form-data post
        form_data = {k: str(v) if v is not None else "" for k, v in payload.items()}

        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as img_file:
                # Force filename and mime-type so n8n guarantees it reads as a binary file
                files = {"image": (Path(image_path).name, img_file, "image/jpeg")}
                
                response = requests.post(
                    INCIDENT_WEBHOOK_URL,
                    data=form_data,
                    files=files,
                    timeout=10
                )

        else:
            response = requests.post(
                INCIDENT_WEBHOOK_URL,
                data=form_data,
                timeout=10
            )
           
        if response.status_code in [200, 201]:
            return "sent"
        else:
            print(f"[Incident Webhook] Webhook returned status code {response.status_code}: {response.text}")
            return f"failed (status {response.status_code})"
    except Exception as e:
        print(f"[Incident Webhook] Exception occurred while sending to webhook: {e}")
        return f"failed ({str(e)})"
 
def log_incident_if_confirmed(
    severity: str,
    detection_confidence: float,
    classification_confidence: float | None,
    media_type: str,
    frame=None
) -> dict:
    """
    Orchestrator function: generates ID, saves frame, sends webhook (if enabled),
    and appends payload to local logs.
    """
    incident_id = generate_incident_id()
   
    # Save frame first if available to get the file path
    saved_frame_path = ""
    if frame is not None:
        saved_frame_path = save_accident_frame(frame, incident_id)
       
    # Build payload template
    payload = build_incident_payload(
        incident_id=incident_id,
        severity=severity,
        detection_confidence=detection_confidence,
        classification_confidence=classification_confidence,
        media_type=media_type,
        saved_accident_frame=saved_frame_path,
        webhook_status="disabled"
    )
   
    # Attempt webhook dispatch
    if ENABLE_INCIDENT_WEBHOOK:
        webhook_status = send_incident_webhook(payload, saved_frame_path)
        payload["webhook_status"] = webhook_status
    else:
        payload["webhook_status"] = "disabled"
       
    # Save log locally
    save_incident_log(payload)
   
    return payload
 
