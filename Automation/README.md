# AcciEye | Automation & Telemetry System

This directory contains the **Automation and Telemetry Module** of the AcciEye platform. The module implements a non-blocking webhook and event-driven pipeline that coordinates evidence archiving, logging, and emergency dispatch alerts.

---

## 📖 Module Overview

High-fidelity computer vision and legal reasoning are most useful when paired with immediate automated notification. The AcciEye automation system bridges deep learning inference with real-world operations by forwarding confirmed incident data through a production-ready **n8n** automation workflow.

When the visual pipeline detects an accident, the incident metadata and the best accident image frame are captured. An asynchronous payload is sent to the automation system, which executes three downstream actions:
1. **Cloud Archiving**: Automatically categorizes and uploads accident photos to dedicated Google Drive folders based on severity.
2. **Master Incident Registry**: Logs detailed telemetry records into a centralized Google Sheets spreadsheet for audit trails.
3. **Emergency Dispatch Broadcast**: Triggers instant Telegram alerts with structured markdown details and inline photo previews to emergency response channels if a severe accident is detected.

---

## 📂 Production Automation Config

* `Automation/AcciEye_Automation_System.json`: The complete, production-ready n8n workflow configuration file. It contains the coordinate system, node configurations, parameter mappings, Google OAuth2 schemas, and Telegram alert formatting rules. This file can be imported directly into any n8n instance.

---

## ⚙️ n8n Workflow Nodes

The workflow comprises six distinct nodes working in a directed acyclic graph:

### 1. Webhook Node (`POST /accident-alert`)
* **Role**: Acts as the system entry point.
* **Mechanism**: Listens for HTTP POST requests from the front-end pipeline. The request uses `multipart/form-data` to submit structured incident JSON alongside the binary accident image.
* **Configured Path**: `accident-alert`

### 2. Upload File Node (Google Drive)
* **Role**: Archives evidence photo attachments.
* **Mechanism**: Reads the incoming binary image buffer and uploads it to a target Google Drive.
* **Dynamic Routing**: Uses conditional folders to separate evidence based on severity:
  * `Severe`: Routed to folder ID `1VDYiyod63ECH49ivM5hBF_oiQxLm8F2t`
  * `Moderate`: Routed to folder ID `11oE-c7X5fXF4J0K8bmraz8jINpnDe7aK`
  * `Minor`: Routed to folder ID `1TNCD8XoN9Mty4VIw9gtOrZN4SIc509x4`

### 3. Append Row Node (Google Sheets)
* **Role**: Maintains a secure audit trail of all detected accidents.
* **Mechanism**: Appends a new logging row inside the `AcciEye_Incident_Logs` Google Sheet (Spreadsheet ID: `19Ba1uLO9ADE_-UONdY1tK4Ce_Xv0uXU1BGbpSiTRXhY`).
* **Logged Fields**:
  * `incident_id`: Structured unique ID (e.g. `INC-F89A21C0`).
  * `date` & `time`: Time stamps of detection.
  * `location`: Google Maps coordinates (e.g. fake PoC location in Amman, Jordan).
  * `location_label`: Textual localization.
  * `severity`: Categorized severity (`severe-accident`, `moderate-accident`, `no-accident`).
  * `photo_storage_category`: Severity-mapped folder category.
  * `accident_photo_link`: Public sharing URL returned by the Google Drive upload node.
  * `escalation_status`: Deemed `Escalated` for high-severity issues, or `Not Escalated` otherwise.

### 4. Merge Node
* **Role**: Synchronizes parallel asynchronous tasks.
* **Mechanism**: Combines the output of the local webhook node with the Google Drive webViewLink from the upload node to ensure both are fully processed before the conditional filter evaluates.

### 5. If Node (Severity Evaluator)
* **Role**: Evaluates the escalation criteria.
* **Mechanism**: Evaluates whether the incoming `severity` field is equal to `severe-accident`. If true, the workflow proceeds to the emergency alert node. If false, the workflow terminates safely.

### 6. Send Photo Message Node (Telegram)
* **Role**: Immediate emergency notification broadcast.
* **Mechanism**: Contacts the Telegram Bot API to dispatch a high-priority photo alert containing the binary evidence file and an inline markdown caption.
* **Alert Caption Format**:
  ```text
  🚨 SEVERE TRAFFIC ACCIDENT DETECTED
  
  🆔 Incident ID: {incident_id}
  ⚠ Severity: {severity}
  🕒 Time: {time}
  📍 Location: {location}
  📍 Location Label: {location_label}
  📂 Photo Link: {google_drive_webview_link}
  ```

---

## 🛠️ Pipeline Integration

The frontend app triggers this automation pipeline through `modelling/incident_logger.py`. 

The core orchestration function is `log_incident_if_confirmed()`:

```python
import requests

def send_incident_webhook(payload: dict, image_path: str = None) -> str:
    # Resolves ACCIDENT_ALERT_WEBHOOK_URL from .env
    url = "https://your-n8n-instance.com/webhook/accident-alert"
    
    form_data = {k: str(v) for k, v in payload.items()}
    
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            files = {"image": (os.path.basename(image_path), img_file, "image/jpeg")}
            response = requests.post(url, data=form_data, files=files, timeout=10)
    else:
        response = requests.post(url, data=form_data, timeout=10)
        
    return "sent" if response.status_code in [200, 201] else "failed"
```

This request executes in a safe, non-blocking manner. If the n8n server is offline or experiencing network drops, the Gradio user interface will log the failure locally to disk without interrupting the user experience.

---

## 🚀 Deployment & Importing into n8n

To deploy this automation flow:
1. Open your self-hosted or cloud **n8n** dashboard.
2. Click on **Workflows** -> **New Workflow**.
3. In the top-right corner, click on the **Settings Menu (three dots)** -> **Import from File**.
4. Select the `Automation/AcciEye_Automation_System.json` file.
5. Configure credentials for:
   * **Google Drive OAuth2 API** (to link the archive folders).
   * **Google Sheets OAuth2 API** (to link the logging sheet).
   * **Telegram API** (using your custom Bot Token and Target Chat ID).
6. Click **Save** and toggle the workflow status to **Active**.
7. Copy the production Webhook URL from the Webhook node and paste it into your `.env` file under `ACCIDENT_ALERT_WEBHOOK_URL`.
