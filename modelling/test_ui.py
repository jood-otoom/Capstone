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
    media_label = get_media_label(source)

    if state == "active":
        return f"""
        <section class="emergency-alert-card state-active" role="alert" aria-live="assertive">
            <div class="emergency-alert-icon">&#9888;</div>
            <div class="emergency-alert-copy">
                <div class="emergency-alert-topline">
                    <span class="emergency-alert-eyebrow">Emergency Alert</span>
                    <div class="emergency-alert-badges">
                        <span class="emergency-alert-badge badge-danger">Alert Active</span>
                        <span class="emergency-alert-badge badge-source">{escape(media_label)}</span>
                    </div>
                </div>
                <h3 class="emergency-alert-title">ACCIDENT DETECTED</h3>
                <p class="emergency-alert-message">
                    Immediate attention required. Review the detection result and take action.
                </p>
                <div class="emergency-alert-actions" style="margin-top: 14px;">
                    <button id="stop-alert-btn" type="button" class="stop-alert-button">
                        &#9208; Stop Alert Sound
                    </button>
                </div>
            </div>
        </section>
        """

    if state == "clear":
        return f"""
        <section class="emergency-alert-card state-clear" aria-live="polite">
            <div class="emergency-alert-icon">&#10003;</div>
            <div class="emergency-alert-copy">
                <div class="emergency-alert-topline">
                    <span class="emergency-alert-eyebrow">Alert Cleared</span>
                    <div class="emergency-alert-badges">
                        <span class="emergency-alert-badge badge-clear">No Alarm</span>
                        <span class="emergency-alert-badge badge-source">{escape(media_label)}</span>
                    </div>
                </div>
                <h3 class="emergency-alert-title">No active accident alert</h3>
                <p class="emergency-alert-message">
                    The final displayed result did not confirm an accident, so the alert remains off.
                </p>
            </div>
        </section>
        """

    if state == "processing":
        return f"""
        <section class="emergency-alert-card state-processing" aria-live="polite">
            <div class="emergency-alert-icon">&#9711;</div>
            <div class="emergency-alert-copy">
                <div class="emergency-alert-topline">
                    <span class="emergency-alert-eyebrow">Alert Monitoring</span>
                    <div class="emergency-alert-badges">
                        <span class="emergency-alert-badge badge-standby">Processing</span>
                        <span class="emergency-alert-badge badge-source">{escape(media_label)}</span>
                    </div>
                </div>
                <h3 class="emergency-alert-title">Detection in progress</h3>
                <p class="emergency-alert-message">
                    The emergency alert will activate only if the final displayed result confirms an accident.
                </p>
            </div>
        </section>
        """

    if state == "armed":
        return f"""
        <section class="emergency-alert-card state-standby" aria-live="polite">
            <div class="emergency-alert-icon">&#9889;</div>
            <div class="emergency-alert-copy">
                <div class="emergency-alert-topline">
                    <span class="emergency-alert-eyebrow">Alert Monitoring</span>
                    <div class="emergency-alert-badges">
                        <span class="emergency-alert-badge badge-standby">Armed</span>
                        <span class="emergency-alert-badge badge-source">{escape(media_label)}</span>
                    </div>
                </div>
                <h3 class="emergency-alert-title">Alert ready</h3>
                <p class="emergency-alert-message">
                    Alert features are armed and will stay silent unless the final displayed result confirms an accident.
                </p>
            </div>
        </section>
        """

    if state == "error":
        return f"""
        <section class="emergency-alert-card state-error" aria-live="polite">
            <div class="emergency-alert-icon">&#9888;</div>
            <div class="emergency-alert-copy">
                <div class="emergency-alert-topline">
                    <span class="emergency-alert-eyebrow">Alert Reset</span>
                    <div class="emergency-alert-badges">
                        <span class="emergency-alert-badge badge-warning">No Alarm</span>
                        <span class="emergency-alert-badge badge-source">{escape(media_label)}</span>
                    </div>
                </div>
                <h3 class="emergency-alert-title">Alert inactive</h3>
                <p class="emergency-alert-message">
                    The alert was reset because inference did not complete with a confirmed accident result.
                </p>
            </div>
        </section>
        """

    return f"""
    <section class="emergency-alert-card state-standby" aria-live="polite">
        <div class="emergency-alert-icon">&#9711;</div>
        <div class="emergency-alert-copy">
            <div class="emergency-alert-topline">
                <span class="emergency-alert-eyebrow">Alert Standby</span>
                <div class="emergency-alert-badges">
                    <span class="emergency-alert-badge badge-standby">Monitoring Off</span>
                    <span class="emergency-alert-badge badge-source">{escape(media_label)}</span>
                </div>
            </div>
            <h3 class="emergency-alert-title">Awaiting detection result</h3>
            <p class="emergency-alert-message">
                Upload media and run inference. The emergency alert will activate only for a confirmed accident result.
            </p>
        </div>
    </section>
    """


def build_alert_controls() -> str:
    return ""


def build_alert_signal(alert_active: bool, status: str, source: str) -> str:
    title = "Accident Detected" if alert_active else "Alert Standby"
    message = "The uploaded media was classified as an accident." if alert_active else ""
    return f"""
    <div
        class="alert-signal-data"
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
          active: false,
          status: "idle",
          source: "image",
          title: "Accident Detected",
          message: "The uploaded media was classified as an accident.",
        },
        audioSupported: Boolean(window.AudioContext || window.webkitAudioContext),
        notificationSupported: typeof window.Notification !== "undefined",
        soundEnabled: false,
        mutedByUser: false,
        hasUserInteracted: false,
        audioContext: null,
        alarmTimer: null,
        signalObserver: null,
        signalObserverHost: null,
        attachTimer: null,
        handlersBound: false,
        lastNotificationKey: "",

        init() {
          this.bindGlobalHandlers();
          this.attachSignalObserver();
          this.syncFromSignal();
          this.updateUI();
        },

        enableSoundAndNotifications() {
          this.soundEnabled = true;
          this.mutedByUser = false;
          this.registerInteraction();
          this.ensureAudioContext();

          if (this.notificationSupported && Notification.permission === "default") {
            Notification.requestPermission()
              .then((permission) => {
                this.updateUI();
                if (permission === "granted") {
                  this.lastNotificationKey = "";
                  this.notify();
                }
              })
              .catch(() => {
                this.updateUI();
              });
          } else {
            this.updateUI();
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

            if (button.id === "stop-alert-btn") {
              event.preventDefault();
              this.stopAlertByUser();
              this.updateUI();
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
              active: false,
              status: "idle",
              source: "image",
              title: "Accident Detected",
              message: "The uploaded media was classified as an accident.",
            };
          }

          return {
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
          this.state = next;

          if (!next.active) {
            if (wasActive || previousStatus !== next.status) {
              this.resetRuntime();
            } else {
              this.applyClasses();
              this.updateUI();
            }
            return;
          }

          if (!wasActive) {
            this.activateAlert();
            return;
          }

          this.applyClasses();
          this.updateUI();
        },

        activateAlert() {
          this.applyClasses();
          this.updateUI();
          this.triggerVibration();

          if (this.soundEnabled && !this.mutedByUser && this.hasUserInteracted) {
            this.startAlarm();
          }

          this.notify();
        },

        resetRuntime() {
          this.stopAlarm();
          this.stopVibration();
          this.mutedByUser = false;
          this.lastNotificationKey = "";
          this.applyClasses();
          this.updateUI();
        },

        startAlarm() {
          if (!this.state.active || !this.soundEnabled || this.mutedByUser || !this.hasUserInteracted) return;
          if (!this.ensureAudioContext()) return;
          if (this.alarmTimer) return;

          this.playAlarmPattern();
          this.alarmTimer = window.setInterval(() => {
            this.playAlarmPattern();
          }, 1900);
        },

        stopAlarm() {
          if (this.alarmTimer) {
            window.clearInterval(this.alarmTimer);
            this.alarmTimer = null;
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

        enableSound() {
          this.soundEnabled = true;
          this.mutedByUser = false;
          this.registerInteraction();
          this.ensureAudioContext();

          if (this.state.active) {
            this.startAlarm();
          }

          this.applyClasses();
          this.updateUI();
        },

        stopAlertByUser() {
          this.stopAlarm();
          this.stopVibration();

          if (this.state.active) {
            this.mutedByUser = true;
          }

          this.applyClasses();
          this.updateUI();
        },

        enableBrowserAlerts() {
          this.registerInteraction();
          if (!this.notificationSupported) {
            this.updateUI();
            return;
          }

          if (Notification.permission === "granted") {
            this.updateUI();
            this.notify();
            return;
          }

          Notification.requestPermission()
            .then((permission) => {
              this.updateUI();
              if (permission === "granted") {
                this.lastNotificationKey = "";
                this.notify();
              }
            })
            .catch(() => {
              this.updateUI();
            });
        },

        applyClasses() {
          const body = document.body;
          if (!body) return;

          body.classList.toggle("accident-alert-active", this.state.active);
          body.classList.toggle("accident-alert-muted", this.state.active && this.mutedByUser);
          body.classList.toggle("accident-alert-source-image", this.state.active && this.state.source === "image");
          body.classList.toggle("accident-alert-source-video", this.state.active && this.state.source === "video");
        },

        setPill(id, text, tone) {
          const pill = document.getElementById(id);
          if (!pill) return;
          pill.textContent = text;
          pill.className = `alert-pill ${tone}`;
        },

        updateUI() {
          const stopButton = document.getElementById("stop-alert-btn");
          if (stopButton) {
            if (this.mutedByUser) {
              stopButton.innerHTML = "&#9208; Alert Sound Muted";
              stopButton.style.backgroundColor = "#4b5563";
              stopButton.style.borderColor = "#374151";
              stopButton.disabled = true;
            } else {
              stopButton.innerHTML = "&#9208; Stop Alert Sound";
              stopButton.disabled = !this.state.active;
            }
          }
        },

        destroy() {
          this.stopAlarm();
          this.stopVibration();
          if (this.signalObserver) {
            this.signalObserver.disconnect();
          }
          window.clearTimeout(this.attachTimer);
          if (this.audioContext && typeof this.audioContext.close === "function" && this.audioContext.state !== "closed") {
            this.audioContext.close().catch(() => {});
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
    if model is None:
        return input_image, build_status_banner(
            title="Model unavailable",
            message="No model was loaded. Please check the configured model paths.",
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

    # Inference
    results = model(input_image, conf=conf_threshold)

    # Render Bounding Boxes
    rendered_image = results[0].plot()

    accident_detected = detect_accident_from_collection(results)

    if accident_detected:
        status_html = build_status_banner(
            title="Accident detected",
            message="The selected model found at least one accident-related detection in the uploaded image.",
            tone="alert",
            icon="&#9888;",
        )
        alert_html = build_alert_banner("active", "image")
        alert_signal = build_alert_signal(True, "accident", "image")
    else:
        status_html = build_status_banner(
            title="No accident detected",
            message="Inference completed and no accident class was detected above the selected confidence threshold.",
            tone="safe",
            icon="&#10003;",
        )
        alert_html = build_alert_banner("clear", "image")
        alert_signal = build_alert_signal(False, "clear", "image")

    return rendered_image, status_html, alert_html, alert_signal


def handle_image_upload(image: np.ndarray | None):
    if image is None:
        return build_status_banner(
            title="No image uploaded",
            message="Please upload a valid road image.",
            tone="neutral",
            icon="&#8682;",
        ), None, build_alert_banner("standby", "image"), build_alert_signal(False, "idle", "image")
    return build_status_banner(
        title="Image uploaded successfully",
        message="Review the preview and click Run Inference to detect accidents.",
        tone="safe",
        icon="&#10003;",
    ), None, build_alert_banner("armed", "image"), build_alert_signal(False, "armed", "image")


def handle_video_upload(video_path: str | None):
    if not video_path:
        return build_status_banner(
            title="No video uploaded",
            message="Please upload a road video for testing.",
            tone="neutral",
            icon="&#8682;",
        ), None, gr.update(visible=True), build_alert_banner("standby", "video"), build_alert_signal(False, "idle", "video")
    
    # Validate file extension
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(video_path).suffix.lower()
    if ext not in allowed_exts:
        return build_status_banner(
            title="Unsupported video format",
            message=f"The uploaded format {ext} is not supported. Please use MP4, AVI, MOV, or MKV.",
            tone="alert",
            icon="&#10005;",
        ), None, gr.update(visible=True), build_alert_banner("error", "video"), build_alert_signal(False, "error", "video")
    
    return build_status_banner(
        title="Video uploaded successfully",
        message="Review the preview and click Run Video Inference to process.",
        tone="safe",
        icon="&#10003;",
    ), None, gr.update(visible=True), build_alert_banner("armed", "video"), build_alert_signal(False, "armed", "video")


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
    )


def run_image_inference(input_image: np.ndarray, conf_threshold: float):
    """
    Handles image inference input from the UI.
    This function should not modify models.
    """
    return predict_accident_gui(input_image, conf_threshold)


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


def run_video_inference(video_path: str | None, confidence_threshold: float):
    """
    Handles video inference input from the UI.
    This function should not train or modify models.
    It should only pass the uploaded video to the already-loaded best model
    if video inference is supported.
    """
    import traceback

    print("[VIDEO] run_video_inference started")
    print(f"[VIDEO] received video_path: {video_path}")
    print(f"[VIDEO] confidence_threshold: {confidence_threshold}")

    if model is None:
        print("[VIDEO] model is None")
        yield None, build_status_banner(
            title="Model unavailable",
            message="No model was loaded. Please check the configured model paths.",
            tone="alert",
            icon="&#10005;",
        ), gr.update(visible=True), build_alert_banner("error", "video"), build_alert_signal(False, "error", "video")
        return

    if not video_path:
        print("[VIDEO] no video_path provided")
        yield None, build_status_banner(
            title="No video uploaded",
            message="Upload a road video to run accident detection inference.",
            tone="neutral",
            icon="&#8682;",
        ), gr.update(visible=True), build_alert_banner("standby", "video"), build_alert_signal(False, "idle", "video")
        return

    video_file = Path(video_path)
    print(f"[VIDEO] video_path exists: {video_file.exists()}")
    print(f"[VIDEO] video_path suffix: {video_file.suffix}")

    # Validate video format
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if video_file.suffix.lower() not in allowed_exts:
        print(f"[VIDEO] unsupported format: {video_file.suffix}")
        yield None, build_status_banner(
            title="Unsupported video format",
            message=f"The uploaded format {video_file.suffix} is not supported. Please use MP4, AVI, MOV, or MKV.",
            tone="alert",
            icon="&#10005;",
        ), gr.update(visible=True), build_alert_banner("error", "video"), build_alert_signal(False, "error", "video")
        return

    # Yield processing status
    yield None, build_status_banner(
        title="Processing video...",
        message="Processing video frame by frame. This may take time depending on video length and device. Please wait...",
        tone="neutral",
        icon="&#9711;",
    ), gr.update(visible=True), build_alert_banner("processing", "video"), build_alert_signal(False, "processing", "video")

    try:
        run_id = str(int(time.time()))
        project_dir = PROJECT_ROOT / "runs" / "detect"
        name_dir = f"video_predictions_{run_id}"
        out_dir = project_dir / name_dir

        print(f"[VIDEO] output directory path: {out_dir}")
        print("[VIDEO] calling model.predict() now...")

        # Run inference using the natively supported YOLO predict function with stride
        results = model.predict(
            source=video_path,
            conf=confidence_threshold,
            save=True,
            project=str(project_dir),
            name=name_dir,
            exist_ok=True,
            imgsz=640,
            vid_stride=3,
        )

        print("[VIDEO] model.predict() finished")
        accident_detected = detect_accident_from_collection(results)
        print(f"[VIDEO] accident_detected in collection: {accident_detected}")

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

        if preview_path:
            if accident_detected:
                yield preview_path, build_status_banner(
                    title="Accident detected in video",
                    message=f"Video processing finished and the uploaded media was classified as an accident. Annotated MP4 output loaded below. File saved to: {preview_path}",
                    tone="alert",
                    icon="&#9888;",
                ), gr.update(visible=False), build_alert_banner("active", "video"), build_alert_signal(True, "accident", "video")
            else:
                yield preview_path, build_status_banner(
                    title="No accident detected in video",
                    message=f"Video processing finished and no accident class was detected above the selected confidence threshold. Annotated MP4 output loaded below. File saved to: {preview_path}",
                    tone="safe",
                    icon="&#10003;",
                ), gr.update(visible=False), build_alert_banner("clear", "video"), build_alert_signal(False, "clear", "video")
        elif conversion_failed:
            if accident_detected:
                yield None, build_status_banner(
                    title="Accident detected in video (preview unavailable)",
                    message=f"Video processing finished and the uploaded media was classified as an accident. The annotated AVI was saved successfully, but could not be converted for browser preview: {output_video_path}",
                    tone="alert",
                    icon="&#9888;",
                ), gr.update(visible=True), build_alert_banner("active", "video"), build_alert_signal(True, "accident", "video")
            else:
                yield None, build_status_banner(
                    title="No accident detected in video (preview unavailable)",
                    message=f"Video processing finished and no accident class was detected. The annotated AVI was saved successfully, but could not be converted for browser preview: {output_video_path}",
                    tone="safe",
                    icon="&#10003;",
                ), gr.update(visible=True), build_alert_banner("clear", "video"), build_alert_signal(False, "clear", "video")
        else:
            if accident_detected:
                yield None, build_status_banner(
                    title="Accident detected in video",
                    message="Video inference completed and the uploaded media was classified as an accident, but the annotated preview file could not be located. Check the runs output folder.",
                    tone="alert",
                    icon="&#9888;",
                ), gr.update(visible=True), build_alert_banner("active", "video"), build_alert_signal(True, "accident", "video")
            else:
                yield None, build_status_banner(
                    title="No accident detected in video",
                    message="Video inference completed and no accident class was detected above the selected confidence threshold, but the annotated preview file could not be located. Check the runs output folder.",
                    tone="safe",
                    icon="&#10003;",
                ), gr.update(visible=True), build_alert_banner("clear", "video"), build_alert_signal(False, "clear", "video")

    except Exception as e:
        print(f"[VIDEO] ERROR: {str(e)}")
        traceback.print_exc()
        yield None, build_status_banner(
            title="Video processing failed",
            message=f"Error: {str(e)}. Check the terminal logs for full traceback.",
            tone="alert",
            icon="&#10005;",
        ), gr.update(visible=True), build_alert_banner("error", "video"), build_alert_signal(False, "error", "video")


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
        --alert-pulse-shadow: rgba(220, 38, 38, 0.28);
        --alert-ring-soft: rgba(220, 38, 38, 0.16);
        --alert-ring-strong: rgba(220, 38, 38, 0.30);
        --warning-bg: #fff7ed;
        --warning-border: #fdba74;
        --warning-text: #9a3412;
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

    .mode-selector {
        background: rgba(37, 99, 235, 0.04) !important;
        border: 1px solid rgba(147, 197, 253, 0.3) !important;
        border-radius: 16px !important;
        padding: 6px !important;
        margin-bottom: 20px !important;
    }

    .content-grid {
        gap: 24px;
    }

    .panel-card {
        padding: 28px;
        min-width: 0;
        transition: box-shadow 0.28s ease, border-color 0.28s ease;
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
        transition: box-shadow 0.28s ease, border-color 0.28s ease;
    }

    #alert-signal-region {
        display: none;
    }

    .emergency-alert-card {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 18px;
        padding: 22px 24px;
        border-radius: 22px;
        border: 1px solid transparent;
        transition: box-shadow 0.28s ease, border-color 0.28s ease, background 0.28s ease;
    }

    .emergency-alert-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        flex: 0 0 48px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.78);
        font-size: 1.18rem;
        font-weight: 900;
    }

    .emergency-alert-copy {
        min-width: 0;
    }

    .emergency-alert-topline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px 12px;
    }

    .emergency-alert-eyebrow {
        color: inherit;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .emergency-alert-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .emergency-alert-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 30px;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .badge-danger {
        background: rgba(185, 28, 28, 0.12);
        border-color: rgba(185, 28, 28, 0.18);
        color: #991b1b;
    }

    .badge-clear {
        background: rgba(22, 163, 74, 0.10);
        border-color: rgba(22, 163, 74, 0.18);
        color: var(--safe-text);
    }

    .badge-standby,
    .badge-source {
        background: rgba(37, 99, 235, 0.08);
        border-color: rgba(147, 197, 253, 0.42);
        color: var(--primary-dark);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.25);
        color: #b45309;
    }

    .emergency-alert-title {
        margin: 10px 0 0;
        font-size: 1.42rem;
        font-weight: 900;
        letter-spacing: -0.02em;
    }

    .emergency-alert-message {
        margin: 10px 0 0;
        font-size: 0.97rem;
        line-height: 1.68;
    }

    .state-active {
        background: #fee2e2 !important;
        border: 2px solid #ef4444 !important;
        color: #7f1d1d !important;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.15) !important;
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

    .state-active .badge-danger {
        background: #fca5a5 !important;
        border: 1px solid #ef4444 !important;
        color: #7f1d1d !important;
        font-weight: 800 !important;
    }

    .state-active .badge-source {
        background: #bfdbfe !important;
        border: 1px solid #3b82f6 !important;
        color: #1e3a8a !important;
        font-weight: 800 !important;
    }

    .stop-alert-button {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: #dc2626 !important;
        color: #ffffff !important;
        border: 1px solid #b91c1c !important;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
        transition: background-color 0.2s, transform 0.2s;
        margin-top: 10px;
    }
    .stop-alert-button:hover {
        background: #b91c1c !important;
        transform: translateY(-1px);
    }
    .stop-alert-button:active {
        transform: translateY(0);
    }

    /* HIGH-CONTRAST FOR ALL OTHER ALERT BANNER STATES (NAVY BLUE & BLACK FOR STANDBY/PROCESSING) */
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
    .state-clear .badge-clear {
        background: #a7f3d0 !important;
        border: 1px solid #10b981 !important;
        color: #064e3b !important;
        font-weight: 800 !important;
    }
    .state-clear .badge-source {
        background: #a7f3d0 !important;
        border: 1px solid #10b981 !important;
        color: #064e3b !important;
        font-weight: 800 !important;
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
        color: #1e3a8a !important; /* Navy Blue text */
    }
    .state-standby .emergency-alert-icon,
    .state-processing .emergency-alert-icon {
        background: #bfdbfe !important;
        color: #1e3a8a !important;
    }
    .state-standby .badge-standby,
    .state-processing .badge-standby {
        background: #bfdbfe !important;
        border: 1px solid #93c5fd !important;
        color: #1e3a8a !important;
        font-weight: 800 !important;
    }
    .state-standby .badge-source,
    .state-processing .badge-source {
        background: #bfdbfe !important;
        border: 1px solid #3b82f6 !important;
        color: #1e3a8a !important;
        font-weight: 800 !important;
    }

    .state-error {
        background: #ffedd5 !important;
        border: 1px solid #f97316 !important;
        color: #7c2d12 !important;
    }
    .state-error .emergency-alert-title,
    .state-error .emergency-alert-message,
    .state-error .emergency-alert-eyebrow {
        color: #7c2d12 !important;
    }
    .state-error .emergency-alert-icon {
        background: #fed7aa !important;
        color: #7c2d12 !important;
    }
    .state-error .badge-warning {
        background: #fed7aa !important;
        border: 1px solid #f97316 !important;
        color: #7c2d12 !important;
        font-weight: 800 !important;
    }
    .state-error .badge-source {
        background: #fed7aa !important;
        border: 1px solid #f97316 !important;
        color: #7c2d12 !important;
        font-weight: 800 !important;
    }

    .alert-controls-head {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 12px 16px;
        margin-bottom: 16px;
    }

    .alert-controls-kicker {
        margin: 0 0 6px;
        color: var(--primary-dark);
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .alert-controls-title {
        margin: 0;
        color: var(--text-primary);
        font-size: 1.08rem;
        font-weight: 800;
    }

    .alert-controls-chip {
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.08);
        color: var(--primary-dark);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .alert-controls-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }

    .alert-control-stat {
        display: grid;
        gap: 8px;
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(226, 232, 240, 0.88);
    }

    .alert-control-label {
        color: var(--text-muted);
        font-size: 0.84rem;
        font-weight: 700;
    }

    .alert-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: fit-content;
        min-height: 30px;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .pill-standby {
        background: rgba(37, 99, 235, 0.08);
        border-color: rgba(147, 197, 253, 0.42);
        color: var(--primary-dark);
    }

    .pill-success {
        background: rgba(22, 163, 74, 0.10);
        border-color: rgba(134, 239, 172, 0.48);
        color: var(--safe-text);
    }

    .pill-warning {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(251, 191, 36, 0.35);
        color: #b45309;
    }

    .pill-active {
        background: rgba(220, 38, 38, 0.12);
        border-color: rgba(248, 113, 113, 0.35);
        color: var(--alert-text);
    }

    .pill-muted {
        background: rgba(71, 85, 105, 0.12);
        border-color: rgba(148, 163, 184, 0.34);
        color: #334155;
    }

    .alert-controls-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }

    .alert-control-button {
        min-height: 46px;
        padding: 0 16px;
        border-radius: 14px;
        font-family: inherit;
        font-size: 0.94rem;
        font-weight: 800;
        border: 1px solid transparent;
        cursor: pointer;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }

    .alert-control-button:hover:not(:disabled) {
        transform: translateY(-1px);
    }

    .alert-control-button:disabled {
        cursor: not-allowed;
        opacity: 0.64;
    }

    .button-primary {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #ffffff;
        box-shadow: 0 14px 24px rgba(37, 99, 235, 0.18);
    }

    .button-secondary {
        background: rgba(255, 255, 255, 0.96);
        border-color: rgba(148, 163, 184, 0.42);
        color: var(--text-primary);
    }

    .alert-controls-note {
        margin: 14px 0 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    @keyframes alertPulse {
        0%, 100% {
            box-shadow:
                0 0 0 0 rgba(220, 38, 38, 0.16),
                0 18px 40px rgba(220, 38, 38, 0.14);
        }
        50% {
            box-shadow:
                0 0 0 10px rgba(220, 38, 38, 0.08),
                0 22px 48px rgba(220, 38, 38, 0.22);
        }
    }

    @keyframes alertBorderPulse {
        0%, 100% {
            box-shadow:
                0 0 0 0 rgba(220, 38, 38, 0.10),
                0 0 0 4px rgba(220, 38, 38, 0.08);
        }
        50% {
            box-shadow:
                0 0 0 6px rgba(220, 38, 38, 0.12),
                0 0 0 10px rgba(220, 38, 38, 0.06);
        }
    }

    body.accident-alert-active:not(.accident-alert-muted) #emergency-alert-region .state-active,
    body.accident-alert-active:not(.accident-alert-muted) #detection-output-panel {
        animation: alertPulse 2.4s ease-in-out infinite;
    }

    body.accident-alert-active.accident-alert-source-image #image-preview-input,
    body.accident-alert-active.accident-alert-source-image #image-preview-output,
    body.accident-alert-active.accident-alert-source-image #image-status-region .status-banner,
    body.accident-alert-active.accident-alert-source-video #video-preview-input,
    body.accident-alert-active.accident-alert-source-video #video-preview-output,
    body.accident-alert-active.accident-alert-source-video #video-status-region .status-banner {
        border-color: var(--alert-ring-strong) !important;
        box-shadow:
            0 0 0 1px rgba(220, 38, 38, 0.18),
            0 0 0 6px var(--alert-ring-soft),
            0 18px 38px rgba(220, 38, 38, 0.12) !important;
    }

    body.accident-alert-active:not(.accident-alert-muted).accident-alert-source-image #image-preview-input,
    body.accident-alert-active:not(.accident-alert-muted).accident-alert-source-image #image-preview-output,
    body.accident-alert-active:not(.accident-alert-muted).accident-alert-source-image #image-status-region .status-banner,
    body.accident-alert-active:not(.accident-alert-muted).accident-alert-source-video #video-preview-input,
    body.accident-alert-active:not(.accident-alert-muted).accident-alert-source-video #video-preview-output,
    body.accident-alert-active:not(.accident-alert-muted).accident-alert-source-video #video-status-region .status-banner {
        animation: alertBorderPulse 2.4s ease-in-out infinite;
    }

    body.accident-alert-active #detection-output-panel {
        border-color: rgba(248, 113, 113, 0.36);
        box-shadow:
            0 0 0 1px rgba(220, 38, 38, 0.12),
            0 24px 60px rgba(220, 38, 38, 0.16);
    }

    body.accident-alert-muted #detection-output-panel,
    body.accident-alert-muted #image-preview-input,
    body.accident-alert-muted #image-preview-output,
    body.accident-alert-muted #video-preview-input,
    body.accident-alert-muted #video-preview-output,
    body.accident-alert-muted #image-status-region .status-banner,
    body.accident-alert-muted #video-status-region .status-banner,
    body.accident-alert-muted #emergency-alert-region .state-active {
        animation: none !important;
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
            with gr.Column(elem_classes="hero-card"):
                gr.HTML(
                    """
                    <div class="hero-grid">
                        <div class="hero-copy">
                            <span class="hero-label">CAPSTONE PROJECT INTERFACE</span>
                            <h1 class="hero-title">Total Accident Detection</h1>
                            <p class="hero-subtitle">
                                Capstone Project &ndash; AI-Based Road Accident Detection System
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
                                    Choose between Image or Video testing mode, upload your file, tune parameters, and trigger YOLO detection.
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
                        video_threshold_readout = gr.HTML(build_threshold_readout(0.25))
                        video_conf_slider = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.25,
                            step=0.01,
                            show_label=False,
                            elem_classes="threshold-slider",
                        )
                        video_submit_btn = gr.Button(
                            "Run Video Inference",
                            variant="primary",
                            elem_classes="primary-action",
                            elem_id="run-video-inference-btn",
                        )

                with gr.Column(scale=6, elem_classes="panel-card output-card", elem_id="detection-output-panel"):
                    gr.HTML(
                        """
                        <div class="panel-intro">
                            <span class="section-icon">OUT</span>
                            <div class="panel-copy">
                                <p class="panel-kicker">YOLO Detection Output</p>
                                <h2 class="panel-title">YOLO Detection Output</h2>
                                <p class="panel-description">
                                    Inspect the detection status banner and rendered output with bounding boxes from the active model.
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
            ],
        )

        # Upload and Clear Event handlers to dynamically update status banners
        image_input.upload(
            fn=handle_image_upload,
            inputs=image_input,
            outputs=[status_output, image_output, alert_banner_output, alert_signal_output],
        )
        image_input.clear(
            fn=handle_image_clear,
            outputs=[status_output, image_output, alert_banner_output, alert_signal_output],
        )

        video_input.upload(
            fn=handle_video_upload,
            inputs=video_input,
            outputs=[video_status_output, video_output, video_placeholder, alert_banner_output, alert_signal_output],
        )
        video_input.clear(
            fn=handle_video_clear,
            outputs=[video_status_output, video_output, video_placeholder, alert_banner_output, alert_signal_output],
        )

        # Inference Trigger Event Handlers
        submit_btn.click(
            fn=run_image_inference,
            inputs=[image_input, conf_slider],
            outputs=[image_output, status_output, alert_banner_output, alert_signal_output],
        )

        video_submit_btn.click(
            fn=run_video_inference,
            inputs=[video_input, video_conf_slider],
            outputs=[video_output, video_status_output, video_placeholder, alert_banner_output, alert_signal_output],
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
