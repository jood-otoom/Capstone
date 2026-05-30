import time
import cv2
import os
import traceback
from pathlib import Path
from uuid import uuid4
import gradio as gr
from ultralytics import YOLO
import numpy as np
from ui_app.config import DETECTION_MODEL_DIR, CLASSIFICATION_MODEL_DIR, PROJECT_ROOT
from ui_app.model_loader import find_best_pt
from ui_app.detection_pipeline import detect_accident_from_collection
from ui_app.styles import render_final_summary_html, format_severity_label, build_status_banner, build_pipeline_status_banner
from ui_app.media_utils import (
    VIDEO_EXTENSIONS,
    convert_video_to_browser_mp4,
    detect_media_type,
    resolve_uploaded_media_path,
)
from ui_app.alerts import build_alert_banner, build_alert_signal
from ui_app.chat_service import render_chat_html
from ui_app.agent_service import get_accident_agent, safe_agent_call
from incident_logger import log_incident_if_confirmed


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
        self.last_accident_detected = False
        self.last_best_frame = None
        self.last_best_crop = None

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
        results = self.detector(frame, conf=conf_threshold, max_det=1, verbose=False)
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
            
        self.last_accident_detected = accident_detected
        self.last_det_conf = max_conf
        self.last_sev_label = severity_label
        self.last_sev_conf = severity_conf
        if accident_detected:
            self.last_best_frame = image_path_or_array
            self.last_best_crop = best_crop if best_crop is not None else image_path_or_array
        else:
            self.last_best_frame = None
            self.last_best_crop = None
            
        return {
            "accident_detected": accident_detected,
            "detection_confidence": max_conf,
            "severity_label": severity_label,
            "severity_confidence": severity_conf,
            "annotated_image": annotated_image,
            "num_detections": len(results[0].boxes) if (results and getattr(results[0], "boxes", None) is not None) else 0
        }

pipeline = AccidentSeverityPipeline()
try:
    pipeline.load_models()
    model = pipeline.detector
    BEST_MODEL_PATH = pipeline.detector_path
except Exception as init_err:
    print(f"CRITICAL: Failed to load pipeline models: {init_err}")
    model = None
    BEST_MODEL_PATH = None

def _video_log(message: str):
    print(f"[VIDEO] {message}")

def _extract_class_name(result, class_id: int) -> str:
    names = getattr(result, "names", {}) or {}
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, ""))
        return str(names[class_id])
    except Exception:
        return ""

def _select_top_confidence_box(result, conf_threshold: float):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best_box = None
    for box in boxes:
        conf = float(box.conf[0].item())
        if conf < conf_threshold:
            continue

        class_id = -1
        try:
            class_id = int(box.cls[0].item())
        except Exception:
            class_id = -1

        xyxy = box.xyxy[0].cpu().numpy()
        candidate = {
            "confidence": conf,
            "class_id": class_id,
            "class_name": _extract_class_name(result, class_id),
            "xyxy": xyxy,
        }
        if best_box is None or candidate["confidence"] > best_box["confidence"]:
            best_box = candidate

    return best_box

def _crop_box_from_frame(frame_bgr: np.ndarray, xyxy) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

def _draw_selected_box(frame_bgr: np.ndarray, detection: dict):
    x1, y1, x2, y2 = map(int, detection["xyxy"])
    class_label = detection["class_name"] or "accident"
    label = f"{class_label} {detection['confidence']:.2f}"
    color = (0, 0, 255)
    thickness = max(2, int(round(min(frame_bgr.shape[:2]) / 240)))

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )
    top = max(0, y1 - text_height - baseline - 8)
    cv2.rectangle(
        frame_bgr,
        (x1, top),
        (x1 + text_width + 10, top + text_height + baseline + 8),
        color,
        -1,
    )
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 5, top + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

def _open_intermediate_video_writer(output_dir: Path, base_name: str, fps: float, frame_size: tuple[int, int]):
    codec_candidates = [
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
        ("mp4v", ".mp4"),
        ("WMV2", ".avi"),
    ]

    for codec, extension in codec_candidates:
        candidate_path = output_dir / f"{base_name}_annotated_intermediate{extension}"
        writer = cv2.VideoWriter(
            str(candidate_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            _video_log(
                f"Opened intermediate writer with codec={codec}, fps={fps:.3f}, "
                f"size={frame_size}, path={candidate_path}"
            )
            return writer, candidate_path, codec
        writer.release()

    raise RuntimeError(
        f"Could not open an intermediate video writer for {frame_size} at {fps:.3f} FPS."
    )

def _process_video_pipeline(uploaded_video, conf_threshold: float) -> dict:
    resolved_path = resolve_uploaded_media_path(uploaded_video)
    media_type = detect_media_type(resolved_path)

    _video_log(f"Uploaded value type: {type(uploaded_video).__name__}")
    _video_log(f"Uploaded file path: {resolved_path}")
    _video_log(f"Media type detected: {media_type}")

    if not resolved_path:
        raise ValueError("Uploaded video path could not be resolved from the Gradio payload.")

    input_video = Path(resolved_path)
    _video_log(f"Input video exists: {input_video.exists()}")
    _video_log(f"Input video extension: {input_video.suffix.lower()}")

    if media_type != "video" or input_video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format: {input_video.suffix}. "
            f"Expected one of {sorted(VIDEO_EXTENSIONS)}."
        )
    if not input_video.exists():
        raise FileNotFoundError(f"Uploaded video does not exist: {input_video}")

    output_dir = PROJECT_ROOT / "runs" / "detect" / f"video_predictions_{int(time.time() * 1000)}_{uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _video_log(f"Output directory path: {output_dir}")

    cap = cv2.VideoCapture(str(input_video))
    cap_opened = cap.isOpened()
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    _video_log(f"cap.isOpened(): {cap_opened}")
    _video_log(
        f"Capture metadata -> fps={fps_raw}, width={width_raw}, "
        f"height={height_raw}, frame_count={frame_count}"
    )

    if not cap_opened:
        cap.release()
        raise RuntimeError(f"OpenCV could not open the uploaded video: {input_video}")

    fps = fps_raw if fps_raw > 0 else 24.0
    if fps != fps_raw:
        _video_log(f"FPS metadata was invalid ({fps_raw}); falling back to {fps:.1f} FPS.")

    width = width_raw
    height = height_raw
    buffered_first_frame = None
    if width <= 0 or height <= 0:
        ok, buffered_first_frame = cap.read()
        if not ok or buffered_first_frame is None:
            cap.release()
            raise RuntimeError("Could not infer video dimensions from the first frame.")
        height, width = buffered_first_frame.shape[:2]
        _video_log(f"Width/height metadata was invalid; inferred dimensions from first frame: {width}x{height}")

    writer, intermediate_output_path, writer_codec = _open_intermediate_video_writer(
        output_dir,
        input_video.stem,
        fps,
        (width, height),
    )
    _video_log(f"Intermediate output path: {intermediate_output_path}")

    frames_processed = 0
    frames_with_accident_detections = 0
    highest_det_conf = 0.0
    best_crop_rgb = None
    best_frame_rgb = None

    def process_single_frame(frame_bgr: np.ndarray):
        nonlocal frames_processed
        nonlocal frames_with_accident_detections
        nonlocal highest_det_conf
        nonlocal best_crop_rgb
        nonlocal best_frame_rgb

        results = pipeline.detector(frame_bgr, conf=conf_threshold, verbose=False)
        result = results[0] if results else None
        annotated_frame = frame_bgr.copy()
        top_detection = _select_top_confidence_box(result, conf_threshold)

        if top_detection is not None:
            frames_with_accident_detections += 1
            _draw_selected_box(annotated_frame, top_detection)
            if top_detection["confidence"] > highest_det_conf:
                highest_det_conf = top_detection["confidence"]
                best_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                crop_bgr = _crop_box_from_frame(frame_bgr, top_detection["xyxy"])
                if crop_bgr is not None:
                    best_crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        writer.write(annotated_frame)
        frames_processed += 1

    try:
        if buffered_first_frame is not None:
            process_single_frame(buffered_first_frame)

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            process_single_frame(frame_bgr)
    finally:
        cap.release()
        writer.release()
        _video_log("Released capture and writer resources.")

    if frames_processed == 0:
        raise RuntimeError("No frames were processed from the uploaded video.")

    accident_detected = frames_with_accident_detections > 0
    severity_label = "Not Applied"
    severity_conf = 0.0

    _video_log(f"Number of frames processed: {frames_processed}")
    _video_log(f"Number of frames with accident detections: {frames_with_accident_detections}")

    if accident_detected:
        label = None
        conf_val = 0.0
        if best_crop_rgb is not None:
            _video_log(
                "Accident detected. Running severity classification on the highest-confidence crop "
                f"(conf={highest_det_conf:.4f})."
            )
            label, conf_val = pipeline.classify_severity(best_crop_rgb)

        if label is None and best_frame_rgb is not None:
            _video_log("Crop classification failed. Falling back to the highest-confidence full frame.")
            label, conf_val = pipeline.classify_severity(best_frame_rgb)

        severity_label = label if label else "Classification Failed"
        severity_conf = conf_val
        _video_log(f"Severity classification result: {severity_label} ({severity_conf:.4f})")
    else:
        _video_log("No accident detected in the processed video.")

    final_output_path = convert_video_to_browser_mp4(
        str(intermediate_output_path),
        str(output_dir / f"{input_video.stem}_annotated.mp4"),
        logger=_video_log,
    )
    _video_log(f"Final converted output path: {final_output_path}")

    if not final_output_path:
        raise RuntimeError(
            "Annotated video was written to an intermediate file, but conversion to a browser-playable MP4 failed."
        )

    final_output_file = Path(final_output_path)
    output_exists = final_output_file.exists()
    output_size = final_output_file.stat().st_size if output_exists else 0
    _video_log(f"Final output exists: {output_exists}")
    _video_log(f"Final output size (bytes): {output_size}")

    if not output_exists or output_size <= 0:
        raise RuntimeError(f"Final processed video is missing or empty: {final_output_path}")

    pipeline.last_det_conf = highest_det_conf
    pipeline.last_sev_label = severity_label
    pipeline.last_sev_conf = severity_conf
    pipeline.last_accident_detected = accident_detected
    pipeline.last_best_frame = best_frame_rgb
    pipeline.last_best_crop = best_crop_rgb

    return {
        "input_video_path": str(input_video),
        "media_type": media_type,
        "intermediate_output_path": str(intermediate_output_path),
        "writer_codec": writer_codec,
        "processed_video_path": final_output_path,
        "accident_detected": accident_detected,
        "detection_confidence": highest_det_conf,
        "severity_label": severity_label,
        "severity_confidence": severity_conf,
        "frames_processed": frames_processed,
        "frames_with_accident_detections": frames_with_accident_detections,
    }

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
    resolved_video_path = resolve_uploaded_media_path(video_path)
    if not resolved_video_path:
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

    ext = Path(resolved_video_path).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return (
            build_status_banner(
                title="Unsupported video format",
                message=(
                    f"The uploaded format {ext} is not supported. "
                    "Please use MP4, AVI, MOV, MKV, or WEBM."
                ),
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
        
        # --- NEW WEBHOOK INTEGRATION ---
        if accident_detected:
            try:
                # Gradio uses RGB, but your backend uses cv2 which expects BGR
                bgr_frame = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
                
                # Send the data to your backend!
                log_incident_if_confirmed(
                    severity=sev_label,
                    detection_confidence=det_conf,
                    classification_confidence=sev_conf,
                    media_type="image",
                    frame=bgr_frame
                )
                print("[UI] Successfully sent accident data to backend logger/webhook.")
            except Exception as e:
                print(f"[UI] Error sending data to webhook: {e}")
        # -------------------------------

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
        traceback.print_exc()
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
            traceback.print_exc()
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
        traceback.print_exc()
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
        traceback.print_exc()
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

    resolved_video_path = resolve_uploaded_media_path(video_path)
    media_type = detect_media_type(resolved_video_path)

    print("[VIDEO] run_video_inference started")
    print(f"[VIDEO] received video_path: {video_path}")
    print(f"[VIDEO] resolved video_path: {resolved_video_path}")
    print(f"[VIDEO] detected media_type: {media_type}")
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

    if not resolved_video_path:
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

    video_file = Path(resolved_video_path)
    print(f"[VIDEO] video_path exists: {video_file.exists()}")
    print(f"[VIDEO] video_path suffix: {video_file.suffix}")

    if media_type != "video" or video_file.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"[VIDEO] unsupported format: {video_file.suffix}")
        chat_history = chat_history or []
        yield (
            None,
            build_status_banner(
                title="Unsupported video format",
                message=f"The uploaded format {video_file.suffix} is not supported. Please use MP4, AVI, MOV, MKV, or WEBM.",
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
        video_result = _process_video_pipeline(resolved_video_path, confidence_threshold)
        preview_path = video_result["processed_video_path"]
        accident_detected = video_result["accident_detected"]
        highest_det_conf = video_result["detection_confidence"]
        severity_label = video_result["severity_label"]
        severity_conf = video_result["severity_confidence"]
        number_of_detected_frames = video_result["frames_with_accident_detections"]
        best_frame = pipeline.last_best_frame
        best_crop = pipeline.last_best_crop

<<<<<<< HEAD
        _video_log(f"Exact return value sent to Gradio video component: {preview_path}")
=======
        print(f"[VIDEO] output directory path: {out_dir}")
        print("[VIDEO] calling pipeline.detector.predict() now...")

        # Run inference using the natively supported YOLO predict function with stride
        results = pipeline.detector.predict(
            source=video_path,
            conf=confidence_threshold,
            max_det=1,
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
>>>>>>> dcd2c80127c3cebb58a29cfe7eb6913f565d56d6
        status_html = build_pipeline_status_banner(
            accident_detected=accident_detected,
            detection_conf=highest_det_conf,
            severity_label=severity_label,
            severity_conf=severity_conf,
            num_detections=number_of_detected_frames,
            source="video",
            processed_video_path=preview_path,
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


def predict_accident_video_gui(video_path: str, conf_threshold: float):
    if pipeline.detector is None:
        raise RuntimeError("YOLO detector not loaded.")

<<<<<<< HEAD
    video_result = _process_video_pipeline(video_path, conf_threshold)
    preview_path = video_result["processed_video_path"]
    accident_detected = video_result["accident_detected"]
    highest_det_conf = video_result["detection_confidence"]
    severity_label = video_result["severity_label"]
    severity_conf = video_result["severity_confidence"]
    number_of_detected_frames = video_result["frames_with_accident_detections"]
=======
    run_id = str(int(time.time()))
    project_dir = PROJECT_ROOT / "runs" / "detect"
    name_dir = f"video_predictions_{run_id}"
    out_dir = project_dir / name_dir

    print(f"[VIDEO] output directory path: {out_dir}")
    print("[VIDEO] calling pipeline.detector.predict() now...")

    results = pipeline.detector.predict(
        source=video_path,
        conf=conf_threshold,
        max_det=1,
        save=True,
        project=str(project_dir),
        name=name_dir,
        exist_ok=True,
        imgsz=640,
        vid_stride=3,
    )

    print("[VIDEO] pipeline.detector.predict() finished")
    
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
            
        if label is None:
            if best_frame is not None:
                print("[VIDEO] Classification on crop failed. Falling back to full frame of best detection...")
                label, conf_val = pipeline.classify_severity(best_frame)
                
        severity_label = label if label else "Classification Failed"
        severity_conf = conf_val
        print(f"[VIDEO] Severity Classification result: {severity_label} ({severity_conf:.4f})")
    else:
        print("[VIDEO] No accident detected in video.")

    pipeline.last_det_conf = highest_det_conf
    pipeline.last_sev_label = severity_label
    pipeline.last_sev_conf = severity_conf
    pipeline.last_accident_detected = accident_detected
    pipeline.last_best_frame = best_frame
    pipeline.last_best_crop = best_crop

    # Locate output video
    output_video_path = None
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if out_dir.exists():
        for file in out_dir.rglob("*"):
            if file.is_file():
                if file.suffix.lower() in allowed_exts and not output_video_path:
                    output_video_path = str(file)

    preview_path = None
    if output_video_path and Path(output_video_path).exists():
        output_file = Path(output_video_path)
        if output_file.suffix.lower() == ".mp4":
            preview_path = str(output_file)
        elif output_file.suffix.lower() == ".avi":
            print(f"[VIDEO] Found AVI output. Converting to MP4: {output_video_path}")
            converted_path = convert_avi_to_mp4(str(output_file))
            if converted_path:
                preview_path = converted_path
            else:
                preview_path = str(output_file)
        else:
            preview_path = str(output_file)
>>>>>>> dcd2c80127c3cebb58a29cfe7eb6913f565d56d6

    status_html = build_pipeline_status_banner(
        accident_detected=accident_detected,
        detection_conf=highest_det_conf,
        severity_label=severity_label,
        severity_conf=severity_conf,
        num_detections=number_of_detected_frames,
        source="video",
        processed_video_path=preview_path,
    )

    alert_banner = build_alert_banner("active" if accident_detected else "clear", "video")
    alert_sig = build_alert_signal(accident_detected, "accident" if accident_detected else "clear", "video")

    return preview_path, status_html, alert_banner, alert_sig

def prepare_uploaded_evidence_preview(file_path: str | None):
    resolved_path = resolve_uploaded_media_path(file_path)
    media_type = detect_media_type(resolved_path)

    if not resolved_path:
        yield (
            None, None,
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            None, None,
            build_status_banner("No file uploaded", "Please upload a valid evidence file.", "neutral", "&#8682;"),
            build_status_banner("No file uploaded", "Please upload a valid evidence file.", "neutral", "&#8682;"),
            build_alert_banner("standby", "image"),
            build_alert_signal(False, "idle", "image"),
            render_chat_html([]),
            gr.update(interactive=False, placeholder="Upload file to start..."),
            gr.update(interactive=False, value="Locked"),
            [],
            gr.update(visible=True),
            gr.update(visible=False),
        )

    ext = Path(resolved_path).suffix.lower()
    if media_type not in {"image", "video"}:
        status_banner = build_status_banner(
            title="Unsupported file format",
            message=(
                f"The uploaded format {ext} is not supported. Please upload an image "
                "(.jpg, .jpeg, .png, .bmp, .webp) or a video (.mp4, .avi, .mov, .mkv, .webm)."
            ),
            tone="alert",
            icon="&#10005;",
        )
        return (
            None, None,
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            None, None,
            status_banner, status_banner,
            build_alert_banner("error", "image"),
            build_alert_signal(False, "error", "image"),
            render_chat_html([]),
            gr.update(interactive=False, placeholder="Unsupported file format."),
            gr.update(interactive=False, value="Locked"),
            [],
            gr.update(visible=True),
            gr.update(visible=False),
        )

    locked_prompt = gr.update(interactive=False, placeholder="Processing uploaded evidence...")
    locked_button = gr.update(interactive=False, value="Analyzing...")
    chat_html = render_chat_html([])

    if media_type == "image":
        try:
            bgr_img = cv2.imread(resolved_path)
            if bgr_img is None:
                raise ValueError("Could not read image using OpenCV.")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            err_msg = f"Failed to load image preview: {e}"
            yield (
                None, None,
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False),
                None, None,
                build_status_banner("Image load error", err_msg, "alert", "&#10005;"),
                build_status_banner("Image load error", err_msg, "alert", "&#10005;"),
                build_alert_banner("error", "image"),
                build_alert_signal(False, "error", "image"),
                chat_html,
                gr.update(interactive=False, placeholder="Load error."),
                gr.update(interactive=False, value="Locked"),
                [],
                gr.update(visible=True),
                gr.update(visible=False),
            )

        return (
            rgb_img, None,
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            None, None,
            build_status_banner(
                "Image uploaded successfully",
                "Preview loaded. Starting accident detection automatically...",
                "neutral",
                "&#9711;",
            ),
            None,
            build_alert_banner("processing", "image"),
            build_alert_signal(False, "processing", "image"),
            chat_html,
            locked_prompt,
            locked_button,
            [],
            gr.update(visible=False),
            gr.update(visible=True),
        )

    return (
        None, gr.update(value=resolved_path),
        gr.update(visible=False), gr.update(visible=True),
        gr.update(visible=False), gr.update(visible=True),
        None, None,
        None,
        build_status_banner(
            "Video uploaded successfully",
            "Original preview loaded. Starting video processing automatically...",
            "neutral",
            "&#9711;",
        ),
        build_alert_banner("processing", "video"),
        build_alert_signal(False, "processing", "video"),
        chat_html,
        locked_prompt,
        locked_button,
        [],
        gr.update(visible=False),
        gr.update(visible=True),
    )


def run_model_inference_flow(file_path: str | None):
    resolved_path = resolve_uploaded_media_path(file_path)
    media_type = detect_media_type(resolved_path)

    print(f"[Pipeline] Uploaded evidence value type: {type(file_path).__name__}")
    print(f"[Pipeline] Uploaded evidence resolved path: {resolved_path}")
    print(f"[Pipeline] Uploaded evidence media type: {media_type}")

    if not resolved_path:
        yield (
            None, None,
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            None, None,
            build_status_banner("No file uploaded", "Please upload a valid evidence file.", "neutral", "&#8682;"),
            build_status_banner("No file uploaded", "Please upload a valid evidence file.", "neutral", "&#8682;"),
            build_alert_banner("standby", "image"),
            build_alert_signal(False, "idle", "image"),
            render_chat_html([]),
            gr.update(interactive=False, placeholder="Upload file to start..."),
            gr.update(interactive=False, value="Locked"),
            [],
            gr.update(visible=True),  # Show evidence_upload
            gr.update(visible=False)  # Hide clear_btn
        )
        return

    ext = Path(resolved_path).suffix.lower()

    if media_type not in {"image", "video"}:
        status_banner = build_status_banner(
            title="Unsupported file format",
            message=(
                f"The uploaded format {ext} is not supported. Please upload an image "
                "(.jpg, .jpeg, .png, .bmp, .webp) or a video (.mp4, .avi, .mov, .mkv, .webm)."
            ),
            tone="alert",
            icon="&#10005;"
        )
        yield (
            None, None,
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            None, None,
            status_banner, status_banner,
            build_alert_banner("error", "image"),
            build_alert_signal(False, "error", "image"),
            render_chat_html([]),
            gr.update(interactive=False, placeholder="Unsupported file format."),
            gr.update(interactive=False, value="Locked"),
            [],
            gr.update(visible=True),  # Show evidence_upload
            gr.update(visible=False)  # Hide clear_btn
        )
        return

    if media_type == "image":
        try:
            bgr_img = cv2.imread(resolved_path)
            if bgr_img is None:
                raise ValueError("Could not read image using OpenCV.")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            err_msg = f"Failed to load image: {e}"
            print(f"[Pipeline] {err_msg}")
            yield (
                None, None,
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False),
                None, None,
                build_status_banner("Image load error", err_msg, "alert", "&#10005;"),
                build_status_banner("Image load error", err_msg, "alert", "&#10005;"),
                build_alert_banner("error", "image"),
                build_alert_signal(False, "error", "image"),
                render_chat_html([]),
                gr.update(interactive=False, placeholder="Load error."),
                gr.update(interactive=False, value="Locked"),
                [],
                gr.update(visible=True),  # Show evidence_upload
                gr.update(visible=False)  # Hide clear_btn
            )
            return

        yield (
            rgb_img, None,
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            None, None,
            build_status_banner(
                "Image uploaded successfully",
                "Preview loaded. Starting accident detection automatically...",
                "neutral",
                "&#9711;",
            ),
            None,
            build_alert_banner("processing", "image"),
            build_alert_signal(False, "processing", "image"),
            render_chat_html([]),
            gr.update(interactive=False, placeholder="Processing uploaded evidence..."),
            gr.update(interactive=False, value="Analyzing..."),
            [],
            gr.update(visible=False),
            gr.update(visible=True),
        )

        try:
            annotated_image, status_html, alert_html, alert_signal = predict_accident_gui(rgb_img, conf_threshold=0.60)
        except Exception as e:
            err_msg = f"Image inference failed: {e}"
            print(f"[Pipeline] {err_msg}")
            yield (
                rgb_img, None,
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False),
                None, None,
                build_status_banner("Inference failed", err_msg, "alert", "&#10005;"),
                None,
                build_alert_banner("error", "image"),
                build_alert_signal(False, "error", "image"),
                render_chat_html([]),
                gr.update(interactive=False, placeholder="Inference failed."),
                gr.update(interactive=False, value="Locked"),
                [],
                gr.update(visible=True),  # Show evidence_upload
                gr.update(visible=False)  # Hide clear_btn
            )
            return

        accident_detected = pipeline.last_accident_detected
        
        if accident_detected:
            chatbot_html = render_chat_html([("System", "⏳ <b>Model Processing Complete: Accident Detected!</b> Initiating Jordan Traffic Law Liability Reasoning in the background... Please wait...")])
            prompt_update = gr.update(interactive=False, placeholder="AI Agent is analyzing the scene...")
            button_update = gr.update(interactive=False, value="Analyzing...")
        else:
            chatbot_html = render_chat_html([("System", "✅ <b>Model Processing Complete: No Accident Detected.</b> Legal assistant is available for general road safety or traffic law questions.")])
            prompt_update = gr.update(interactive=True, placeholder="Ask the legal assistant general questions...")
            button_update = gr.update(interactive=True, value="Send")

        yield (
            rgb_img, None,
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            annotated_image, None,
            status_html, None,
            alert_html,
            alert_signal,
            chatbot_html,
            prompt_update,
            button_update,
            [],
            gr.update(visible=False), # Hide evidence_upload!
            gr.update(visible=True)   # Show clear_btn!
        )

    else:
        yield (
            None, resolved_path,
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=True),
            None, None,
            None,
            build_status_banner(
                "Video uploaded successfully",
                "Original preview loaded. Starting video processing automatically...",
                "neutral",
                "&#9711;",
            ),
            build_alert_banner("processing", "video"),
            build_alert_signal(False, "processing", "video"),
            render_chat_html([]),
            gr.update(interactive=False, placeholder="Processing uploaded evidence..."),
            gr.update(interactive=False, value="Analyzing..."),
            [],
            gr.update(visible=False),
            gr.update(visible=True),
        )

        try:
            preview_path, status_html, alert_html, alert_signal = predict_accident_video_gui(resolved_path, conf_threshold=0.70)
        except Exception as e:
            err_msg = f"Video inference failed: {e}"
            print(f"[Pipeline] {err_msg}")
            yield (
                None, resolved_path,
                gr.update(visible=False), gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=True),
                None, None,
                None,
                build_status_banner("Video processing failed", err_msg, "alert", "&#10005;"),
                build_alert_banner("error", "video"),
                build_alert_signal(False, "error", "video"),
                render_chat_html([]),
                gr.update(interactive=False, placeholder="Processing failed."),
                gr.update(interactive=False, value="Locked"),
                [],
                gr.update(visible=True),  # Show evidence_upload
                gr.update(visible=False)  # Hide clear_btn
            )
            return

        accident_detected = pipeline.last_accident_detected

        if accident_detected:
            chatbot_html = render_chat_html([("System", "⏳ <b>Model Processing Complete: Accident Detected!</b> Initiating Jordan Traffic Law Liability Reasoning in the background... Please wait...")])
            prompt_update = gr.update(interactive=False, placeholder="AI Agent is analyzing the video...")
            button_update = gr.update(interactive=False, value="Analyzing...")
        else:
            chatbot_html = render_chat_html([("System", "✅ <b>Model Processing Complete: No Accident Detected.</b> Legal assistant is available for general road safety or traffic law questions.")])
            prompt_update = gr.update(interactive=True, placeholder="Ask the legal assistant general questions...")
            button_update = gr.update(interactive=True, value="Send")

        _video_log(f"Exact return value sent to Gradio video component: {preview_path}")
        yield (
            None, resolved_path,
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=True),
            None, preview_path,
            None, status_html,
            alert_html,
            alert_signal,
            chatbot_html,
            prompt_update,
            button_update,
            [],
            gr.update(visible=False), # Hide evidence_upload!
            gr.update(visible=True)   # Show clear_btn!
        )
        return


def run_agent_analysis_flow(file_path: str | None, chat_history, agent):
    resolved_path = resolve_uploaded_media_path(file_path)
    media_type = detect_media_type(resolved_path)

    if not resolved_path:
        return render_chat_html([]), gr.update(interactive=False, placeholder="Upload file to start..."), gr.update(interactive=False, value="Locked"), agent, []

    accident_detected = pipeline.last_accident_detected
    chat_history = chat_history or []

    if media_type not in {"image", "video"}:
        return (
            render_chat_html(chat_history),
            gr.update(interactive=False, placeholder="Unsupported file format."),
            gr.update(interactive=False, value="Locked"),
            agent,
            chat_history
        )

    if not accident_detected:
        return (
            render_chat_html(chat_history),
            gr.update(interactive=True, placeholder="Ask the legal assistant general questions..."),
            gr.update(interactive=True, value="Send"),
            agent,
            chat_history
        )

    if not agent:
        try:
            print("[Agent] Initializing AccidentAgent...")
            agent = get_accident_agent()
        except Exception as e:
            print(f"[Agent] Initialization failed: {e}")
            chat_history.append(("Assistant", f"❌ Failed to load AI Agent folder or configs: {str(e)}"))
            return (
                render_chat_html(chat_history),
                gr.update(interactive=False, placeholder="Agent offline. Config missing."),
                gr.update(interactive=False, value="Locked"),
                agent,
                chat_history,
            )

    if media_type == "image":
        temp_path = os.path.join(str(PROJECT_ROOT), "accident_agent", "temp_input.jpg")
        try:
            if pipeline.last_best_frame is not None:
                bgr_img = cv2.cvtColor(pipeline.last_best_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_path, bgr_img)
            else:
                import shutil
                shutil.copy2(resolved_path, temp_path)
        except Exception as e:
            print(f"[Agent] Failed to save temporary image frame: {e}")

        try:
            print(f"[Agent] Calling generate_initial_analysis on: {temp_path}")
            analysis_report = safe_agent_call(agent, "generate_initial_analysis", [temp_path])
            chat_history.append(("Assistant", analysis_report))
        except Exception as e:
            print(f"[Agent] Analysis invocation failed: {e}")
            chat_history.append(("Assistant", f"❌ Agent analysis failed: {str(e)}\n\nCheck terminal logs for debugging details."))

    elif media_type == "video":
        temp_paths = []
        if pipeline.last_best_frame is not None:
            tf = os.path.join(str(PROJECT_ROOT), "accident_agent", "temp_video_frame.jpg")
            try:
                cv2.imwrite(tf, cv2.cvtColor(pipeline.last_best_frame, cv2.COLOR_RGB2BGR))
                temp_paths.append(tf)
            except Exception as e:
                print(f"[Video Agent] Error saving frame: {e}")
        if pipeline.last_best_crop is not None:
            tc = os.path.join(str(PROJECT_ROOT), "accident_agent", "temp_video_crop.jpg")
            try:
                cv2.imwrite(tc, cv2.cvtColor(pipeline.last_best_crop, cv2.COLOR_RGB2BGR))
                temp_paths.append(tc)
            except Exception as e:
                print(f"[Video Agent] Error saving crop: {e}")

        if not temp_paths:
            temp_paths = [resolved_path]

        try:
            print(f"[Video Agent] Calling generate_initial_analysis on: {temp_paths}")
            analysis_report = safe_agent_call(agent, "generate_initial_analysis", temp_paths)
            chat_history.append(("Auto Accident Analysis", analysis_report))
        except Exception as e:
            print(f"[Video Agent] KAG analysis failed: {e}")
            chat_history.append(("Auto Accident Analysis", f"❌ Video analysis failed: {str(e)}\n\nCheck terminal logs for traceback."))

    return (
        render_chat_html(chat_history),
        gr.update(interactive=True, placeholder="Ask the legal assistant..."),
        gr.update(interactive=True, value="Send"),
        agent,
        chat_history
    )

