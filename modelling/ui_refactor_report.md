# UI Refactor Report for test_ui.py

This is the diagnostic analysis of `C:/Capstone/modelling/test_ui.py` before the refactor begins.

## App Framework Detection
- **Framework**: Gradio
- **Launch Command**: `python test_ui.py` or run via Gradio's automatic launch script.

## 1. Imports Used
`pathlib.Path`, `html.escape`, `gradio as gr`, `dotenv`, `zipfile`, `traceback`, `uuid`, `time`, `sys`, `subprocess`, `numpy as np`, `os`, `cv2`, `csv`, `ultralytics.YOLO`, `app.services.agent_service.AccidentAgent`, `app.core.config.settings`

## 2. Global Constants & Model Paths
- `PROJECT_ROOT`
- `AGENT_DIR`
- `DETECTION_MODEL_DIR`
- `CLASSIFICATION_MODEL_DIR`
- `BEST_MODEL_PATH`

## 3. Model-loading Functions / Logic
- `find_best_model`
- `find_best_pt`
- `load_models` (inside `AccidentSeverityPipeline`)

## 4. Image-processing Functions
- `process_image`
- `run_image_inference`
- `handle_image_upload`
- `handle_image_clear`
- `predict_accident_gui`

## 5. Video-processing Functions
- `run_video_inference`
- `handle_video_upload`
- `handle_video_clear`
- `convert_avi_to_mp4`

## 6. Merged Detection + Severity Pipeline
- Class `AccidentSeverityPipeline` (combines `detect_accident` and `classify_severity`)
- `detect_accident_from_result`
- `detect_accident_from_collection`
- `classify_severity`

## 7. Emergency Alert / Buzzing Logic
- `build_alert_controls`
- `build_alert_signal`
- `build_alert_banner`
- `build_alert_controller_head`

## 8. AI Agent Analysis Logic
- `get_accident_agent`
- `safe_agent_call`
- Class `APIKeyManager` with `rotate_key` and `get_current_key`

## 9. Chat Interaction Logic
- `initiate_chat`
- `generate_chat_reply`
- `render_chat_html`

## 10. Gradio UI Layout Sections
- `build_app` (the main layout entrypoint with `gr.Blocks`)
- Helper HTML rendering functions like `render_final_summary_html`, `build_pipeline_status_banner`, `build_threshold_readout`, `build_model_badge`, `build_status_banner`

## 11. Duplicated Functions / Repeated Blocks
- The logic checking for API key authorization failures is slightly spread between `APIKeyManager` and `safe_agent_call`, but overall modularity is achievable by grouping.
- HTML templates are heavily embedded into python functions and will be centralized in `styles.py`.

## 12. Risky Dependencies
- State dependencies with Gradio components inside `build_app`. The event listeners (`.click()`, `.change()`) tie heavily to the locally scoped Gradio components. We must extract these carefully.
- Paths for temporary files (video processing creates temp AVI/MP4). We must ensure `PROJECT_ROOT` resolves properly from inside `ui_app/config.py`.
