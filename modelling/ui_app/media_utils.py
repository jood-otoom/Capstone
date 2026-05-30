<<<<<<< HEAD
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
=======
# import subprocess
# import os
# from pathlib import Path

# def get_media_label(source: str) -> str:
#     return "Video Analysis" if source == "video" else "Image Analysis"

# def convert_avi_to_mp4(input_path: str) -> str | None:
#     """
#     Converts an AVI video file to a web-compatible H.264 MP4 format using FFmpeg.
#     """
#     input_file = Path(input_path)
#     if not input_file.exists():
#         return None

#     output_file = input_file.with_suffix(".mp4")

#     try:
#         # Run FFmpeg to convert to H.264 for web browser compatibility
#         subprocess.run(
#             [
#                 "ffmpeg", "-y", 
#                 "-i", str(input_file), 
#                 "-vcodec", "libx264", 
#                 "-acodec", "aac", 
#                 str(output_file)
#             ],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             check=True
#         )
        
#         if output_file.exists() and output_file.stat().st_size > 0:
#             return str(output_file)
            
#     except subprocess.CalledProcessError as e:
#         print(f"FFmpeg conversion failed: {e}")
        
#     return None

import subprocess
import os
from pathlib import Path
import cv2
>>>>>>> dcd2c80127c3cebb58a29cfe7eb6913f565d56d6

def get_media_label(source: str) -> str:
    return "Video Analysis" if source == "video" else "Image Analysis"

def resolve_uploaded_media_path(uploaded_value: Any) -> str | None:
    """Normalizes Gradio filepath / FileData-like payloads into a real path string."""
    if uploaded_value is None:
        return None

    if isinstance(uploaded_value, (str, os.PathLike)):
        return str(Path(uploaded_value))

    if isinstance(uploaded_value, dict):
        for key in ("path", "name"):
            value = uploaded_value.get(key)
            if value:
                return str(Path(value))

    for attr in ("path", "name"):
        value = getattr(uploaded_value, attr, None)
        if value:
            return str(Path(value))

    return None

def detect_media_type(uploaded_value: Any) -> str | None:
    media_path = resolve_uploaded_media_path(uploaded_value)
    if not media_path:
        return None

    ext = Path(media_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None

def _opencv_reencode_to_mp4(
    input_file: Path,
    output_file: Path,
    logger: Callable[[str], None],
) -> str | None:
    """
<<<<<<< HEAD
    Fallback MP4 conversion path when ffmpeg/H.264 is unavailable.
    This is less browser-safe than the ffmpeg path, but still produces
    a concrete MP4 file instead of returning a raw AVI.
    """
    import cv2

    if not input_file.exists():
        return None

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        logger(f"[media_utils] OpenCV fallback could not open source video: {input_file}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_frame = None

    if width <= 0 or height <= 0:
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            logger("[media_utils] OpenCV fallback could not infer output dimensions from the first frame.")
            return None
        height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        logger(f"[media_utils] OpenCV fallback could not open writer for: {output_file}")
        return None

    if first_frame is not None:
        writer.write(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()

    if output_file.exists() and output_file.stat().st_size > 0:
        logger(
            f"[media_utils] OpenCV fallback created MP4 output at {output_file} "
            f"({output_file.stat().st_size} bytes)."
        )
        return str(output_file)

    return None

def convert_video_to_browser_mp4(
    input_path: str,
    output_path: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> str | None:
    """
    Converts an arbitrary video file into a browser-friendly H.264 MP4 when
    ffmpeg is available. Falls back to OpenCV MP4 writing as a last resort.
    """
    logger = logger or print
    input_file = Path(input_path)
    if not input_file.exists():
        logger(f"[media_utils] Source video does not exist: {input_file}")
        return None

    if output_path is None:
        output_file = input_file.with_name(f"{input_file.stem}_browser.mp4")
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_file),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(output_file),
        ]
        logger(f"[media_utils] Converting video to browser MP4 with ffmpeg: {output_file}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
            logger(
                f"[media_utils] ffmpeg conversion successful: {output_file} "
                f"({output_file.stat().st_size} bytes)."
            )
            return str(output_file)

        stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
        logger(
            "[media_utils] ffmpeg conversion failed; falling back to OpenCV MP4 re-encode. "
            + " | ".join(stderr_tail)
        )
    else:
        logger("[media_utils] ffmpeg not found on PATH; falling back to OpenCV MP4 re-encode.")

    return _opencv_reencode_to_mp4(input_file, output_file, logger)

def convert_avi_to_mp4(input_path: str) -> str | None:
    """
    Backward-compatible wrapper for older call sites.
    """
    return convert_video_to_browser_mp4(input_path)
=======
    Converts AVI to MP4. Tries FFmpeg first (for production deployment), 
    then gracefully falls back to OpenCV, and finally returns the raw AVI 
    if all else fails so the UI never crashes.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return None

    output_file = input_file.with_suffix(".mp4")

    # ATTEMPT 1: FFmpeg (Will work perfectly on Hugging Face / Docker)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", 
                "-i", str(input_file), 
                "-vcodec", "libx264", 
                "-acodec", "aac", 
                str(output_file)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        if output_file.exists() and output_file.stat().st_size > 0:
            return str(output_file)
    except Exception as e:
        print(f"[Warning] FFmpeg bypassed (Error: {e}). Trying OpenCV fallback...")

    # ATTEMPT 2: OpenCV Fallback (For local Windows testing)
    try:
        cap = cv2.VideoCapture(str(input_file))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try H.264 codec first, fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
                
            if writer.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                    
                cap.release()
                writer.release()
                
                if output_file.exists() and output_file.stat().st_size > 0:
                    return str(output_file)
    except Exception as cv_e:
        print(f"[Warning] OpenCV conversion failed: {cv_e}")

    # ULTIMATE FALLBACK: Return the original file so Gradio does not crash
    print("[Warning] All MP4 conversions failed. Returning raw AVI.")
    return str(input_file)
>>>>>>> dcd2c80127c3cebb58a29cfe7eb6913f565d56d6
