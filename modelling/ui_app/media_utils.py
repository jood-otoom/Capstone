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

def get_media_label(source: str) -> str:
    return "Video Analysis" if source == "video" else "Image Analysis"

def convert_avi_to_mp4(input_path: str) -> str | None:
    """
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