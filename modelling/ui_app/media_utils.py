import subprocess
import os

def get_media_label(source: str) -> str:
    return "Video Analysis" if source == "video" else "Image Analysis"

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
