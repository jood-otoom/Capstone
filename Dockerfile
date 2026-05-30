# ============================================================
# AcciEye — Traffic Accident Detection & AI Liability Engine
# Multi-stage Dockerfile
# ============================================================

# --------------- Stage 1: Base image ---------------
FROM python:3.11-slim AS base

# System-level dependencies for OpenCV, FAISS, and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime (Updated package name)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # PDF processing (pdfplumber)
    poppler-utils \
    # Networking & utilities
    curl \
    wget \
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# -------------------------------------------------------
# Stage 2: Python dependencies (cached layer)
# -------------------------------------------------------
FROM base AS builder

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# Stage 3: Final runtime image
# -------------------------------------------------------
FROM base AS runtime

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Ultralytics: store config inside container (no home dir write issues)
    YOLO_CONFIG_DIR=/app/.ultralytics \
    # HuggingFace: cache models inside container volume
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Gradio: listen on all interfaces
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Create required runtime directories
RUN mkdir -p \
    /app/.ultralytics \
    /app/.cache/huggingface \
    /app/incident_logs/frames \
    /app/runs/detect \
    /app/data/vectorstore \
    /app/accident_agent/docs

# Copy application source code
COPY . /app

# Expose Gradio UI port
EXPOSE 7860

# Health check — verifies the Gradio server is up
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default: launch the Gradio UI
CMD ["python", "modelling/test_ui.py"]
