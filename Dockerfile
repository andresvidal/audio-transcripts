# syntax=docker/dockerfile:1
# ── Base: Python 3.12 (whisperx requires <3.14) ──────────────────────────────
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        # Required to build some Python wheels (e.g. tokenizers)
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy only packaging metadata first so pip layers are cached independently
# of source code changes.
COPY pyproject.toml .

# Create minimal stub so the editable install resolves the package metadata
RUN mkdir -p src/audio_transcripts && \
    touch src/audio_transcripts/__init__.py

# Install local extras (faster-whisper, transformers, torch) then whisperx
RUN pip install --no-cache-dir -e ".[local]" && \
    pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# ── Copy application source ───────────────────────────────────────────────────
COPY src/ src/
COPY transcribe.py .

# ── Runtime mount points ──────────────────────────────────────────────────────
# /audio      — bind-mount your audio folder here
# /transcripts — output lands here
# /root/.cache/huggingface — model cache (use a named volume to persist)
RUN mkdir -p /audio /transcripts

# HuggingFace home so models are written to the persistent cache volume
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["python", "transcribe.py"]
# Default: show help. Override at runtime, e.g.:
#   docker run ... transcribe /audio --backend faster-whisper -f txt
CMD ["--help"]
