# ══════════════════════════════════════════════════════════════════════════════
# Code Autopsy — Serve Dockerfile
# ══════════════════════════════════════════════════════════════════════════════
# Runs the FastAPI inference server on port 8000.
# Model weights are NOT baked in — mount them as volumes at runtime.
#
# Build (from repo root — required so COPY paths resolve correctly):
#   docker build -f serve/Dockerfile -t code-autopsy-serve .
#
# Run (with adapter mounted from host, needs NVIDIA Docker runtime):
#   docker run -p 8000:8000 \
#     -v $(pwd)/adapter:/app/adapter:ro \
#     --env-file .env \
#     --gpus all \
#     code-autopsy-serve
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy from repo root (build context must be the repo root, not ./serve)
COPY requirements.txt ./requirements.txt
COPY training/utils.py ./training/utils.py
COPY serve/api.py ./serve/api.py

# Create empty __init__.py so training is a proper package
RUN touch training/__init__.py

# Install Python dependencies (CPU torch for Docker; override at runtime with GPU)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        transformers>=4.46.0 \
        peft>=0.13.0 \
        bitsandbytes>=0.44.0 \
        fastapi>=0.115.0 \
        uvicorn[standard]>=0.30.0 \
        pydantic>=2.0.0 \
        python-dotenv>=1.0.0 \
        accelerate>=0.34.0 \
        sentencepiece>=0.2.0 \
        numpy>=1.26.0

# Model weights are mounted at runtime, not baked in
VOLUME ["/app/adapter", "/root/.cache/huggingface"]

# Environment defaults (override via --env-file or -e)
ENV BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
ENV ADAPTER_PATH="/app/adapter"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
