# ══════════════════════════════════════════════════════════════════════════════
# Code Autopsy — Demo Dockerfile
# ══════════════════════════════════════════════════════════════════════════════
# Runs the Gradio UI on port 7860. CPU-only — no model weights needed here.
# This container just calls the FastAPI API at API_URL.
#
# Build (from repo root):
#   docker build -f demo/Dockerfile -t code-autopsy-demo .
#
# Run (standalone):
#   docker run -p 7860:7860 -e API_URL=http://your-api-host:8000 code-autopsy-demo
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the demo needs
COPY requirements.txt ./requirements.txt
COPY demo/app.py ./demo/app.py

# Install lightweight subset (no torch, no GPU libs)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        gradio>=5.0.0 \
        requests>=2.31.0 \
        python-dotenv>=1.0.0

# Environment defaults (override via --env-file or -e)
ENV API_URL="http://localhost:8000"
ENV GRADIO_PORT="7860"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

CMD ["python", "demo/app.py"]
