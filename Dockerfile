# Dockerfile — CustomerSupportEnv
# Hugging Face Spaces compatible (port 7860)
# Build: docker build -t customer-support-env .
# Run:   docker run -p 7860:7860 -e HF_TOKEN=<key> customer-support-env

FROM python:3.11-slim

LABEL maintainer="OpenEnv Hackathon"
LABEL org.opencontainers.image.title="CustomerSupportEnv"
LABEL org.opencontainers.image.description="Autonomous Customer Support Ops — OpenEnv v1"
LABEL space_sdk="docker"

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── App user (HF Spaces runs as non-root) ───────────────────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Create empty __init__ files for package imports ───────────────────────────
RUN touch customer_support_env/__init__.py \
         customer_support_env/env/__init__.py \
         customer_support_env/tasks/__init__.py \
         customer_support_env/graders/__init__.py

# ── Environment defaults (override at runtime) ───────────────────────────────
ENV PORT=7860
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
# HF_TOKEN must be provided at runtime

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

USER appuser
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "customer_support_env.server:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
