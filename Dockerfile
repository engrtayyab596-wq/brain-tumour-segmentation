# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — build dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed for Pillow / TIFF support
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libtiff5-dev \
        libwebp-dev \
        libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix we can copy later
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="engrtayyab596@gmail.com"
LABEL description="Brain Tumour Segmentation API — U-Net on LGG MRI (val Dice 0.8586)"
LABEL version="1.0.0"

# Runtime system libraries (Pillow TIFF + libgomp for PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libtiff5 \
        libwebp7 \
        libopenjp2-7 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Working directory
WORKDIR /app

# Copy source tree
COPY src/   ./src/
COPY api/   ./api/
COPY models/ ./models/

# ── Environment variables ────────────────────────────────────────────────────
ENV MODEL_PATH=/app/models/best_model.pth \
    IMG_SIZE=256 \
    THRESHOLD=0.5 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check — hits the / endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Start the API server
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
