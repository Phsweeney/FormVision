# FormVision backend
FROM python:3.12-slim

# OpenCV's wheels link against system libraries that the slim image omits.
# Without libGL and libglib, `import cv2` fails at runtime with an opaque
# "libGL.so.1: cannot open shared object file" — a classic first-deploy trap.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are copied and installed before the source so that editing code
# does not invalidate the (slow) MediaPipe install layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY pyproject.toml ./

# Runtime artefacts and the cached pose model. Declared as volumes in
# docker-compose so uploads and analyses survive a container rebuild.
RUN mkdir -p /app/data /app/models

# Run as a non-root user. The data directories are chowned first, since the
# app writes to them.
RUN useradd --create-home --uid 1000 formvision \
    && chown -R formvision:formvision /app
USER formvision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
