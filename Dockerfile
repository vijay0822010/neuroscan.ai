# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /build

# Copy package files first for layer caching
COPY frontend/package.json ./
RUN npm install

# Copy source and build
COPY frontend/ ./
RUN npm run build
# Result: /build/dist/


# ── Stage 2: Python backend + built frontend ──────────────────────────────────
FROM python:3.11-slim AS production
WORKDIR /app

# System packages needed by Pillow and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    zlib1g-dev \
    libjpeg-dev \
    libfreetype6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Application source
COPY app/          ./app/
COPY run.py        .

# Copy built frontend from Stage 1 into the location FastAPI expects
COPY --from=frontend-build /build/dist ./frontend/dist

# Runtime directories (Railway filesystem is ephemeral — use /tmp)
RUN mkdir -p /tmp/neuroscan_uploads /tmp/neuroscan_reports

# Railway injects PORT at runtime
ENV PORT=8000
ENV HOST=0.0.0.0
EXPOSE 8000

# Health check so Railway knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["python", "run.py"]
