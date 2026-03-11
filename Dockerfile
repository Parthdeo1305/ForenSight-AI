# ============================================================
# Dockerfile — ForenSight AI Backend (Hugging Face Spaces)
# Port: 7860 (HF Spaces requirement)
# ============================================================
FROM python:3.10-slim

# Install curl (required by HF Spaces init system)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (minimal — fast build)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

# Copy source code
COPY api/ ./api/
COPY .env.example ./.env.example

# Create directories
RUN mkdir -p uploads /data

# Set environment variables for deployment
ENV ENABLE_DOCS=true
ENV DATABASE_PATH=/data/forensight.db
ENV PYTHONUNBUFFERED=1

# Expose port 7860 (HF Spaces requirement)
EXPOSE 7860

# Start server (single worker to save memory)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
