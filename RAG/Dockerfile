# Build stage: install Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Final image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from build stage
COPY --from=builder /install /usr/local

# Copy application source
COPY rag_demo/ ./rag_demo/

# Pre-download the sentence-transformer models so the container works offline
# (remove or comment out if you want a smaller image and have internet at runtime)
RUN python -c "\
import os; \
model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'); \
from sentence_transformers import SentenceTransformer; SentenceTransformer(model)" \
  || echo "WARNING: model pre-cache failed — will download at runtime"

WORKDIR /app/rag_demo

EXPOSE 5000

# Bind to 0.0.0.0 and respect Render's dynamic $PORT (default 5000 for local use)
ENV PORT=5000

# Use Gunicorn as the production WSGI server
# --workers: parallel worker processes (adjust based on available CPUs)
# --timeout: allow extra time for model loading on first request
CMD gunicorn --bind "0.0.0.0:${PORT}" --workers 2 --timeout 120 app:app
