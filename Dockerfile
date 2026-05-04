FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirement.txt

FROM python:3.11-slim

WORKDIR /app

# Runtime-only system deps (no build-essential)
RUN apt-get update && apt-get install -y \
    libpq5 \
    libaio1t64 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /install /usr/local

# Copy application code
COPY src/rag-system/ .
COPY documents/ ./documents/
COPY metadata_mapping.json .

# Create writable cache for HuggingFace models
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app
ENV HF_HOME=/app/.cache

# Run as non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
