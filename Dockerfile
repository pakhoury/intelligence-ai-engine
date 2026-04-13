FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Oracle + PostgreSQL + PGVector
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libaio1t64 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy project files
COPY src/rag-system/ .
COPY documents/ ./documents/
COPY metadata_mapping.json .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]