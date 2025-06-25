# ML Service Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY ml_cluster.py .

# Create directories for models and logs
RUN mkdir -p models logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app

USER mluser

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "ml_cluster.py", "--mode", "train"]