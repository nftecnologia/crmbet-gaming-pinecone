# ETL Service Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY run_pipeline.py .

# Create directories for logs
RUN mkdir -p logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash etluser && \
    chown -R etluser:etluser /app

USER etluser

# Health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
  CMD python -c "import psycopg2; print('ETL service healthy')" || exit 1

# Default command
CMD ["python", "run_pipeline.py", "--mode", "schedule"]