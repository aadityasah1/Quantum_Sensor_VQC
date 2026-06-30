# Quantum Sensor Fault Classifier - REST API image
FROM python:3.11-slim

# Faster, cleaner Python in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + the trained model artifacts
COPY qsensor ./qsensor
COPY artifacts ./artifacts

# Drop root for runtime
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Simple container healthcheck against the API
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "qsensor.api:app", "--host", "0.0.0.0", "--port", "8000"]
