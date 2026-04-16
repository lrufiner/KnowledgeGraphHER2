FROM python:3.11-slim

# System deps (for PyMuPDF, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default: run the API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000 8501
