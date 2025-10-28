# ==========================================
# Dockerfile pour ViT+MLP + Mediapipe + Firebase
# Version simplifiée sans docker-cmd
# ==========================================

FROM python:3.11-slim

# Install system dependencies for OpenCV, Mediapipe, and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 5000

# Start application directly with gunicorn
CMD gunicorn main:app \
    --bind 0.0.0.0:${PORT:-5000} \
    --workers 2 \
    --threads 2 \
    --timeout 180 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --preload \
    --log-level info \
    --access-logfile - \
    --error-logfile -