# ==========================================
# Dockerfile pour ViT+MLP + Mediapipe + Firebase
# Optimisé pour Railway
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
    libgfortran5 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make docker-cmd executable
RUN chmod +x /app/docker-cmd

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Railway will override with $PORT)
EXPOSE 5000

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-5000}/health')"

# Run the application
CMD ["/app/docker-cmd"]