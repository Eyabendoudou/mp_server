FROM python:3.11-slim

# Installer dépendances système nécessaires à OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn main:app --bind 0.0.0.0:${PORT:-8080}