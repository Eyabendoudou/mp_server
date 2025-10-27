FROM python:3.11-slim

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage build cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (includes docker-cmd and weights)
COPY . /app

# Ensure start command is executable (do this after final COPY)
RUN chmod +x /app/docker-cmd

CMD ["/app/docker-cmd"]