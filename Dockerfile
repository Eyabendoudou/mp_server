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

# copy only requirements first to leverage cache
COPY requirements.txt /app/

# Install PyTorch CPU wheel matching 2.8.0 and then install other requirements
# If you have a GPU-enabled environment and want CUDA, replace the wheel URL accordingly.
RUN pip install --no-cache-dir "torch==2.8.0+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

# copy the rest of the project (includes docker-cmd and model file)
COPY . /app

# Ensure start command is executable after final COPY
RUN chmod +x /app/docker-cmd

CMD ["/app/docker-cmd"]