FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    portaudio19-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA repository for CUDA and cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-8 \
    libcudnn8 \
    libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f cuda-keyring_1.0-1_all.deb

# Set CUDA related environment variables
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set working directory
WORKDIR /app

# Install all dependencies from the modified requirements file (without PyAudio)
COPY requirements_no_audio.txt .
RUN pip install --no-cache-dir -r requirements_no_audio.txt

# Install whisperx with the correct version
RUN pip install --no-cache-dir whisperx==3.3.0

# Ensure the correct faster-whisper version is installed last to satisfy RealtimeSTT
RUN pip install --no-cache-dir faster-whisper==1.1.1

# Install pydub for audio processing
RUN pip install --no-cache-dir pydub

# Copy application code
COPY . .

# Expose port for the API
EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python", "run_api.py"] 