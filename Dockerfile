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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

# Command to run the application
CMD ["python", "run_api.py"] 