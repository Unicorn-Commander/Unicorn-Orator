FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install WhisperX and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix version compatibility issues
RUN pip install --upgrade "pytorch-lightning>=2.0" "pyannote.audio>=3.1.0"

COPY . .

EXPOSE 9000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
