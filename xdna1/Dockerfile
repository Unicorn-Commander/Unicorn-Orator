FROM python:3.10-slim

# Install dependencies including Intel GPU runtime
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    libgomp1 \
    libtbb12 \
    ocl-icd-libopencl1 \
    espeak-ng \
    # Intel GPU dependencies
    intel-opencl-icd \
    clinfo \
    vainfo \
    # Additional libraries for OpenVINO
    libpugixml1v5 \
    libusb-1.0-0 \
    libnuma1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory (models will be mounted from host)
RUN mkdir -p models

COPY . .

EXPOSE 8880

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8880"]
