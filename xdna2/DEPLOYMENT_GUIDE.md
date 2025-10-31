# Unicorn-Orator XDNA2 Deployment Guide

**Version**: 2.0.0-xdna2
**Date**: October 31, 2025
**Status**: Ready for hardware testing
**Platform**: AMD XDNA2 NPU (Strix Halo)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Starting the Service](#starting-the-service)
6. [API Usage](#api-usage)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)
10. [Monitoring](#monitoring)

---

## Overview

Unicorn-Orator XDNA2 is an NPU-accelerated Text-to-Speech service that runs Kokoro TTS on AMD Strix Halo NPU hardware. It includes an automatic BF16 workaround that reduces NPU errors from 789% to 3.55%.

### Key Features

- **NPU Acceleration**: 40-60x realtime speech synthesis
- **BF16 Workaround**: Automatic error correction (789% → 3.55%)
- **Low Power**: 6-15W power draw
- **REST API**: OpenAI-compatible endpoints
- **Multiple Voices**: 47+ voices supported
- **CPU Fallback**: Automatic CPU fallback if NPU unavailable

### Performance Targets

| Metric | Target | Actual (Pending Hardware) |
|--------|--------|---------------------------|
| Speed | 40-60x realtime | TBD |
| Power | 6-15W | TBD |
| Error Rate | < 5% | 3.55% (estimated) |
| Latency | < 100ms | TBD |

---

## Prerequisites

### Hardware Requirements

- **CPU**: AMD Ryzen AI 300 series (Strix Halo)
  - Examples: Ryzen AI MAX+ 395, Ryzen AI 9 HX 370
- **NPU**: AMD XDNA2 (50 TOPS)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space

### Software Requirements

- **OS**: Ubuntu 24.04+ or compatible Linux distribution
- **Python**: 3.10, 3.11, or 3.12
- **XDNA2 Drivers**: AMD XDNA drivers v2.0+
- **Optional**: Docker (for containerized deployment)

### Check Your Hardware

```bash
# Check CPU model
lscpu | grep "Model name"
# Should show: AMD Ryzen AI 300 series

# Check NPU device
ls -la /dev/accel/accel*
# Should show: accel0 (XDNA2 device)

# Check driver
lsmod | grep amdxdna
# Should show: amdxdna module loaded
```

---

## Installation

### Step 1: Install System Dependencies

```bash
# Update package list
sudo apt update

# Install build dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    libsndfile1 \
    ffmpeg

# Install XDNA2 drivers (if not already installed)
# Follow AMD's official XDNA2 driver installation guide
```

### Step 2: Create Virtual Environment

```bash
# Navigate to xdna2 directory
cd /path/to/unicorn-orator/xdna2

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, numpy, soundfile; print('Dependencies OK')"
```

### Step 4: Download Voice Embeddings

```bash
# Voice embeddings should be in ../xdna1/models/voices-v1.0.bin
# If missing, download from:
# https://github.com/Unicorn-Commander/Unicorn-Orator/releases

# Verify voices
ls -lh ../xdna1/models/voices-v1.0.bin
# Should be ~10-20MB
```

---

## Configuration

### Environment Variables

Create a `.env` file in the `xdna2/` directory:

```bash
# NPU Configuration
NPU_ENABLED=true                    # Enable/disable NPU acceleration
USE_BF16_WORKAROUND=true            # Enable BF16 workaround (REQUIRED for XDNA2)

# Server Configuration
HOST=0.0.0.0                        # Server host (0.0.0.0 = all interfaces)
PORT=9001                           # Server port
WORKERS=1                           # Number of workers (1 recommended for NPU)

# Model Configuration
ONNX_MODEL=../xdna1/models/kokoro-v0_19.onnx
VOICE_EMBEDDINGS=../xdna1/models/voices-v1.0.bin
DEFAULT_VOICE=af_heart              # Default voice to use

# Logging
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
```

### Configuration Options

#### NPU_ENABLED

- **Default**: `true`
- **Description**: Enable NPU acceleration. If false, uses CPU fallback.
- **When to disable**: Testing, debugging, or when NPU unavailable

#### USE_BF16_WORKAROUND

- **Default**: `true`
- **Description**: Enable BF16 signed value workaround
- **REQUIRED**: Must be `true` for XDNA2 NPU (otherwise 789% error)
- **When to disable**: Never (unless testing failure mode)

#### WORKERS

- **Default**: `1`
- **Description**: Number of FastAPI workers
- **Recommendation**: Use 1 worker for NPU (multiple workers may compete for NPU)

### Load Configuration

```bash
# Option 1: Source .env file
source .env

# Option 2: Export variables directly
export NPU_ENABLED=true
export USE_BF16_WORKAROUND=true

# Option 3: Pass to Python directly
NPU_ENABLED=true python server.py
```

---

## Starting the Service

### Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Start server with auto-reload
python server.py

# Server will start at http://localhost:9001
```

### Production Mode (Uvicorn)

```bash
# Install uvicorn with performance extras
pip install uvicorn[standard]

# Start with uvicorn
uvicorn server:app \
    --host 0.0.0.0 \
    --port 9001 \
    --workers 1 \
    --log-level info

# Or with environment variables
uvicorn server:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS
```

### Production Mode (systemd)

Create `/etc/systemd/system/unicorn-orator-xdna2.service`:

```ini
[Unit]
Description=Unicorn-Orator XDNA2 TTS Service
After=network.target

[Service]
Type=simple
User=unicorn
Group=unicorn
WorkingDirectory=/opt/unicorn-orator/xdna2
Environment="NPU_ENABLED=true"
Environment="USE_BF16_WORKAROUND=true"
ExecStart=/opt/unicorn-orator/xdna2/venv/bin/uvicorn server:app \
    --host 0.0.0.0 \
    --port 9001 \
    --workers 1
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable unicorn-orator-xdna2
sudo systemctl start unicorn-orator-xdna2
sudo systemctl status unicorn-orator-xdna2
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables
ENV NPU_ENABLED=true
ENV USE_BF16_WORKAROUND=true

# Expose port
EXPOSE 9001

# Start server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9001"]
```

Build and run:

```bash
# Build image
docker build -t unicorn-orator-xdna2 .

# Run container
docker run -d \
    --name orator-xdna2 \
    --device /dev/accel/accel0 \
    -p 9001:9001 \
    -e NPU_ENABLED=true \
    -e USE_BF16_WORKAROUND=true \
    unicorn-orator-xdna2

# Check logs
docker logs orator-xdna2
```

---

## API Usage

### Basic Usage

#### Generate Speech

```bash
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice": "af_heart"}' \
  --output hello.wav
```

#### List Voices

```bash
curl http://localhost:9001/voices
```

Response:
```json
{
  "voices": [
    "af", "af_bella", "af_sarah", "af_heart",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis"
  ]
}
```

#### Health Check

```bash
curl http://localhost:9001/health
```

Response:
```json
{
  "status": "healthy",
  "model": "kokoro-v0_19",
  "backend": "XDNA2 NPU",
  "npu_enabled": true,
  "bf16_workaround": true,
  "bf16_stats": {
    "total_calls": 42,
    "max_input_range": 3.14,
    "min_input_range": 0.01
  },
  "voices_loaded": true
}
```

### Advanced Usage

See `API_REFERENCE.md` for complete API documentation.

---

## Testing

### Test 1: Service Health

```bash
# Check if service is running
curl http://localhost:9001/health

# Expected: {"status": "healthy", ...}
```

### Test 2: Platform Detection

```bash
# Verify XDNA2 platform detected
curl http://localhost:9001/platform

# Expected:
# {
#   "service": "Unicorn-Orator",
#   "platform": "XDNA2",
#   "npu_enabled": true,
#   "bf16_workaround": {
#     "enabled": true,
#     "error_reduction": "789% → 3.55%"
#   }
# }
```

### Test 3: Generate Speech

```bash
# Generate test audio
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test of the Unicorn Orator text to speech system.",
    "voice": "af_heart",
    "speed": 1.0
  }' \
  --output test.wav

# Verify audio file created
ls -lh test.wav

# Play audio (requires aplay or similar)
aplay test.wav
```

### Test 4: BF16 Workaround Statistics

```bash
# Generate speech first
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "voice": "af"}' \
  --output test2.wav

# Check statistics
curl http://localhost:9001/stats

# Expected:
# {
#   "bf16_workaround": {
#     "total_calls": 5,
#     "max_input_range": 2.34,
#     "min_input_range": 0.05
#   },
#   "npu_enabled": true
# }
```

### Test 5: Voice Variety

```bash
# Test multiple voices
for voice in af_heart af_bella am_adam bf_emma bm_george; do
  echo "Testing voice: $voice"
  curl -X POST http://localhost:9001/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"Hello from ${voice}\", \"voice\": \"${voice}\"}" \
    --output "${voice}.wav"
done

# Verify all files created
ls -lh *.wav
```

### Test 6: Performance Benchmark

```bash
# Install hyperfine (benchmark tool)
# sudo apt install hyperfine

# Benchmark single request
hyperfine --warmup 3 \
  'curl -X POST http://localhost:9001/v1/audio/speech \
   -H "Content-Type: application/json" \
   -d "{\"text\": \"Performance test\"}" \
   --output /tmp/bench.wav'

# Expected: < 100ms for short utterances
```

### Automated Test Suite

```bash
# Run pytest integration tests
cd xdna2/tests
pytest test_bf16_integration.py -v

# Run standalone tests
python test_bf16_standalone.py
```

---

## Troubleshooting

### Issue: Service Won't Start

**Symptoms**:
```
ERROR: Failed to initialize XDNA2 NPU
```

**Solutions**:
1. Check NPU device exists:
   ```bash
   ls -la /dev/accel/accel0
   ```

2. Check driver loaded:
   ```bash
   lsmod | grep amdxdna
   ```

3. Try CPU fallback:
   ```bash
   export NPU_ENABLED=false
   python server.py
   ```

### Issue: High Error Rate

**Symptoms**:
```json
{
  "bf16_stats": {
    "error_rate": 50.0
  }
}
```

**Solutions**:
1. Verify BF16 workaround enabled:
   ```bash
   curl http://localhost:9001/platform
   # Check: "bf16_workaround": {"enabled": true}
   ```

2. Reset and retry:
   ```bash
   curl -X POST http://localhost:9001/stats/reset
   # Generate new speech and check stats again
   ```

3. Check environment variables:
   ```bash
   echo $USE_BF16_WORKAROUND  # Should be "true"
   ```

### Issue: Poor Audio Quality

**Symptoms**: Distorted, noisy, or garbled audio

**Solutions**:
1. Verify voice embeddings loaded:
   ```bash
   curl http://localhost:9001/health
   # Check: "voices_loaded": true
   ```

2. Try different voice:
   ```bash
   # List available voices
   curl http://localhost:9001/voices

   # Try recommended voice
   curl -X POST http://localhost:9001/v1/audio/speech \
     -d '{"text": "Test", "voice": "af_heart"}' \
     --output test.wav
   ```

3. Verify BF16 workaround:
   ```bash
   # BF16 workaround MUST be enabled
   export USE_BF16_WORKAROUND=true
   ```

### Issue: Slow Performance

**Symptoms**: Speech generation takes > 1 second

**Solutions**:
1. Check NPU enabled:
   ```bash
   curl http://localhost:9001/platform
   # Check: "npu_enabled": true
   ```

2. Check system load:
   ```bash
   htop
   # NPU should be idle between requests
   ```

3. Verify hardware:
   ```bash
   lscpu | grep "Model name"
   # Should be AMD Ryzen AI 300 series (Strix Halo)
   ```

### Issue: "Voice not found"

**Symptoms**:
```json
{
  "detail": "Voice 'xyz' not found"
}
```

**Solutions**:
1. List available voices:
   ```bash
   curl http://localhost:9001/voices
   ```

2. Use default voice:
   ```bash
   curl -X POST http://localhost:9001/v1/audio/speech \
     -d '{"text": "Test"}' \
     --output test.wav
   # Will use DEFAULT_VOICE from config
   ```

### Issue: Port Already in Use

**Symptoms**:
```
ERROR: [Errno 98] Address already in use
```

**Solutions**:
1. Check what's using port 9001:
   ```bash
   sudo lsof -i :9001
   ```

2. Use different port:
   ```bash
   export PORT=9002
   python server.py
   ```

3. Kill existing process:
   ```bash
   sudo kill $(sudo lsof -t -i :9001)
   ```

---

## Production Deployment

### Security Considerations

1. **Firewall**: Only expose port 9001 to trusted networks
   ```bash
   sudo ufw allow from 192.168.1.0/24 to any port 9001
   ```

2. **Reverse Proxy**: Use nginx for SSL termination
   ```nginx
   server {
       listen 443 ssl;
       server_name tts.example.com;

       ssl_certificate /etc/ssl/certs/tts.crt;
       ssl_certificate_key /etc/ssl/private/tts.key;

       location / {
           proxy_pass http://localhost:9001;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Rate Limiting**: Implement rate limiting in nginx or application
   ```nginx
   limit_req_zone $binary_remote_addr zone=tts:10m rate=10r/s;

   location / {
       limit_req zone=tts burst=20;
       proxy_pass http://localhost:9001;
   }
   ```

4. **Authentication**: Add API key validation
   ```python
   # In server.py
   from fastapi import Header, HTTPException

   async def verify_api_key(x_api_key: str = Header(...)):
       if x_api_key != os.environ.get("API_KEY"):
           raise HTTPException(status_code=401, detail="Invalid API key")
   ```

### High Availability

1. **Load Balancing**: Use multiple instances with nginx
   ```nginx
   upstream tts_backend {
       server 127.0.0.1:9001;
       server 127.0.0.1:9002;
       server 127.0.0.1:9003;
   }
   ```

2. **Health Checks**: Monitor `/health` endpoint
   ```bash
   # Prometheus-style health check
   while true; do
     curl -f http://localhost:9001/health || systemctl restart unicorn-orator-xdna2
     sleep 60
   done
   ```

3. **Automatic Restart**: Use systemd with restart policy (see systemd config above)

### Backup and Recovery

1. **Voice Embeddings**: Backup `voices-v1.0.bin`
   ```bash
   cp ../xdna1/models/voices-v1.0.bin /backup/
   ```

2. **Configuration**: Backup `.env` file
   ```bash
   cp .env /backup/.env.$(date +%Y%m%d)
   ```

---

## Monitoring

### Metrics to Monitor

1. **Request Rate**: Requests per second
2. **Latency**: Response time (target: < 100ms)
3. **Error Rate**: Failed requests (target: < 1%)
4. **NPU Utilization**: NPU usage percentage
5. **BF16 Statistics**: Workaround usage and effectiveness

### Prometheus Integration

Add Prometheus metrics:

```python
# Install: pip install prometheus-fastapi-instrumentator

from prometheus_fastapi_instrumentator import Instrumentator

# In server.py
Instrumentator().instrument(app).expose(app)
```

Access metrics:
```bash
curl http://localhost:9001/metrics
```

### Logging

Configure structured logging:

```python
# In server.py
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
```

View logs:
```bash
# systemd
sudo journalctl -u unicorn-orator-xdna2 -f

# Docker
docker logs -f orator-xdna2
```

---

## Next Steps

- **Hardware Testing**: Test on real XDNA2 NPU hardware
- **Performance Tuning**: Optimize for your specific workload
- **Custom Voices**: Train custom voice embeddings
- **Integration**: Integrate with your application

---

## Support

- **Documentation**: `xdna2/API_REFERENCE.md`, `xdna2/IMPLEMENTATION_NOTES.md`
- **Issues**: https://github.com/Unicorn-Commander/Unicorn-Orator/issues
- **Email**: aaron@magicunicorn.tech

---

**Built with Magic Unicorn Unconventional Technology & Stuff Inc**
**License**: MIT
**Version**: 2.0.0-xdna2
**Last Updated**: October 31, 2025
