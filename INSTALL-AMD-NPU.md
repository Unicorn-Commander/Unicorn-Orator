# Unicorn Orator - AMD NPU Installation Guide

**For AMD Ryzen AI Systems with Phoenix NPU (XDNA1)**

This guide is for installing Unicorn Orator with NPU acceleration on systems like:
- AMD Ryzen 9 8945HS
- AMD Ryzen AI 300 Series
- Any AMD CPU with Phoenix NPU

## Prerequisites

- AMD Ryzen AI CPU with Phoenix NPU
- Ubuntu 24.10+ (or compatible Linux)
- NPU drivers installed (`/dev/accel/accel0` should exist)
- Docker and Docker Compose installed
- Python 3 with `huggingface-hub` package

## Quick Install (3 Steps)

### 1. Clone the Repository

```bash
git clone https://github.com/Unicorn-Commander/Unicorn-Orator.git
cd Unicorn-Orator/kokoro-tts
```

### 2. Download NPU Models

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download NPU-optimized models (~150 MB total)
chmod +x download_npu_models.sh
./download_npu_models.sh
```

This downloads:
- `voices-v1.0.bin` (26.9 MB) - Voice embeddings
- `kokoro-npu-quantized-int8.onnx` (121.9 MB) - INT8 quantized model for NPU
- `kokoro-v0_19.onnx` (311 MB) - Standard model (fallback)

### 3. Build and Start

```bash
# Build Docker image with NPU support
docker compose -f docker-compose.yml build

# Start the service
docker compose up -d

# Check status
docker compose logs -f
```

## Verify NPU Acceleration

### Check Health Endpoint

```bash
curl http://localhost:8880/health
```

You should see:
```json
{
  "status": "healthy",
  "model": "kokoro-v0_19",
  "backend": "NPU (INT8)",
  "npu_enabled": true,
  "model_loaded": true,
  "voices_loaded": true
}
```

### Check WebGUI

Open http://localhost:8880 in your browser.

Under **Advanced Settings → Processing Backend**, you should see:
```
AMD XDNA1 Phoenix NPU (13x realtime)
```

## Performance

With NPU acceleration:
- **Speed**: 13x realtime synthesis
- **Quality**: INT8 quantized (minimal quality loss)
- **Power**: Low power consumption vs GPU

## Troubleshooting

### Models Not Found

**Symptom**: "voices-v1.0.bin not found" in logs

**Fix**:
```bash
cd Unicorn-Orator/kokoro-tts
./download_npu_models.sh
docker compose build --no-cache
```

### NPU Not Detected

**Symptom**: `npu_enabled: false` in health check

**Checks**:
1. NPU device exists: `ls -la /dev/accel/accel0`
2. XRT runtime installed: `xrt-smi examine`
3. Container has device access (check docker-compose.yml)

**Fix** (add to docker-compose.yml):
```yaml
services:
  unicorn-orator:
    devices:
      - /dev/accel/accel0:/dev/accel/accel0
      - /dev/dri:/dev/dri
```

### CPU Fallback Working

**Symptom**: Service works but shows "CPUExecutionProvider"

This is normal if:
- NPU device not accessible
- Models not optimized for NPU
- XRT runtime not installed

The service will work with CPU, just slower (1-2x realtime instead of 13x).

## File Structure

```
Unicorn-Orator/
└── kokoro-tts/
    ├── download_npu_models.sh    # Download script
    ├── Dockerfile.amd-npu         # NPU-optimized Dockerfile
    ├── requirements.npu.txt       # NPU dependencies
    ├── server.py                  # TTS server with NPU support
    ├── kokoro_mlir_npu.py        # NPU acceleration module
    ├── npu/                       # NPU runtime files
    └── models/                    # Downloaded models go here
        ├── voices-v1.0.bin
        ├── kokoro-npu-quantized-int8.onnx
        └── kokoro-v0_19.onnx
```

## Integration with UC-1

If using as part of UC-1 Pro:

```bash
cd /home/ucadmin/UC-1
docker compose -f docker-compose-uc1-optimized.yml up -d unicorn-orator
```

The NPU service will be available at:
- WebGUI: http://localhost:8880
- API: http://localhost:8880/v1/audio/speech

## Advanced Configuration

### Use FP16 Model (Higher Quality)

Edit `server.py` line 108:
```python
npu_model = "models/kokoro-npu-fp16.onnx"  # Instead of INT8
```

Then download the FP16 model:
```bash
cd models
python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('magicunicorn/kokoro-npu-quantized', \
    'kokoro-npu-fp16.onnx', local_dir='.')"
```

### Set NPU to Max Performance

```bash
# Find xrt-smi
XRT_SMI=$(find /opt -name xrt-smi 2>/dev/null | head -1)

# Set high performance mode
$XRT_SMI configure --device 0 --power-mode high

# Verify
$XRT_SMI examine | grep -i power
```

## Support

- GitHub Issues: https://github.com/Unicorn-Commander/Unicorn-Orator/issues
- Documentation: https://github.com/Unicorn-Commander/Unicorn-Orator

## License

MIT License - See LICENSE file

---

**Magic Unicorn Unconventional Technology & Stuff Inc**
Professional AI Voice Synthesis for AMD Ryzen AI
