# Unicorn Orator - Intel iGPU Setup

## Intel iGPU Optimization

This configuration is optimized for Intel Integrated Graphics (UHD Graphics 770 and similar) with OpenVINO acceleration.

### Hardware Support
- **Intel UHD Graphics 770** (tested and working)
- Intel Iris Xe Graphics (96 EU and higher)
- Intel Arc iGPU (128 EU) - Meteor Lake
- Intel UHD Graphics (32 EU) - Budget systems

### Performance
- **3-5x faster** than CPU inference
- **~150ms per sentence** (FP16 precision)
- **Memory usage**: <500MB total
- **Model size**: 311MB ONNX + 25MB voice embeddings

## Quick Start (Intel iGPU)

### Using Docker Compose
```bash
# Clone repository
git clone https://github.com/Unicorn-Commander/Unicorn-Orator.git
cd Unicorn-Orator

# Start with Intel iGPU optimization
docker-compose -f docker-compose-intel-igpu.yml up -d

# Check logs
docker-compose -f docker-compose-intel-igpu.yml logs -f
```

### Using Pre-built Docker Image
```bash
docker run -d \
  --name unicorn-orator \
  -p 8885:8880 \
  --device /dev/dri:/dev/dri \
  --group-add 44 --group-add 993 \
  -e DEVICE=IGPU \
  -e OPENVINO_PRECISION=FP16 \
  -e COMPUTE_TYPE=float16 \
  magicunicorn/unicorn-orator:intel-igpu-v2.0
```

## Configuration

### Environment Variables
```bash
DEVICE=IGPU                    # Use Intel iGPU
OPENVINO_PRECISION=FP16        # FP16 for optimal iGPU performance
COMPUTE_TYPE=float16           # Float16 compute type
TTS_MODEL=kokoro               # Kokoro TTS model
DEFAULT_VOICE=af               # Default voice
```

### Prerequisites
The Docker image includes all required dependencies:
- OpenVINO Runtime 2024.0+
- Intel GPU drivers
- ONNX Runtime with OpenVINO provider

### Verification
Access the web interface at http://localhost:8885 and verify:
- Status: "Intel iGPU (OpenVINO FP16)" in the processing backend
- Health check returns `{"backend": "OpenVINO iGPU (FP16)"}`

## API Usage

### OpenAI Compatible
```bash
curl -X POST http://localhost:8885/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello from Intel iGPU accelerated TTS!",
    "voice": "af",
    "speed": 1.0
  }' \
  --output speech.wav
```

### Native API
```bash
curl -X POST http://localhost:8885/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Testing Intel iGPU acceleration",
    "voice": "af_heart",
    "speed": 1.0
  }' \
  --output test.wav
```

## Performance Comparison

| Backend | Inference Time | Memory Usage | Hardware |
|---------|---------------|--------------|----------|
| CPU | ~500ms | 800MB | Any x86_64 |
| Intel iGPU | ~150ms | 500MB | UHD 770+ |
| XDNA NPU | ~80ms | 300MB | Phoenix/Strix |

## Troubleshooting

### Check iGPU Detection
```bash
docker exec unicorn-orator clinfo | grep "Device Name"
# Should show: Intel(R) UHD Graphics 770 [0x4680]
```

### Verify OpenVINO GPU Support
```bash
curl http://localhost:8885/health | jq .backend
# Should return: "OpenVINO iGPU (FP16)"
```

### Common Issues
1. **"GPU device not found"**: Ensure `/dev/dri` is accessible
2. **"Permission denied"**: Add video/render groups (44, 993)
3. **"FP16 not supported"**: Use compatible Intel GPU (UHD 630+)

## Model Information
- **Source**: https://huggingface.co/magicunicorn/kokoro-tts-intel
- **Format**: ONNX (FP32 model, FP16 inference)
- **Voices**: 50+ professional voices
- **Languages**: English with various accents