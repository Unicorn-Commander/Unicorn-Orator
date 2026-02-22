# Unicorn Orator - AMD NPU Quick Start

**Professional AI Voice Synthesis Optimized for AMD Phoenix NPU**

## One-Command Install

```bash
git clone https://github.com/Unicorn-Commander/Unicorn-Orator.git
cd Unicorn-Orator
chmod +x install-amd-npu.sh
./install-amd-npu.sh
```

That's it! The script automatically:
- ✅ Downloads models (~320 MB)
- ✅ Builds Docker image
- ✅ Starts service on port 8880
- ✅ Verifies NPU acceleration

## System Requirements

### Hardware
- **CPU**: AMD Ryzen 9 8945HS, Ryzen AI 300 Series, or any AMD CPU with Phoenix NPU
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB for models and image

### Software
- **OS**: Ubuntu 24.10+ (or compatible Linux)
- **NPU Drivers**: XRT and NPU firmware installed
- **Docker**: Docker Engine 20.10+ with Compose V2
- **Permissions**: User in `docker` group, access to `/dev/accel/accel0`

## Quick Test

After installation:

```bash
# Test via WebGUI
open http://localhost:8880

# Test via API
curl -X POST http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world!","voice":"af_heart","speed":1.0}' \
  -o hello.wav && aplay hello.wav
```

## Performance

| Backend | Speed | Quality | Power |
|---------|-------|---------|-------|
| **AMD NPU** | **13x realtime** | High (INT8) | Low |
| CPU | 1-2x realtime | High | Medium |
| Intel iGPU | 3-5x realtime | High | Medium |

## Voices

50+ professional voices including:

- **American**: af_bella, am_michael, af_sarah, am_adam
- **British**: bf_emma, bm_george
- **Special**: af_sky, af_heart, af_nova, am_echo

Full voice list: http://localhost:8880

## API Usage

### OpenAI Compatible

```python
import requests

response = requests.post('http://localhost:8880/v1/audio/speech',
    json={
        'text': 'Welcome to Unicorn Orator!',
        'voice': 'af_bella',
        'speed': 1.0
    }
)

with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### Advanced Options

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Professional quality speech synthesis",
    "voice": "bf_emma",
    "speed": 1.2,
    "sample_rate": 24000
  }' -o output.wav
```

## Management

```bash
# View logs
docker compose logs -f unicorn-orator

# Restart service
docker compose restart unicorn-orator

# Stop service
docker compose down

# Update to latest
git pull
./install-amd-npu.sh
```

## Troubleshooting

### NPU Not Detected

**Check NPU device:**
```bash
ls -la /dev/accel/accel0
```

**Check XRT installation:**
```bash
xrt-smi examine
```

**Set NPU to high performance:**
```bash
xrt-smi configure --device 0 --power-mode high
```

### Service Not Starting

**Check Docker logs:**
```bash
docker compose logs unicorn-orator
```

**Verify ports available:**
```bash
sudo netstat -tlnp | grep 8880
```

**Restart Docker:**
```bash
sudo systemctl restart docker
```

### Models Not Found

**Re-download models:**
```bash
cd kokoro-tts
rm -rf models/*.bin models/*.onnx
./download_npu_models.sh
docker compose build --no-cache
```

## Integration Examples

### Python
```python
from kokoro import Kokoro

tts = Kokoro("http://localhost:8880")
audio = tts.speak("Hello world!", voice="af_bella")
audio.save("output.wav")
```

### JavaScript/Node.js
```javascript
const response = await fetch('http://localhost:8880/v1/audio/speech', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Hello from Node.js!',
    voice: 'am_michael'
  })
});
const buffer = await response.arrayBuffer();
```

### cURL
```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d @request.json -o response.wav
```

## Architecture

```
┌─────────────────────────────────────────┐
│          Web Interface :8880            │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         FastAPI TTS Server              │
│    (server.py + kokoro_mlir_npu.py)    │
└────────────────┬────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
┌────────▼─────┐  ┌──────▼──────┐
│  AMD Phoenix │  │   Kokoro    │
│  NPU (XDNA1) │  │  TTS Model  │
│  13x RT      │  │   (ONNX)    │
└──────────────┘  └─────────────┘
```

## File Structure

```
Unicorn-Orator/
├── install-amd-npu.sh           # Automated installer
├── README-AMD-NPU.md            # This file
├── docker-compose.yml           # Service orchestration
└── kokoro-tts/
    ├── download_npu_models.sh   # Model downloader
    ├── Dockerfile.amd-npu       # NPU-optimized build
    ├── requirements.npu.txt     # NPU dependencies
    ├── server.py                # TTS server
    ├── kokoro_mlir_npu.py      # NPU acceleration
    ├── npu/                     # NPU runtime
    ├── static/                  # Web interface
    └── models/                  # Downloaded models
```

## Contributing

Found a bug or want to contribute?
- **GitHub**: https://github.com/Unicorn-Commander/Unicorn-Orator
- **Issues**: https://github.com/Unicorn-Commander/Unicorn-Orator/issues

## License

MIT License - See LICENSE file

## Credits

- **Kokoro TTS**: High-quality multilingual TTS model
- **AMD XDNA**: NPU acceleration framework
- **Magic Unicorn**: Professional AI infrastructure

---

**Magic Unicorn Unconventional Technology & Stuff Inc**
*Professional AI Voice Synthesis for AMD Ryzen AI*
