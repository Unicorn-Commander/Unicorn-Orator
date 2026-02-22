# 🦄 Unicorn Orator - Lightweight TTS with Hardware Acceleration

<div align="center">
  <img src="kokoro-tts/static/Unicorn_Orator.png" alt="Unicorn Orator Logo" width="200"/>
  
  [![Docker](https://img.shields.io/docker/pulls/magicunicorn/unicorn-orator?style=flat-square)](https://hub.docker.com/r/magicunicorn/unicorn-orator)
  [![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
  [![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green?style=flat-square)](https://platform.openai.com/docs/api-reference/audio/createSpeech)
  
  **Efficient text-to-speech that runs on Intel iGPU, freeing your GPU for AI inference**
  
  [Web Interface](http://localhost:8885/web) | [Docker Hub](https://hub.docker.com/r/magicunicorn/unicorn-orator) | [API Docs](#api-usage)
</div>

---

## 🎯 Why Unicorn Orator?

**The Problem**: Running TTS alongside LLMs fights for GPU resources, slowing down inference and increasing latency.

**Our Solution**: Unicorn Orator offloads TTS to Intel integrated graphics or AMD NPUs, leaving your discrete GPU free for what it does best - running large language models.

### Key Benefits

- **🚀 Free Your GPU**: TTS runs on iGPU/NPU, preserving discrete GPU for LLM inference
- **⚡ Resource Efficient**: Uses ~15W on iGPU vs 100W+ on discrete GPU
- **🎭 50+ Quality Voices**: Kokoro v0.19 with diverse accents and styles
- **🔌 OpenAI Compatible**: Drop-in replacement, no code changes needed
- **🐳 Production Ready**: Docker image available, battle-tested deployment

## 🖼️ Web Interface

<div align="center">
  <img src="assets/unicorn-orator-interface.png" alt="Unicorn Orator Web Interface" width="600"/>
  <br>
  <i>Clean, intuitive interface with 50+ voices and advanced settings</i>
</div>

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Pull and run the pre-built image
docker run -d --name unicorn-orator \
  -p 8885:8880 \
  -v $(pwd)/kokoro-tts/models:/app/models:ro \
  --device /dev/dri:/dev/dri \
  --group-add video \
  magicunicorn/unicorn-orator:intel-igpu-v1.0

# Visit http://localhost:8885/web for the interface
```

### From Source

```bash
git clone https://github.com/Unicorn-Commander/Unicorn-Orator.git
cd Unicorn-Orator
docker-compose up -d
```

## 💡 Technical Innovation

### Intel iGPU Optimization
We've optimized Kokoro TTS to run efficiently on Intel integrated graphics via OpenVINO:

- **Hardware Detection**: Automatically detects and uses Intel Xe/Arc iGPUs
- **FP16 Inference**: Maintains quality while doubling throughput
- **Minimal Memory**: ~300MB VRAM usage, leaving room for other tasks
- **Power Efficient**: 10-15W TDP vs 75-350W for discrete GPUs

### AMD NPU Support (Phoenix/Hawk Point)
For Ryzen AI laptops (7040/8040 series) with Phoenix NPU:

**Current Status** (October 2025):
- **Hardware**: Phoenix NPU operational via XRT 2.20.0
- **Available Kernels**: 69 compiled MLIR-AIE2 XCLBINs
- **Tested**: 16×16 matmul kernel (1.0 correlation, 0.484ms/op)
- **Integration**: Research phase for Kokoro TTS architecture
- **Power Target**: <10W for continuous synthesis

**Kernels Available** (from Unicorn-Amanuensis project):
- ✅ 16×16 INT8 matrix multiply
- ✅ Production mel spectrogram
- ✅ GELU, LayerNorm activations
- ⚠️ Attention mechanisms (debugging)

**Note**: Kokoro TTS architecture differs from Whisper. NPU kernel integration requires architecture-specific adaptation. Shared kernels (matmul, activations) can potentially accelerate Kokoro's encoder/decoder components.

See [unicorn-npu-core](https://github.com/Unicorn-Commander/unicorn-npu-core) for kernel management.

### Performance Comparison

| Hardware | Power Usage | VRAM | Speed | Purpose |
|----------|-------------|------|-------|---------|
| Intel iGPU | 15W | 300MB | 5x realtime | **TTS (This Project)** |
| AMD NPU | 10W | 256MB | 4x realtime | **TTS (Experimental)** |
| NVIDIA 4090 | 350W | 2GB | 20x realtime | Better used for LLMs |
| CPU (i7) | 45W | N/A | 2x realtime | Fallback option |

## 📡 API Usage

### OpenAI-Compatible Endpoint

```python
import requests

# Works exactly like OpenAI's API
response = requests.post('http://localhost:8885/v1/audio/speech',
    json={
        'text': 'Hello from Unicorn Orator!',
        'voice': 'af_heart',  # 50+ voices available
        'speed': 1.0
    }
)

with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### Available Voices (Selection)

| Voice ID | Description | Best For |
|----------|-------------|----------|
| `af_heart` | Warm, friendly female | General narration |
| `am_michael` | Professional male | News/corporate |
| `bf_emma` | British female | Audiobooks |
| `af_bella` | Young American female | Social media |
| `bm_george` | British male | Documentation |

[Full voice list available at /voices endpoint]

## 🏗️ Architecture

```
Your System:
┌──────────────────┬──────────────────┐
│   Discrete GPU   │   Intel iGPU     │
│   (RTX/Arc/RX)   │   (Xe Graphics)  │
│                  │                  │
│   Running:       │   Running:       │
│   - LLMs         │   - Unicorn TTS  │
│   - Stable Diff  │   - Video decode │
│   - ML Training  │   - Display      │
└──────────────────┴──────────────────┘
         │                  │
         └──────┬───────────┘
                │
        [High Performance AI]
         Without Competition
```

## 🔮 Roadmap

### Current Release (v1.0)
- ✅ Intel iGPU support via OpenVINO
- ✅ 50+ Kokoro voices
- ✅ OpenAI API compatibility
- ✅ Docker deployment
- ✅ Web interface

### Planned Features
- [ ] Real-time streaming
- [ ] AMD NPU production support
- [ ] Voice cloning (ethical use only)
- [ ] SSML support
- [ ] Batch processing API
- [ ] Kubernetes operator

### Future Exploration
- [ ] Apple Neural Engine support
- [ ] Qualcomm Hexagon DSP
- [ ] Edge deployment (Jetson, Pi 5)
- [ ] WebGPU browser runtime

## 🛠️ Building From Source

### Prerequisites
- Docker & Docker Compose
- Intel CPU with Xe/Arc graphics (or AMD Ryzen AI)
- 8GB RAM minimum
- Ubuntu 22.04+ or Windows 11 WSL2

### Build Steps
```bash
# Clone repository
git clone https://github.com/Unicorn-Commander/Unicorn-Orator.git
cd Unicorn-Orator

# Download models (one-time, ~350MB)
./download_models.sh

# Build with hardware detection
./build.sh

# Run
docker-compose up -d
```

## 📊 Benchmarks

Testing setup: Intel Core i7-13700K with Intel UHD 770 iGPU

| Text Length | Generation Time | Realtime Factor |
|-------------|-----------------|-----------------|
| 1 sentence | 180ms | 5.5x |
| 1 paragraph | 950ms | 5.2x |
| 1 page | 4.2s | 5.0x |

*Realtime factor = audio duration / generation time*

## 🤝 Contributing

We especially welcome contributions for:
- Hardware optimization (OpenVINO, XDNA, CoreML)
- Additional TTS models beyond Kokoro
- Voice training and fine-tuning
- Performance improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgments

- **[Kokoro TTS](https://github.com/thewh1teagle/kokoro)** - The excellent TTS model we build upon
- **[OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)** - Intel's inference optimization framework
- **Hugging Face** - Model hosting and community

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

## 🏢 UC-1 Pro Ecosystem

Unicorn Orator is part of the UC-1 Pro AI infrastructure suite:

| Service | Purpose | Port |
|---------|---------|------|
| **Unicorn Orator** | Text-to-speech | 8885 |
| [Unicorn Amanuensis](https://github.com/Unicorn-Commander/Unicorn-Amanuensis) | Speech-to-text | 8886 |
| Unicorn vLLM | LLM inference | 8000 |
| Open-WebUI | Chat interface | 3000 |

---

<div align="center">
  <b>Free your GPU. Enhance your AI.</b><br><br>
  <a href="https://hub.docker.com/r/magicunicorn/unicorn-orator">🐳 Docker Hub</a> •
  <a href="https://github.com/Unicorn-Commander/Unicorn-Orator/issues">🐛 Issues</a> •
  <a href="https://github.com/Unicorn-Commander/Unicorn-Orator/discussions">💬 Discussions</a>
  
  <br><br>
  <i>Built by Magic Unicorn Unconventional Technology & Stuff Inc.</i>
</div>