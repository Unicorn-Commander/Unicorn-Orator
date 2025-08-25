# 🦄 Unicorn Orator

<div align="center">
  <img src="assets/unicorn-orator-logo.png" alt="Unicorn Orator Logo" width="200">
  
  **Professional AI Speech Processing Platform**
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
  
  *High-quality Speech-to-Text and Text-to-Speech services in one powerful platform*
</div>

---

## 🌟 Overview

Unicorn Orator is a comprehensive speech processing platform that combines state-of-the-art Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities into a single, easy-to-deploy solution. Built for both developers and end-users, it provides OpenAI-compatible APIs alongside beautiful web interfaces.

### ✨ Key Features

- **🎙️ Advanced Speech-to-Text** - Powered by WhisperX with word-level timestamps and speaker diarization
- **🔊 Natural Text-to-Speech** - High-quality voice synthesis using Kokoro models
- **🎯 OpenAI API Compatible** - Drop-in replacement for OpenAI's speech APIs
- **🎨 Beautiful Web Interface** - Professional UI for both STT and TTS
- **🚀 Hardware Optimized** - Support for CPU, GPU, and Intel iGPU acceleration
- **🐳 Docker Ready** - Simple deployment with Docker Compose
- **🔧 Highly Configurable** - Extensive customization options

## 📋 Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- (Optional) NVIDIA GPU for accelerated processing
- (Optional) Intel integrated GPU for TTS acceleration

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Unicorn-Commander/Unicorn-Orator.git
cd Unicorn-Orator
```

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env with your preferred settings
nano .env
```

### 3. Start the Services

```bash
docker-compose up -d
```

### 4. Access the Services

- **TTS Web Interface**: http://localhost:8880
- **STT API**: http://localhost:9000
- **TTS API**: http://localhost:8880

## 🎙️ Speech-to-Text (WhisperX)

### Features
- High-accuracy transcription using OpenAI Whisper models
- Word-level timestamps for precise alignment
- Speaker diarization (who said what)
- Batch processing for efficiency
- Support for 100+ languages

### API Usage

```bash
# Transcribe an audio file
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "response_format=json"

# With speaker diarization (requires HF_TOKEN)
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "diarize=true"
```

### Available Models
- `tiny` - Fastest, lowest accuracy (39M parameters)
- `base` - Good balance (74M parameters)
- `small` - Better accuracy (244M parameters)
- `medium` - High accuracy (769M parameters)
- `large-v3` - Best accuracy (1550M parameters)

## 🔊 Text-to-Speech (Kokoro)

### Features
- Natural-sounding voice synthesis
- Multiple voice options (male/female, different accents)
- Adjustable speed and pitch
- Low-latency generation
- OpenVINO optimization for Intel GPUs

### Web Interface

Access the beautiful Unicorn Orator interface at http://localhost:8880

Features include:
- Real-time voice preview
- Voice selection with descriptions
- Speed control
- Download generated audio
- Multiple theme options

### API Usage

```bash
# Generate speech from text
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to Unicorn Orator!",
    "voice": "af",
    "speed": 1.0
  }' \
  --output speech.wav
```

### Available Voices

#### Female Voices
- `af` - Friendly and warm (default)
- `af_bella` - Elegant and professional
- `af_nicole` - Energetic and clear
- `af_sarah` - Calm and soothing
- `af_sky` - Young and cheerful
- `bf_emma` - British accent, sophisticated
- `bf_isabella` - British accent, friendly

#### Male Voices
- `am_adam` - Deep and authoritative
- `am_michael` - Friendly and casual
- `bm_george` - British accent, professional
- `bm_lewis` - British accent, warm

## ⚙️ Configuration

### Environment Variables

#### WhisperX Configuration
- `WHISPER_MODEL` - Model size (tiny, base, small, medium, large)
- `WHISPERX_DEVICE` - Computing device (cpu, cuda)
- `WHISPERX_COMPUTE_TYPE` - Computation precision (int8, float16)
- `WHISPERX_BATCH_SIZE` - Batch processing size
- `HF_TOKEN` - Hugging Face token for diarization

#### Kokoro Configuration
- `KOKORO_DEVICE` - Computing device (CPU, IGPU, CUDA)
- `KOKORO_VOICE` - Default voice selection
- `EXTERNAL_HOST` - Domain/IP for remote access
- `EXTERNAL_PROTOCOL` - HTTP or HTTPS

### Hardware Acceleration

#### NVIDIA GPU
```bash
# In .env
WHISPERX_DEVICE=cuda
WHISPERX_COMPUTE_TYPE=float16
KOKORO_DEVICE=CUDA
```

#### Intel iGPU (for TTS)
```bash
# In .env
KOKORO_DEVICE=IGPU
```

## 🔌 Integration

### OpenAI SDK Compatible

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:9000/v1"  # For STT
    # base_url="http://localhost:8880/v1"  # For TTS
)

# Speech-to-Text
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

# Text-to-Speech
response = client.audio.speech.create(
    model="tts-1",
    voice="af",
    input="Hello world!"
)
response.stream_to_file("output.mp3")
```

### Open-WebUI Integration

Add to Open-WebUI settings:

```yaml
TTS_PROVIDER: openai
TTS_OPENAI_API_BASE_URL: http://localhost:8880/v1
TTS_OPENAI_API_KEY: dummy-key
TTS_MODEL: kokoro

STT_PROVIDER: openai
STT_OPENAI_API_BASE_URL: http://localhost:9000/v1
STT_OPENAI_API_KEY: dummy-key
STT_MODEL: whisper-1
```

## 📊 Performance

### WhisperX Performance (CPU)
| Model | Speed | Memory | Accuracy |
|-------|-------|---------|----------|
| tiny  | ~10x realtime | 1GB | Good |
| base  | ~7x realtime | 1GB | Better |
| small | ~4x realtime | 2GB | Great |
| medium | ~2x realtime | 5GB | Excellent |
| large | ~1x realtime | 10GB | Best |

### Kokoro TTS Performance
- CPU: ~0.3x realtime generation
- Intel iGPU: ~2x realtime generation
- NVIDIA GPU: ~5x realtime generation

## 🛠️ Development

### Project Structure
```
Unicorn-Orator/
├── whisperx/           # Speech-to-Text service
│   ├── Dockerfile
│   ├── server.py
│   └── requirements.txt
├── kokoro-tts/         # Text-to-Speech service
│   ├── Dockerfile
│   ├── server.py
│   ├── static/         # Web interface
│   └── requirements.txt
├── docker-compose.yml  # Service orchestration
├── .env.template       # Configuration template
└── README.md          # This file
```

### Building from Source

```bash
# Build services
docker-compose build

# Run in development mode
docker-compose up
```

### Testing

```bash
# Test STT
./test-stt.sh

# Test TTS
./test-tts.sh

# Run all tests
./test-all.sh
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional voice models
- Language support improvements
- Performance optimizations
- UI enhancements
- Documentation improvements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the base STT model
- [WhisperX](https://github.com/m-bain/whisperX) for enhanced STT capabilities
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for TTS models
- The Unicorn Commander community for support and feedback

## 🔗 Links

- **GitHub**: https://github.com/Unicorn-Commander/Unicorn-Orator
- **Issues**: https://github.com/Unicorn-Commander/Unicorn-Orator/issues
- **Discussions**: https://github.com/Unicorn-Commander/Unicorn-Orator/discussions
- **UC-1 Pro**: https://github.com/Unicorn-Commander/UC-1-Pro

---

<div align="center">
  Made with ❤️ by Unicorn Commander
  
  🦄 *Speak Naturally with AI* 🦄
</div>