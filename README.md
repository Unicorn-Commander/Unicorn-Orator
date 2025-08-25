# 🦄 Unicorn Orator

<div align="center">
  <img src="assets/unicorn-orator-logo.png" alt="Unicorn Orator Logo" width="200">
  
  **Professional Text-to-Speech Service**
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
  
  *High-Quality Voice Synthesis with Hardware Acceleration*
</div>

---

## 🌟 Overview

Unicorn Orator is a professional text-to-speech service powered by Kokoro, offering natural voice synthesis with multiple voice options and hardware acceleration support. Built for both API integration and standalone use, it provides an OpenAI-compatible endpoint for seamless integration with existing applications.

**Looking for Speech-to-Text?** Check out [Unicorn Amanuensis](https://github.com/Unicorn-Commander/Unicorn-Amanuensis) - our dedicated transcription service powered by WhisperX.

### ✨ Key Features

- **🎭 Multiple Voices** - 20+ natural male and female voices
- **🌐 Multi-Language** - Support for English, Spanish, French, German, Chinese, Japanese, and more
- **🔊 High Quality** - Natural-sounding speech synthesis with emotion and emphasis
- **⚡ Fast Generation** - Real-time voice synthesis
- **🚀 Hardware Acceleration** - Support for CPU, NVIDIA GPU, AMD NPU, and Intel iGPU
- **🔌 OpenAI Compatible** - Drop-in replacement for OpenAI TTS API
- **🎨 Web Interface** - Interactive demo UI included
- **🐳 Docker Ready** - Easy deployment with Docker Compose
- **📊 SSML Support** - Fine control over speech synthesis

## 📋 Prerequisites

- Docker and Docker Compose
- 4GB+ RAM (8GB+ recommended)
- (Optional) NVIDIA GPU for CUDA acceleration
- (Optional) AMD Ryzen AI processor for NPU acceleration
- (Optional) Intel iGPU for OpenVINO acceleration

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Unicorn-Commander/Unicorn-Orator.git
cd Unicorn-Orator
```

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env with your settings
nano .env
```

### 3. Install with Hardware Detection

```bash
./install.sh
# Automatically detects and configures for your hardware
```

### 4. Access the Service

- **TTS API**: http://localhost:8880
- **Web Interface**: http://localhost:8880/demo
- **API Documentation**: http://localhost:8880/docs

## 🎙️ API Usage

### OpenAI-Compatible Endpoint

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:8880/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="af",
    input="Hello, welcome to Unicorn Orator!"
)

# Save the audio
with open("speech.mp3", "wb") as f:
    f.write(response.content)
```

### Direct API Call

```bash
# Generate speech
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world!",
    "voice": "af",
    "speed": 1.0,
    "format": "wav"
  }' \
  --output speech.wav
```

### Streaming Response

```python
import requests

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "text": "Long text to stream...",
        "voice": "am_michael",
        "stream": True
    },
    stream=True
)

with open("stream.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        f.write(chunk)
```

## 📊 Supported Voices

### Female Voices

| Voice | Description | Language | Accent |
|-------|-------------|----------|--------|
| `af` | Friendly and warm | English | American |
| `af_bella` | Elegant and professional | English | American |
| `af_nicole` | Energetic and clear | English | American |
| `af_sarah` | Calm and soothing | English | American |
| `af_sky` | Young and cheerful | English | American |
| `bf_emma` | Sophisticated | English | British |
| `bf_isabella` | Friendly | English | British |

### Male Voices

| Voice | Description | Language | Accent |
|-------|-------------|----------|--------|
| `am_adam` | Deep and authoritative | English | American |
| `am_michael` | Friendly and casual | English | American |
| `bm_george` | Professional | English | British |
| `bm_lewis` | Warm | English | British |

### Multi-Language Voices

| Voice | Languages Supported |
|-------|-------------------|
| `es_maria` | Spanish (Spain, Mexico) |
| `fr_marie` | French (France) |
| `de_anna` | German |
| `jp_yuki` | Japanese |
| `cn_xiaoli` | Chinese (Mandarin) |

## 🔧 Hardware Support

### Automatic Detection
The installer automatically detects and configures for available hardware:

| Hardware | Performance | Use Case |
|----------|-------------|----------|
| **AMD NPU** | ~8x realtime | Ryzen AI laptops (7040/8040 series) |
| **Intel iGPU** | ~10x realtime | Intel Arc/Iris Xe graphics |
| **NVIDIA GPU** | ~15x realtime | Dedicated GPU systems |
| **CPU** | ~3x realtime | Universal fallback |

### Manual Configuration

```bash
# Force specific backend
./install.sh --backend=npu

# Optimize for speed
./install.sh --variant=fast
```

## 🌐 Web Interface

Access the beautiful Unicorn Orator interface at http://localhost:8880/demo

Features:
- Real-time voice preview
- Voice selection with audio samples
- Speed and pitch controls
- Text formatting options
- Audio download in multiple formats
- Theme selection (Magic Unicorn, Dark, Light)

## 🔌 Integration Examples

### Open-WebUI Integration

Add to your Open-WebUI `.env`:
```env
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_KEY=dummy-key
AUDIO_TTS_OPENAI_API_BASE_URL=http://localhost:8880/v1
AUDIO_TTS_MODEL=tts-1
AUDIO_TTS_VOICE=af
```

### Home Assistant Integration

```yaml
tts:
  - platform: openai_tts
    api_key: dummy-key
    base_url: http://localhost:8880/v1
    model: tts-1
    voice: af
```

### Node.js Example

```javascript
const OpenAI = require('openai');

const openai = new OpenAI({
  apiKey: 'dummy-key',
  baseURL: 'http://localhost:8880/v1',
});

async function speak(text) {
  const mp3 = await openai.audio.speech.create({
    model: "tts-1",
    voice: "af",
    input: text,
  });
  
  const buffer = Buffer.from(await mp3.arrayBuffer());
  await fs.promises.writeFile("speech.mp3", buffer);
}
```

## 🛠️ Advanced Configuration

### Environment Variables

```env
# Kokoro Configuration
KOKORO_DEVICE=auto          # auto, cpu, cuda, npu, igpu
KOKORO_VOICE=af             # Default voice
KOKORO_MODELS_PATH=/models  # Model cache location

# Performance
NUM_THREADS=4                # CPU threads
BATCH_SIZE=1                 # Batch processing
MAX_TEXT_LENGTH=5000         # Maximum input text

# API Settings
API_KEY=your-api-key        # Optional API authentication
CORS_ORIGINS=*              # CORS configuration
```

### Docker Compose Override

```yaml
# docker-compose.override.yml
services:
  kokoro-tts:
    environment:
      - KOKORO_DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📈 Performance Optimization

### For Speed
- Use smaller voices (base models)
- Enable hardware acceleration
- Increase batch size
- Use streaming for long texts

### For Quality
- Use larger voices (enhanced models)
- Enable SSML processing
- Adjust phoneme weights
- Fine-tune prosody parameters

## 🤝 API Compatibility

Unicorn Orator is compatible with:
- OpenAI TTS API
- Azure Cognitive Services Speech (adapter available)
- Google Cloud Text-to-Speech (adapter available)
- Amazon Polly (adapter available)

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kokoro](https://github.com/thewh1teagle/kokoro) - Core TTS engine
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel optimization
- The Unicorn Commander community

## 🔗 Related Projects

- [Unicorn Amanuensis](https://github.com/Unicorn-Commander/Unicorn-Amanuensis) - Speech-to-Text companion
- [UC-1 Pro](https://github.com/Unicorn-Commander/UC-1-Pro) - Complete AI infrastructure stack

---

<div align="center">
  Made with ❤️ by Unicorn Commander
  
  🦄 *Speak with Intelligence* 🦄
</div>