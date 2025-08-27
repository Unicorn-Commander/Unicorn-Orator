# Unicorn Orator - Project Context for Claude

## Project Overview

Unicorn Orator is a professional AI voice synthesis platform optimized for Intel iGPU acceleration via OpenVINO. It provides high-quality text-to-speech with 50+ voices and specialized tools for dialogue, podcasts, stories, and commercials.

**Company**: Magic Unicorn Unconventional Technology & Stuff Inc  
**Repository**: https://github.com/Unicorn-Commander/Unicorn-Orator  
**Docker Hub**: https://hub.docker.com/r/magicunicorn/unicorn-orator  
**License**: MIT

## Key Features

- **Multi-Hardware Support**: Intel iGPU (OpenVINO), AMD NPU (XDNA), CPU fallback
- **50+ Professional Voices**: Multiple accents, genders, and styles
- **Voice Acting Tools**: 
  - Dialogue Generator (multi-character conversations)
  - Podcast Creator (professional podcast production)
  - Story Narration (kids stories & audiobooks)
  - Commercial Voiceover (ads & promos)
- **OpenAI API Compatible**: Drop-in replacement for OpenAI TTS
- **Open-WebUI Tool Servers**: Compatible with Open-WebUI for extended features

## Technical Stack

### Core Technologies
- **TTS Model**: Kokoro v0.19 (ONNX format)
- **Hardware Acceleration**: OpenVINO for Intel iGPU
- **Framework**: FastAPI with Python 3.10
- **Voice Embeddings**: 50+ pre-trained voices (1.0 MB)
- **Model Size**: ~100MB ONNX model

### Working Docker Image
```bash
# Intel iGPU optimized image (1.2GB)
docker pull magicunicorn/unicorn-orator:intel-igpu-v1.0
```

## Service Architecture

### Main TTS Service (Port 8885/8880)
- OpenAI-compatible API at `/v1/audio/speech`
- Web interface at `/web`
- Admin panel at `/admin`
- Health check at `/health`

### Voice Acting Tool Servers
| Tool | Port | Purpose |
|------|------|---------|
| Dialogue Generator | 13060 | Multi-character conversations |
| Story Narration | 13061 | Kids stories & audiobooks |
| Podcast Creator | 13062 | Professional podcasts |
| Commercial Voiceover | 13063 | Ads & commercials |

## Hardware Detection & Optimization

### Intel iGPU (Primary)
```python
# Automatic detection and OpenVINO configuration
if device == "IGPU":
    providers.append(('OpenVINOExecutionProvider', {
        'device_type': 'GPU',
        'precision': 'FP16',
        'cache_dir': './openvino_cache'
    }))
```

### AMD NPU Support (Planned)
- XDNA1 architecture support
- Custom MLIR kernels for optimization
- User has working implementation to integrate

### CPU Fallback
- ONNX Runtime CPU provider
- Works on any x86_64 system

## Quick Start

### Using Docker Compose
```bash
cd /home/ucadmin/Unicorn-Orator
docker-compose up -d
```

### Standalone Container
```bash
docker run -d --name unicorn-orator \
  -p 8885:8880 \
  -v $(pwd)/kokoro-tts/static:/app/static:ro \
  -v $(pwd)/kokoro-tts/server.py:/app/server.py:ro \
  --security-opt seccomp:unconfined \
  magicunicorn/unicorn-orator:intel-igpu-v1.0
```

### Tool Servers
```bash
cd tool-servers
docker-compose up -d
```

## API Usage

### Basic TTS (OpenAI Compatible)
```python
import requests

response = requests.post('http://localhost:8885/v1/audio/speech', 
    json={
        'text': 'Hello from Unicorn Orator!',
        'voice': 'af_bella',
        'speed': 1.0
    }
)

with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### Available Voices
- American: af_bella, am_michael, af_sarah, am_adam
- British: bf_emma, bm_george  
- Special: af_sky, am_echo, af_heart
- 50+ total voices with various styles

## Voice Acting Tools

### Dialogue Generator
Paste dialogue scripts with character names:
```
Character1: Hello there!
Character2: Hi, how are you?
```
Automatically assigns appropriate voices to each character.

### Story Narration
Supports both kids and adult stories:
```
Title: The Magic Forest
Type: kids

Once upon a time...

Rabbit: "Let's explore!"
```
Uses slower pacing and warm voices for children's stories.

### Podcast Creator
Professional podcast production:
```
Host: Welcome to our show!
Guest: Thanks for having me!
```
Includes intro/outro segments and multiple speaker support.

### Commercial Voiceover
Professional ads and promos:
```
[HOOK] Introducing the amazing product!
[BENEFIT] Save time and money!
[CTA] Call now!
```
Supports different styles: hard-sell, soft-sell, conversational.

## Important Notes

### Known Issues
- `onnxruntime-openvino` has executable stack errors in recent builds
- Use the pre-built Docker image from 2 weeks ago (working version)
- Building from scratch today will fail due to package issues

### Security Considerations
- Container requires `--security-opt seccomp:unconfined` for OpenVINO
- This is needed for GPU memory access

### Not Included
- **Transcription**: Use Unicorn Amanuensis for STT
- **Content Generation**: Orator only performs voice acting, not script creation
- **Real-time Streaming**: Batch processing only

## File Structure
```
Unicorn-Orator/
├── kokoro-tts/
│   ├── server.py          # Main TTS server
│   ├── models/            # ONNX model and voices
│   └── static/
│       └── index.html     # Web interface
├── tool-servers/
│   ├── dialogue_tool.py   # Multi-character dialogue
│   ├── story_narration_tool.py  # Story narration
│   ├── podcast_tool.py    # Podcast creation
│   ├── commercial_tool.py # Commercial voiceover
│   └── docker-compose.yml # Tool server orchestration
├── build.sh              # Smart build script with hardware detection
└── docker-compose.yml    # Main service orchestration
```

## Testing

### Check Service Health
```bash
curl http://localhost:8885/health
```

### Test Voice Synthesis
```bash
curl -X POST http://localhost:8885/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella"}' \
  --output test.wav
```

### Check Tool Servers
```bash
# Dialogue tool
curl http://localhost:13060/health

# Story narration
curl http://localhost:13061/health

# Podcast creator  
curl http://localhost:13062/health

# Commercial voiceover
curl http://localhost:13063/health
```

## GUI Features

### Main Interface (http://localhost:8885/web)
- Voice selection with 50+ options
- Speed control (0.5x to 2.0x)
- Advanced settings (sample rate, emotion, pitch)
- Shows "Intel iGPU (OpenVINO)" hardware status
- Voice Synthesis Tools section with 4 specialized tools

### Admin Panel (http://localhost:8885/admin)
- Start/stop individual tool servers
- View container logs
- Monitor service health
- Quick access to all tools

## Recent Updates (2024-12)

- Fixed GUI to properly display Intel iGPU (OpenVINO) status
- Added 4 specialized voice acting tool servers
- Removed transcription features (voice synthesis only)
- Published working Docker image to Docker Hub
- Created Open-WebUI compatible tool servers
- Enhanced admin panel with container management

## Integration with UC-1 Pro

Unicorn Orator is designed to work standalone or as part of the UC-1 Pro AI infrastructure. In UC-1 Pro, it provides TTS capabilities while other services handle:
- **Unicorn Amanuensis**: Speech-to-text (STT)
- **vLLM**: Large language model inference
- **Open-WebUI**: Chat interface integration