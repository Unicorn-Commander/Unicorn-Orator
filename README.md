# Unicorn-Orator

Multi-platform Text-to-Speech (TTS) service with automatic NPU acceleration support using Kokoro TTS.

## Overview

Unicorn-Orator automatically detects and uses the best available compute backend:

- **XDNA2**: Strix Point NPU (AMD Ryzen AI 300 series) - *Under development*
- **XDNA1**: Phoenix/Hawk Point NPU (AMD Ryzen 7040/8040 series) - *Current implementation*
- **CPU**: Software fallback for systems without NPU

## Features

- High-quality neural TTS using Kokoro
- Multiple voice options
- Professional-grade audio output
- Automatic platform detection
- NPU acceleration when available
- ONNX Runtime optimizations

## Architecture

```
Unicorn-Orator/
├── api.py                    # Main entry point with platform detection
├── runtime/
│   └── platform_detector.py  # Auto-detects NPU/CPU
├── xdna1/                    # Phoenix/Hawk Point implementation
│   ├── server.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── phoneme_mapping.json
│   ├── static/
│   └── docker-compose.yml
├── xdna2/                    # Strix Point implementation (WIP)
│   ├── kernels/              # Custom NPU kernels
│   ├── runtime/              # XDNA2 runtime integration
│   └── README_XDNA2.md
└── cpu/                      # CPU fallback (planned)
```

## Quick Start

### Using Docker

```bash
cd xdna1
docker-compose up
```

### Standalone

```bash
pip install -r xdna1/requirements.txt
python api.py
```

## API Endpoints

### POST /v1/audio/speech

Generate speech from text.

**Parameters:**
- `text`: Text to synthesize
- `voice`: Voice ID (optional, default: first available voice)
- `speed`: Speech speed multiplier (optional, default: 1.0)

**Example:**

```bash
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice": "af_heart"}' \
  --output output.wav
```

### GET /voices

List available voices.

```bash
curl http://localhost:9001/voices
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:9001/health
```

### GET /platform

Get current platform and backend information.

```bash
curl http://localhost:9001/platform
```

## Environment Variables

### Model Configuration
- `ONNX_MODEL`: Path to ONNX model file
- `VOICE_EMBEDDINGS`: Path to voice embeddings file
- `DEFAULT_VOICE`: Default voice to use

### Platform Override
- `NPU_PLATFORM`: Force specific platform (xdna1, xdna2, cpu)

## Platform Support Status

| Platform | Status | Notes |
|----------|--------|-------|
| XDNA1 (Phoenix/Hawk Point) | Production | Fully tested with ONNX |
| XDNA2 (Strix Point) | Development | Kernel development in progress |
| CPU | Fallback | Uses XDNA1 backend in CPU mode |

## Development

### Adding XDNA2 Support

XDNA2 implementation is tracked in `xdna2/README_XDNA2.md`. Key components:

1. **Custom Kernels**: INT8 quantized operations for NPU
2. **Runtime Integration**: XDNA2 device management
3. **Model Optimization**: NPU-specific model transformations

### Testing Platform Detection

```python
from runtime.platform_detector import get_platform_info

info = get_platform_info()
print(info)
```

## Integration with CC-1L

This repository is designed to be used as a Git submodule in the CC-1L project:

```bash
cd CC-1L
git submodule add https://github.com/Unicorn-Commander/Unicorn-Orator.git npu-services/unicorn-orator
```

## License

Proprietary - Unicorn Commander Project

## Related Projects

- **Unicorn-Amanuensis**: STT service (sister project)
- **unicorn-cpu-core**: Shared utilities library
- **CC-1L**: Main integration project
