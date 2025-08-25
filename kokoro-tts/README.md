# Kokoro Text-to-Speech Service

Natural text-to-speech synthesis using Kokoro models.

## Features
- Multiple voice options
- Speed control
- Low latency synthesis
- ONNX Runtime backend

## API Endpoints
- `POST /v1/audio/speech` - Generate speech from text
- `GET /voices` - List available voices
- `GET /health` - Service health check

## Environment Variables
- `DEVICE` - Computing device (CPU, GPU)
- `DEFAULT_VOICE` - Default voice to use

## Testing Standalone
```bash
cd services/kokoro-tts
docker-compose up

# Test with curl
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af"}'
```
