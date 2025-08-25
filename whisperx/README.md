# WhisperX Speech-to-Text Service

Advanced speech recognition with word-level timestamps and speaker diarization.

## Features
- High-accuracy transcription using Whisper
- Word-level timestamps
- Speaker diarization (requires HF_TOKEN)
- Batch processing for efficiency

## API Endpoints
- `POST /v1/audio/transcriptions` - Transcribe audio file
- `GET /health` - Service health check

## Environment Variables
- `WHISPER_MODEL` - Model size (tiny, base, small, medium, large)
- `DEVICE` - Computing device (cpu, cuda)
- `COMPUTE_TYPE` - Computation type (int8, float16)
- `BATCH_SIZE` - Batch size for processing
- `HF_TOKEN` - Hugging Face token for diarization models

## Testing Standalone
```bash
cd services/whisperx
docker-compose up

# Test with curl
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "diarize=false"
```
