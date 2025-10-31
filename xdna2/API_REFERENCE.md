# Unicorn-Orator XDNA2 API Reference

**Version**: 2.0.0-xdna2
**Base URL**: `http://localhost:9001`
**Content-Type**: `application/json`
**Date**: October 31, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
   - [POST /v1/audio/speech](#post-v1audiospeech)
   - [GET /voices](#get-voices)
   - [GET /health](#get-health)
   - [GET /platform](#get-platform)
   - [GET /stats](#get-stats)
   - [POST /stats/reset](#post-statsreset)
   - [GET /](#get-)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)

---

## Overview

The Unicorn-Orator XDNA2 API provides REST endpoints for high-performance text-to-speech synthesis using AMD XDNA2 NPU acceleration. The API is OpenAI-compatible where applicable.

### Key Features

- **NPU Acceleration**: 40-60x realtime synthesis
- **BF16 Workaround**: Automatic error correction (789% → 3.55%)
- **Multiple Voices**: 47+ pre-trained voices
- **RESTful Design**: Standard HTTP methods and status codes
- **JSON Responses**: All responses in JSON format
- **WAV Output**: High-quality 24kHz audio

---

## Authentication

**Current Status**: No authentication required (development version)

**Production Recommendation**: Implement API key authentication

```bash
# Future: API key in header
curl -H "X-API-Key: your-api-key" http://localhost:9001/v1/audio/speech
```

---

## Endpoints

### POST /v1/audio/speech

Generate speech audio from text input using XDNA2 NPU.

#### Request

**URL**: `/v1/audio/speech`
**Method**: `POST`
**Content-Type**: `application/json`

**Body Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize (max 5000 characters) |
| `voice` | string | No | `"af"` | Voice ID (see [GET /voices](#get-voices)) |
| `speed` | float | No | `1.0` | Speech speed multiplier (0.5 - 2.0) |
| `use_npu` | boolean | No | `true` | Use NPU acceleration |
| `use_bf16_workaround` | boolean | No | `null` | Override global BF16 workaround setting |

**Request Schema**:

```json
{
  "text": "string",
  "voice": "string",
  "speed": 1.0,
  "use_npu": true,
  "use_bf16_workaround": null
}
```

#### Response

**Success (200 OK)**:
- **Content-Type**: `audio/wav`
- **Body**: Binary WAV audio data (24kHz, 16-bit PCM, mono)

**Headers**:
```
Content-Type: audio/wav
Content-Disposition: inline; filename=speech.wav
```

#### Example

**Request**:
```bash
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world! This is Unicorn Orator speaking.",
    "voice": "af_heart",
    "speed": 1.0,
    "use_npu": true
  }' \
  --output output.wav
```

**Response**:
```
Binary WAV audio data saved to output.wav
```

#### Advanced Usage

**With BF16 Workaround Override**:
```bash
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Testing BF16 workaround",
    "voice": "af_bella",
    "use_bf16_workaround": true
  }' \
  --output test.wav
```

**Disable NPU (CPU Fallback)**:
```bash
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "CPU fallback test",
    "use_npu": false
  }' \
  --output cpu_test.wav
```

**Slow Speech**:
```bash
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This will be spoken slowly",
    "speed": 0.75
  }' \
  --output slow.wav
```

**Fast Speech**:
```bash
curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This will be spoken quickly",
    "speed": 1.5
  }' \
  --output fast.wav
```

#### Error Responses

**400 Bad Request**: Invalid request parameters
```json
{
  "detail": "Text is required"
}
```

**500 Internal Server Error**: Synthesis failed
```json
{
  "detail": "NPU execution failed: kernel timeout"
}
```

---

### GET /voices

List all available voices.

#### Request

**URL**: `/voices`
**Method**: `GET`
**Content-Type**: N/A

#### Response

**Success (200 OK)**:

```json
{
  "voices": [
    "af",
    "af_bella",
    "af_sarah",
    "af_heart",
    "af_jessica",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis"
  ]
}
```

#### Example

**Request**:
```bash
curl http://localhost:9001/voices
```

**Response**:
```json
{
  "voices": [
    "af",
    "af_bella",
    "af_sarah",
    "af_heart",
    "af_jessica",
    "af_nicole",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis"
  ]
}
```

#### Voice Naming Convention

- **Prefix**:
  - `af_`: American Female
  - `am_`: American Male
  - `bf_`: British Female
  - `bm_`: British Male

- **Suffix**: Voice characteristics
  - `_heart`: Warm, expressive
  - `_bella`: Clear, professional
  - `_sarah`: Calm, natural
  - `_sky`: Energetic, bright

---

### GET /health

Health check endpoint for monitoring service status.

#### Request

**URL**: `/health`
**Method**: `GET`
**Content-Type**: N/A

#### Response

**Success (200 OK)**:

```json
{
  "status": "healthy",
  "model": "kokoro-v0_19",
  "backend": "XDNA2 NPU",
  "npu_enabled": true,
  "bf16_workaround": true,
  "bf16_stats": {
    "total_calls": 42,
    "max_input_range": 3.14159,
    "min_input_range": 0.01
  },
  "voices_loaded": true
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service status: `"healthy"` or `"degraded"` |
| `model` | string | TTS model version |
| `backend` | string | Compute backend: `"XDNA2 NPU"` or `"CPU"` |
| `npu_enabled` | boolean | Whether NPU acceleration is active |
| `bf16_workaround` | boolean | Whether BF16 workaround is enabled |
| `bf16_stats` | object | BF16 workaround statistics (see [GET /stats](#get-stats)) |
| `voices_loaded` | boolean | Whether voice embeddings loaded successfully |

#### Example

**Request**:
```bash
curl http://localhost:9001/health
```

**Response**:
```json
{
  "status": "healthy",
  "model": "kokoro-v0_19",
  "backend": "XDNA2 NPU",
  "npu_enabled": true,
  "bf16_workaround": true,
  "bf16_stats": {
    "total_calls": 123,
    "max_input_range": 4.532,
    "min_input_range": 0.001
  },
  "voices_loaded": true
}
```

#### Health Check Script

```bash
#!/bin/bash
# health-check.sh

RESPONSE=$(curl -s http://localhost:9001/health)
STATUS=$(echo $RESPONSE | jq -r '.status')

if [ "$STATUS" = "healthy" ]; then
  echo "✅ Service is healthy"
  exit 0
else
  echo "❌ Service is unhealthy"
  exit 1
fi
```

---

### GET /platform

Get detailed platform and configuration information.

#### Request

**URL**: `/platform`
**Method**: `GET`
**Content-Type**: N/A

#### Response

**Success (200 OK)**:

```json
{
  "service": "Unicorn-Orator",
  "version": "2.0.0-xdna2",
  "platform": "XDNA2",
  "npu_enabled": true,
  "bf16_workaround": {
    "enabled": true,
    "description": "Scales inputs to [0,1] range to avoid AMD XDNA2 BF16 signed value bug",
    "error_reduction": "789% → 3.55%"
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service name |
| `version` | string | Service version |
| `platform` | string | NPU platform: `"XDNA2"`, `"XDNA1"`, or `"CPU"` |
| `npu_enabled` | boolean | Whether NPU is available and enabled |
| `bf16_workaround` | object | BF16 workaround configuration |
| `bf16_workaround.enabled` | boolean | Whether workaround is active |
| `bf16_workaround.description` | string | Workaround description |
| `bf16_workaround.error_reduction` | string | Error rate improvement |

#### Example

**Request**:
```bash
curl http://localhost:9001/platform
```

**Response**:
```json
{
  "service": "Unicorn-Orator",
  "version": "2.0.0-xdna2",
  "platform": "XDNA2",
  "npu_enabled": true,
  "bf16_workaround": {
    "enabled": true,
    "description": "Scales inputs to [0,1] range to avoid AMD XDNA2 BF16 signed value bug",
    "error_reduction": "789% → 3.55%"
  }
}
```

#### Use Case: Platform Detection

```python
import requests

def detect_platform():
    response = requests.get("http://localhost:9001/platform")
    data = response.json()

    print(f"Service: {data['service']}")
    print(f"Platform: {data['platform']}")
    print(f"NPU Enabled: {data['npu_enabled']}")

    if data['bf16_workaround']['enabled']:
        print(f"BF16 Workaround: {data['bf16_workaround']['error_reduction']}")

detect_platform()
```

---

### GET /stats

Get BF16 workaround usage statistics.

#### Request

**URL**: `/stats`
**Method**: `GET`
**Content-Type**: N/A

#### Response

**Success (200 OK)**:

```json
{
  "bf16_workaround": {
    "total_calls": 1523,
    "max_input_range": 4.532,
    "min_input_range": 0.001
  },
  "npu_enabled": true
}
```

**Error (BF16 Workaround Disabled)**:
```json
{
  "error": "BF16 workaround not enabled"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `bf16_workaround.total_calls` | integer | Total number of BF16 operations executed |
| `bf16_workaround.max_input_range` | float | Maximum input range seen (max - min) |
| `bf16_workaround.min_input_range` | float | Minimum input range seen |
| `npu_enabled` | boolean | Whether NPU is enabled |

#### Example

**Request**:
```bash
curl http://localhost:9001/stats
```

**Response**:
```json
{
  "bf16_workaround": {
    "total_calls": 2048,
    "max_input_range": 5.678,
    "min_input_range": 0.0001
  },
  "npu_enabled": true
}
```

#### Monitoring Script

```bash
#!/bin/bash
# monitor-bf16.sh

while true; do
  STATS=$(curl -s http://localhost:9001/stats)
  TOTAL=$(echo $STATS | jq -r '.bf16_workaround.total_calls')
  MAX_RANGE=$(echo $STATS | jq -r '.bf16_workaround.max_input_range')

  echo "[$(date)] Total calls: $TOTAL, Max range: $MAX_RANGE"
  sleep 10
done
```

---

### POST /stats/reset

Reset BF16 workaround statistics counters.

#### Request

**URL**: `/stats/reset`
**Method**: `POST`
**Content-Type**: N/A

#### Response

**Success (200 OK)**:

```json
{
  "status": "statistics reset"
}
```

**Error (BF16 Workaround Disabled)**:
```json
{
  "error": "BF16 workaround not enabled"
}
```

#### Example

**Request**:
```bash
curl -X POST http://localhost:9001/stats/reset
```

**Response**:
```json
{
  "status": "statistics reset"
}
```

#### Verify Reset

```bash
# Reset stats
curl -X POST http://localhost:9001/stats/reset

# Check stats (should be zero)
curl http://localhost:9001/stats
# {
#   "bf16_workaround": {
#     "total_calls": 0,
#     "max_input_range": 0.0,
#     "min_input_range": inf
#   },
#   "npu_enabled": true
# }
```

---

### GET /

Root endpoint with API information.

#### Request

**URL**: `/`
**Method**: `GET`
**Content-Type**: N/A

#### Response

**Success (200 OK)**:

```json
{
  "service": "Unicorn Orator",
  "version": "2.0.0-xdna2",
  "description": "XDNA2 NPU-Accelerated Text-to-Speech with BF16 Workaround",
  "npu_enabled": true,
  "bf16_workaround": true,
  "endpoints": {
    "/v1/audio/speech": "POST - Generate speech from text",
    "/voices": "GET - List available voices",
    "/health": "GET - Health check",
    "/platform": "GET - Platform information",
    "/stats": "GET - BF16 workaround statistics",
    "/stats/reset": "POST - Reset statistics"
  },
  "performance": {
    "target_speedup": "40-60x realtime",
    "power_usage": "6-15W",
    "error_rate": "3.55% (with workaround) vs 789% (without)"
  }
}
```

#### Example

**Request**:
```bash
curl http://localhost:9001/
```

**Response**:
```json
{
  "service": "Unicorn Orator",
  "version": "2.0.0-xdna2",
  "description": "XDNA2 NPU-Accelerated Text-to-Speech with BF16 Workaround",
  "npu_enabled": true,
  "bf16_workaround": true,
  "endpoints": {
    "/v1/audio/speech": "POST - Generate speech from text",
    "/voices": "GET - List available voices",
    "/health": "GET - Health check",
    "/platform": "GET - Platform information",
    "/stats": "GET - BF16 workaround statistics",
    "/stats/reset": "POST - Reset statistics"
  },
  "performance": {
    "target_speedup": "40-60x realtime",
    "power_usage": "6-15W",
    "error_rate": "3.55% (with workaround) vs 789% (without)"
  }
}
```

---

## Data Models

### TTSRequest

Request model for speech synthesis.

```json
{
  "text": "string",
  "voice": "string",
  "speed": 1.0,
  "use_npu": true,
  "use_bf16_workaround": null
}
```

**Fields**:
- `text` (string, required): Text to synthesize (1-5000 characters)
- `voice` (string, optional): Voice ID (default: `"af"`)
- `speed` (float, optional): Speech speed (0.5-2.0, default: 1.0)
- `use_npu` (boolean, optional): Use NPU (default: `true`)
- `use_bf16_workaround` (boolean, optional): Override BF16 setting (default: `null` = use global)

### BF16Stats

BF16 workaround statistics model.

```json
{
  "total_calls": 0,
  "max_input_range": 0.0,
  "min_input_range": 0.0
}
```

**Fields**:
- `total_calls` (integer): Total BF16 operations
- `max_input_range` (float): Maximum input range seen
- `min_input_range` (float): Minimum input range seen

### HealthResponse

Health check response model.

```json
{
  "status": "healthy",
  "model": "string",
  "backend": "string",
  "npu_enabled": true,
  "bf16_workaround": true,
  "bf16_stats": {},
  "voices_loaded": true
}
```

### PlatformResponse

Platform information response model.

```json
{
  "service": "Unicorn-Orator",
  "version": "2.0.0-xdna2",
  "platform": "XDNA2",
  "npu_enabled": true,
  "bf16_workaround": {
    "enabled": true,
    "description": "string",
    "error_reduction": "string"
  }
}
```

---

## Error Handling

### HTTP Status Codes

| Status | Description | Common Causes |
|--------|-------------|---------------|
| 200 | OK | Request succeeded |
| 400 | Bad Request | Invalid parameters |
| 500 | Internal Server Error | NPU error, synthesis failed |

### Error Response Format

All errors return JSON:

```json
{
  "detail": "Error message here"
}
```

### Common Errors

#### 400 Bad Request

**Missing text**:
```json
{
  "detail": "field required: text"
}
```

**Invalid voice**:
```json
{
  "detail": "Voice 'invalid_voice' not found"
}
```

**Invalid speed**:
```json
{
  "detail": "Speed must be between 0.5 and 2.0"
}
```

#### 500 Internal Server Error

**NPU failure**:
```json
{
  "detail": "NPU execution failed: kernel timeout"
}
```

**Synthesis failure**:
```json
{
  "detail": "Failed to generate audio: out of memory"
}
```

### Error Handling Example

```python
import requests

def safe_tts(text, voice="af_heart"):
    try:
        response = requests.post(
            "http://localhost:9001/v1/audio/speech",
            json={"text": text, "voice": voice},
            timeout=10
        )

        if response.status_code == 200:
            with open("output.wav", "wb") as f:
                f.write(response.content)
            return True
        else:
            error = response.json()
            print(f"Error {response.status_code}: {error['detail']}")
            return False

    except requests.exceptions.Timeout:
        print("Request timed out")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Usage
safe_tts("Hello, world!")
```

---

## Rate Limiting

**Current Status**: No rate limiting (development version)

**Production Recommendation**: Implement rate limiting

### Recommended Limits

- **Per IP**: 100 requests/minute
- **Per API Key**: 1000 requests/minute
- **Concurrent Requests**: 10 per client

### nginx Rate Limiting Example

```nginx
limit_req_zone $binary_remote_addr zone=tts:10m rate=100r/m;

location /v1/audio/speech {
    limit_req zone=tts burst=20 nodelay;
    proxy_pass http://localhost:9001;
}
```

---

## Examples

### Python Example

```python
import requests

def text_to_speech(text, voice="af_heart", output_file="output.wav"):
    """Generate speech from text"""
    response = requests.post(
        "http://localhost:9001/v1/audio/speech",
        json={
            "text": text,
            "voice": voice,
            "speed": 1.0
        }
    )

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved to {output_file}")
    else:
        print(f"❌ Error: {response.json()}")

# Usage
text_to_speech("Hello, world!", voice="af_bella")
```

### JavaScript Example

```javascript
async function textToSpeech(text, voice = "af_heart") {
  const response = await fetch("http://localhost:9001/v1/audio/speech", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: text,
      voice: voice,
      speed: 1.0
    })
  });

  if (response.ok) {
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    // Play audio
    const audio = new Audio(url);
    audio.play();

    return url;
  } else {
    const error = await response.json();
    console.error("Error:", error.detail);
    throw new Error(error.detail);
  }
}

// Usage
textToSpeech("Hello, world!", "af_bella");
```

### Shell Script Example

```bash
#!/bin/bash
# tts.sh - Simple TTS script

TEXT="${1:-Hello world}"
VOICE="${2:-af_heart}"
OUTPUT="${3:-output.wav}"

curl -X POST http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TEXT\", \"voice\": \"$VOICE\"}" \
  --output "$OUTPUT"

if [ $? -eq 0 ]; then
  echo "✅ Saved to $OUTPUT"
  aplay "$OUTPUT"  # Play audio
else
  echo "❌ Failed to generate speech"
  exit 1
fi
```

Usage:
```bash
./tts.sh "Hello world" "af_bella" "hello.wav"
```

### Batch Processing Example

```python
import requests
import concurrent.futures

def generate_speech(item):
    text, voice, filename = item
    response = requests.post(
        "http://localhost:9001/v1/audio/speech",
        json={"text": text, "voice": voice}
    )

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return f"✅ {filename}"
    else:
        return f"❌ {filename}: {response.json()['detail']}"

# Batch data
batch = [
    ("Hello from voice 1", "af_heart", "voice1.wav"),
    ("Hello from voice 2", "af_bella", "voice2.wav"),
    ("Hello from voice 3", "am_adam", "voice3.wav"),
    ("Hello from voice 4", "bf_emma", "voice4.wav"),
]

# Process in parallel (max 4 concurrent)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(generate_speech, batch)

for result in results:
    print(result)
```

### Monitoring Example

```python
import requests
import time

def monitor_service():
    """Monitor service health and stats"""
    while True:
        try:
            # Check health
            health = requests.get("http://localhost:9001/health").json()
            print(f"[{time.strftime('%H:%M:%S')}] Health: {health['status']}")

            # Check stats
            stats = requests.get("http://localhost:9001/stats").json()
            bf16 = stats['bf16_workaround']
            print(f"  BF16 calls: {bf16['total_calls']}, "
                  f"Max range: {bf16['max_input_range']:.2f}")

        except Exception as e:
            print(f"❌ Error: {e}")

        time.sleep(60)  # Check every minute

# Usage
monitor_service()
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0-xdna2 | Oct 31, 2025 | Initial XDNA2 release with BF16 workaround |
| 1.0.0-xdna1 | Aug 2025 | XDNA1 production release |

---

## Support

- **Documentation**: `DEPLOYMENT_GUIDE.md`, `IMPLEMENTATION_NOTES.md`
- **Issues**: https://github.com/Unicorn-Commander/Unicorn-Orator/issues
- **Email**: aaron@magicunicorn.tech

---

**Built with Magic Unicorn Unconventional Technology & Stuff Inc**
**License**: MIT
**Last Updated**: October 31, 2025
