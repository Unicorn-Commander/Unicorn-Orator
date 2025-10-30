from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import io
import json
import soundfile as sf
from typing import Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kokoro TTS Service")

# Load voices configuration
try:
    with open("models/voices.json", "r") as f:
        voices_data = json.load(f)
        if isinstance(voices_data, dict) and "voices" in voices_data:
            voices = voices_data["voices"]
        else:
            voices = voices_data
except FileNotFoundError:
    # Fallback voices if file doesn't exist
    voices = ["af", "bf"]

# Initialize ONNX Runtime
logger.info("Loading Kokoro model...")
session = None
try:
    session = ort.InferenceSession("models/kokoro-v0_19.onnx")
    logger.info("Kokoro model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Kokoro model: {e}")
    logger.warning("TTS service will run in mock mode")

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "af"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False

def text_to_phonemes(text: str):
    """Simple text to phoneme conversion - in production use a proper phonemizer"""
    # This is a placeholder - real implementation would use a phonemizer
    return text.lower()

def synthesize_speech(text: str, voice: str = "af", speed: float = 1.0):
    """Synthesize speech using Kokoro model"""
    if session is None:
        # Return silence if model not loaded
        logger.warning("Model not loaded, returning silence")
        # Return 1 second of silence at 24kHz
        return np.zeros(24000, dtype=np.float32)
    
    # Prepare input
    phonemes = text_to_phonemes(text)
    
    # Create input arrays (simplified - real implementation needs proper preprocessing)
    phoneme_ids = np.array([[ord(c) for c in phonemes]], dtype=np.int64)
    
    # Run inference
    inputs = {
        session.get_inputs()[0].name: phoneme_ids
    }
    
    outputs = session.run(None, inputs)
    audio = outputs[0]
    
    # Apply speed adjustment
    if speed != 1.0:
        # Simple speed adjustment - production would use proper resampling
        indices = np.arange(0, len(audio[0]), speed)
        audio = np.interp(indices, np.arange(len(audio[0])), audio[0])
    else:
        audio = audio[0]
    
    # Normalize audio
    audio = np.clip(audio, -1, 1)
    
    return audio


@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest):
    try:
        logger.info(f"Synthesizing speech for text: {request.text[:50]}...")
        
        # Synthesize speech
        audio_data = synthesize_speech(
            request.text,
            request.voice,
            request.speed
        )
        
        # Convert to WAV format
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_data, 24000, format='WAV')
        audio_bytes.seek(0)
        
        return StreamingResponse(
            audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"Error in TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def list_voices():
    return {"voices": voices if isinstance(voices, list) else list(voices.keys())}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "kokoro-v0_19",
        "backend": "ONNX Runtime"
    }

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "service": "Kokoro TTS",
        "version": "1.0",
        "endpoints": {
            "/v1/audio/speech": "POST - Generate speech from text",
            "/voices": "GET - List available voices",
            "/health": "GET - Health check",
            "/web": "GET - Web interface"
        }
    }

@app.get("/web")
async def web_interface():
    """Serve the web interface"""
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Web interface not found")

# Mount static files for web interface assets
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
