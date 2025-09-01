from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import io
import json
import struct
import soundfile as sf
from typing import Optional, Dict
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unicorn Orator - Professional AI Voice Synthesis")

# Load phoneme mapping
phoneme_to_id = {}
try:
    with open("phoneme_mapping.json", "r") as f:
        config = json.load(f)
        phoneme_to_id = config.get("vocab", {})
        logger.info(f"Loaded {len(phoneme_to_id)} phoneme mappings")
except Exception as e:
    logger.error(f"Failed to load phoneme mapping: {e}")

# Load voice embeddings
voices = {}
voice_embeddings = {}

# Try to load voice embeddings from zip file
try:
    import zipfile
    with zipfile.ZipFile("models/voices-v1.0.bin", "r") as zf:
        # List all .npy files in the zip
        voice_files = [f for f in zf.namelist() if f.endswith('.npy')]
        logger.info(f"Found {len(voice_files)} voice files in archive")
        
        for voice_file in voice_files:
            # Extract voice name from filename (e.g., "af_heart.npy" -> "af_heart")
            voice_name = os.path.splitext(os.path.basename(voice_file))[0]
            
            # Load the numpy array from the zip
            with zf.open(voice_file) as f:
                embedding = np.load(f)
                # Ensure it's float32 and has correct shape
                embedding = embedding.astype(np.float32)
                
                # Handle different embedding formats
                if embedding.shape == (256,):
                    # Already in correct format
                    pass
                elif len(embedding.shape) == 3 and embedding.shape[2] == 256:
                    # Format is (frames, 1, 256) - take the mean across frames
                    embedding = np.mean(embedding, axis=(0, 1))
                elif len(embedding.shape) == 2 and embedding.shape[1] == 256:
                    # Format is (frames, 256) - take the mean across frames
                    embedding = np.mean(embedding, axis=0)
                else:
                    logger.warning(f"Voice {voice_name} has unexpected shape {embedding.shape}")
                    continue
                    
            voices[voice_name] = voice_name.replace('_', ' ').title()
            voice_embeddings[voice_name] = embedding
            
        logger.info(f"Loaded {len(voices)} voices: {list(voices.keys())[:10]}...")
except Exception as e:
    logger.error(f"Failed to load voice embeddings: {e}")
    # Fallback - create dummy embeddings
    default_voices = [
        "af", "af_bella", "af_sarah", "af_heart", "af_jessica",
        "am_adam", "am_michael", "bf_emma", "bf_isabella", 
        "bm_george", "bm_lewis"
    ]
    for voice in default_voices:
        voices[voice] = voice.replace('_', ' ').title()
        # Create random embedding for testing
        voice_embeddings[voice] = np.random.randn(256).astype(np.float32) * 0.1

# Initialize ONNX Runtime with device selection
logger.info("Loading Kokoro model...")
session = None
device = os.environ.get("DEVICE", "CPU").upper()

try:
    providers = []
    
    if device == "GPU" or device == "IGPU":
        try:
            # Try OpenVINO execution provider for Intel GPU
            providers.append(('OpenVINOExecutionProvider', {
                'device_type': 'GPU',
                'precision': 'FP16',
                'cache_dir': './openvino_cache'
            }))
            logger.info("Attempting to use Intel GPU with OpenVINO")
        except Exception as e:
            logger.warning(f"OpenVINO GPU not available: {e}")
    
    # Always add CPU as fallback
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession("models/kokoro-v0_19.onnx", providers=providers)
    
    # Check which provider is actually being used
    actual_provider = session.get_providers()[0]
    logger.info(f"Kokoro model loaded successfully using {actual_provider}")
    
except Exception as e:
    logger.error(f"Failed to load Kokoro model: {e}")
    logger.warning("TTS service will run in mock mode")

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "af"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False

def text_to_phonemes(text: str, lang: str = 'en-us') -> str:
    """Convert text to phonemes using espeak-ng"""
    try:
        # Use espeak-ng to convert text to IPA phonemes
        cmd = ['espeak-ng', '-q', '-x', '--ipa=3', f'-v{lang}', text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            phonemes = result.stdout.strip()
            # Clean up the phonemes - remove extra spaces and normalize
            phonemes = ' '.join(phonemes.split())
            # Remove zero-width joiner and other problematic unicode characters
            phonemes = phonemes.replace('\u200d', '')  # Remove zero-width joiner
            phonemes = phonemes.replace('\u200c', '')  # Remove zero-width non-joiner
            return phonemes
        else:
            logger.error(f"espeak-ng failed: {result.stderr}")
            return text
    except Exception as e:
        logger.error(f"Failed to convert to phonemes: {e}")
        return text

def text_to_tokens(text: str, voice: str = "af") -> np.ndarray:
    """Convert text to token IDs using phoneme conversion"""
    # Determine language from voice name
    lang = 'en-us' if voice.startswith('a') else 'en-gb' if voice.startswith('b') else 'en-us'
    
    # Convert text to phonemes
    phonemes = text_to_phonemes(text, lang)
    logger.info(f"Phonemes: {phonemes}")
    
    # Convert phonemes to token IDs
    tokens = []
    
    # Add padding token at start
    tokens.append(0)
    
    i = 0
    while i < len(phonemes):
        # Try to match multi-character phonemes first
        matched = False
        for length in range(3, 0, -1):  # Try 3, 2, 1 character matches
            if i + length <= len(phonemes):
                substring = phonemes[i:i+length]
                if substring in phoneme_to_id:
                    tokens.append(phoneme_to_id[substring])
                    i += length
                    matched = True
                    break
        
        if not matched:
            # If no match, skip this character
            logger.warning(f"Unknown phoneme: '{phonemes[i]}'")
            i += 1
    
    # Add padding token at end
    tokens.append(0)
    
    logger.info(f"Tokens: {tokens[:20]}..." if len(tokens) > 20 else f"Tokens: {tokens}")
    
    return np.array([tokens], dtype=np.int64)

def synthesize_speech(text: str, voice: str = "af", speed: float = 1.0):
    """Synthesize speech using Kokoro model"""
    if session is None:
        # Return silence if model not loaded
        logger.warning("Model not loaded, returning silence")
        return np.zeros(24000, dtype=np.float32)
    
    try:
        # Get voice embedding
        if voice not in voice_embeddings:
            logger.warning(f"Voice {voice} not found, using default")
            logger.warning(f"Available voices: {list(voice_embeddings.keys())}")
            voice = "af"
        
        style_embedding = voice_embeddings[voice].reshape(1, 256)
        logger.info(f"Using voice {voice} with embedding shape {style_embedding.shape}")
        
        # Convert text to tokens
        tokens = text_to_tokens(text, voice)
        logger.info(f"Token shape: {tokens.shape}, first 20 tokens: {tokens[0][:20] if len(tokens[0]) > 0 else 'empty'}")
        
        # Prepare inputs
        inputs = {
            "tokens": tokens,
            "style": style_embedding,
            "speed": np.array([speed], dtype=np.float32)
        }
        
        # Run inference
        outputs = session.run(None, inputs)
        audio = outputs[0]
        logger.info(f"Generated audio shape: {audio.shape}, dtype: {audio.dtype}")
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Normalize audio
        audio = np.clip(audio, -1, 1)
        logger.info(f"Final audio shape: {audio.shape}, min: {audio.min():.3f}, max: {audio.max():.3f}")
        
        return audio
        
    except Exception as e:
        logger.error(f"Error in synthesis: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

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
    return {"voices": list(voices.keys())}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "kokoro-v0_19",
        "backend": "OpenVINO iGPU (FP16)",
        "model_loaded": session is not None,
        "voices_loaded": len(voices) > 0
    }

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "service": "Unicorn Orator",
        "version": "1.0",
        "description": "Professional AI Voice Synthesis Platform",
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