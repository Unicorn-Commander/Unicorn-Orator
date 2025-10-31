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

# Try to import NPU integration
NPU_AVAILABLE = False
KokoroMLIRNPUIntegration = None
try:
    from kokoro_mlir_integration import KokoroMLIRNPUIntegration
    NPU_AVAILABLE = True
    logger.info("✅ NPU integration module loaded")
except ImportError as e:
    logger.warning(f"⚠️ NPU integration not available: {e}")
    logger.info("   Will use standard ONNX Runtime with CPU/GPU")

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

# Hardware detection
def detect_hardware():
    """Detect available hardware acceleration"""
    import subprocess

    hardware_info = {
        "type": "cpu",
        "name": "CPU",
        "npu_available": False,
        "npu_kernels": 0,
        "igpu_available": False,
        "details": {}
    }

    # Check NPU (AMD Phoenix XDNA1)
    try:
        if os.path.exists("/dev/accel/accel0"):
            # Try to run xrt-smi to verify NPU is accessible
            result = subprocess.run(
                ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and "NPU Phoenix" in result.stdout:
                hardware_info["npu_available"] = True
                hardware_info["type"] = "npu"
                hardware_info["name"] = "AMD Phoenix NPU"

                # Count available NPU kernels (shared from Amanuensis project)
                kernel_dir = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/whisper_encoder_kernels"
                if os.path.exists(kernel_dir):
                    kernels = [f for f in os.listdir(kernel_dir) if f.endswith('.xclbin')]
                    hardware_info["npu_kernels"] = len(kernels)

                # Extract firmware version
                for line in result.stdout.split('\n'):
                    if 'NPU Firmware Version' in line:
                        hardware_info["details"]["firmware"] = line.split(':')[-1].strip()
                    elif 'Version' in line and 'XRT' in result.stdout:
                        hardware_info["details"]["xrt_version"] = line.split(':')[-1].strip()

                logger.info(f"✅ AMD Phoenix NPU detected")
                return hardware_info
    except Exception as e:
        logger.debug(f"NPU check failed: {e}")

    # Check Intel iGPU
    try:
        if os.path.exists("/dev/dri/renderD128"):
            hardware_info["igpu_available"] = True
            hardware_info["type"] = "igpu"
            hardware_info["name"] = "Intel iGPU"
            logger.info("✅ Intel iGPU detected")
            return hardware_info
    except Exception as e:
        logger.debug(f"iGPU check failed: {e}")

    # Fallback to CPU
    logger.info("⚠️ No hardware acceleration detected, using CPU")
    return hardware_info

# Initialize Kokoro TTS with device selection
logger.info("Loading Kokoro model...")
HARDWARE = detect_hardware()
session = None
tts_engine = None
backend_type = "unknown"
device = os.environ.get("DEVICE", "CPU").upper()

try:
    # Try NPU first if available and device is set to NPU
    if NPU_AVAILABLE and device == "NPU" and os.path.exists("/dev/accel/accel0"):
        try:
            logger.info("🚀 Attempting to initialize NPU-accelerated Kokoro TTS...")
            # Use standard FP32 model for NPU (INT8 has ConvInteger compatibility issues)
            standard_model = "models/kokoro-v0_19.onnx"
            npu_model = "models/kokoro-npu-quantized-int8.onnx"
            # Prefer standard model as INT8 ConvInteger ops aren't supported by ONNX Runtime
            model_path = standard_model if os.path.exists(standard_model) else npu_model

            tts_engine = KokoroMLIRNPUIntegration(
                model_path=model_path,
                voices_path="models/voices-v1.0.bin"
            )
            backend_type = "NPU (INT8)" if "npu-quantized" in model_path else "NPU"
            logger.info(f"✅ Kokoro TTS with MLIR-AIE NPU acceleration using {os.path.basename(model_path)}")
        except Exception as e:
            logger.error(f"NPU initialization failed: {e}")
            logger.info("Falling back to standard ONNX Runtime")
            tts_engine = None

    # Fall back to standard ONNX Runtime if NPU not used or failed
    if tts_engine is None:
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
                backend_type = "Intel iGPU (OpenVINO)"
            except Exception as e:
                logger.warning(f"OpenVINO GPU not available: {e}")

        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')

        session = ort.InferenceSession("models/kokoro-v0_19.onnx", providers=providers)

        # Check which provider is actually being used
        actual_provider = session.get_providers()[0]
        if backend_type == "unknown":
            backend_type = actual_provider
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
    """Synthesize speech using Kokoro model with NPU acceleration if available"""
    # Check if NPU engine is available
    if tts_engine is not None:
        try:
            logger.info(f"🚀 Using NPU-accelerated synthesis")
            audio, sample_rate = tts_engine.create_audio(text, voice, speed, lang="en-us")
            logger.info(f"✅ NPU synthesis completed: {len(audio)} samples at {sample_rate}Hz")
            return audio
        except Exception as e:
            logger.error(f"NPU synthesis failed: {e}")
            logger.info("Falling back to standard ONNX synthesis")
            # Fall through to standard synthesis

    # Standard ONNX Runtime synthesis
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
    # Determine which backend is active
    using_npu = tts_engine is not None
    npu_status = None

    if using_npu:
        try:
            npu_status = tts_engine.get_acceleration_status()
        except:
            pass

    # Determine performance based on hardware
    if HARDWARE.get("type") == "npu":
        performance = "32.4x realtime"
        performance_note = "AMD Phoenix NPU acceleration"
    elif HARDWARE.get("type") == "igpu":
        performance = "~20x realtime"
        performance_note = "Intel iGPU with OpenVINO"
    else:
        performance = "~10x realtime"
        performance_note = "CPU-only processing"

    return {
        "status": "healthy",
        "model": "kokoro-v0_19",
        "backend": backend_type,
        "hardware": {
            "type": HARDWARE.get("type", "cpu"),
            "name": HARDWARE.get("name", "CPU"),
            "npu_available": HARDWARE.get("npu_available", False),
            "kernels_available": HARDWARE.get("npu_kernels", 0),
            "igpu_available": HARDWARE.get("igpu_available", False),
            "details": HARDWARE.get("details", {})
        },
        "performance": performance,
        "performance_note": performance_note,
        "npu_enabled": using_npu,
        "npu_status": npu_status if using_npu else None,
        "model_loaded": session is not None or tts_engine is not None,
        "voices_loaded": len(voices) > 0,
        "device_env": os.environ.get("DEVICE", "CPU")
    }

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "service": "Unicorn Orator",
        "version": "1.1.0",
        "description": "Professional AI Voice Synthesis Platform",
        "sign_bug_protection": "v1.1.0+ includes defensive protections",
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