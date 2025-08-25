from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import whisperx
import os
import tempfile
import torch
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhisperX STT Service")

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Load model once at startup
logger.info(f"Loading WhisperX model: {MODEL_SIZE} on {DEVICE}")
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

# Load alignment model for English
logger.info("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None)
):
    """Transcribe audio file with optional speaker diarization"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        logger.info(f"Processing audio file: {file.filename}")
        
        # Load audio
        audio = whisperx.load_audio(tmp_path)
        
        # Transcribe with WhisperX
        logger.info("Transcribing...")
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        
        # Align whisper output
        logger.info("Aligning...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)
        
        # Optional: Speaker diarization
        if diarize and HF_TOKEN:
            logger.info("Performing speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Format response
        text = " ".join([segment["text"] for segment in result["segments"]])
        
        return {
            "text": text,
            "segments": result["segments"],
            "language": result.get("language", "en"),
            "words": result.get("word_segments", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise
        
    finally:
        os.unlink(tmp_path)
        gc.collect()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE
    }

@app.get("/")
async def root():
    return {
        "service": "WhisperX STT",
        "version": "1.0",
        "endpoints": {
            "/v1/audio/transcriptions": "POST - Transcribe audio",
            "/health": "GET - Health check"
        }
    }
