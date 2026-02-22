"""
Unicorn Orator - Pocket TTS Server
====================================

FastAPI server using Pocket TTS (Kyutai Labs) as the primary TTS engine
with Kokoro ONNX as a fallback. Maintains the same API contract as the
original Kokoro-only server.

Endpoints:
    POST /v1/audio/speech     - Generate speech from text (WAV response)
    POST /v1/audio/long       - Generate speech from long text with sentence pauses
    POST /v1/audio/dialogue/parse - Parse dialogue script, return characters
    POST /v1/audio/dialogue   - Generate multi-voice dialogue audio
    GET  /v1/voices/clone/status - Check if voice cloning is available
    POST /v1/settings/hf-token   - Save HuggingFace token
    POST /v1/voices/clone     - Clone a voice from uploaded audio
    DELETE /v1/voices/{name}  - Delete a custom cloned voice
    GET  /voices              - List available voices
    GET  /health              - Health check
    GET  /web                 - Web interface
    GET  /                    - API info

Voice resolution order:
    1. Custom voices from /app/voices/*.safetensors
    2. Built-in Pocket TTS voices (alba, marius, javert, jean, fantine, cosette, eponine, azelma)
    3. Kokoro ONNX fallback (if available)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import io
import json
import re
import struct
import soundfile as sf
from typing import Optional, Dict, List
import logging
import os
import subprocess
import tempfile
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unicorn Orator - Pocket TTS Voice Synthesis")

# ============================================================
# Pocket TTS Engine
# ============================================================

pocket_model = None
pocket_voices: Dict[str, dict] = {}  # voice_name -> state dict
pocket_sample_rate = 24000
POCKET_BUILTIN = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices")

# Thread lock — Pocket TTS is NOT thread-safe
_tts_lock = threading.Lock()


def init_pocket_tts():
    """Load Pocket TTS model and all voice states."""
    global pocket_model, pocket_sample_rate

    try:
        from pocket_tts import TTSModel
        logger.info("Loading Pocket TTS model...")
        t0 = time.time()
        pocket_model = TTSModel.load_model()
        pocket_sample_rate = pocket_model.sample_rate
        load_time = time.time() - t0
        logger.info(f"Pocket TTS model loaded in {load_time:.1f}s (sample rate: {pocket_sample_rate})")
    except Exception as e:
        logger.error(f"Failed to load Pocket TTS: {e}")
        return False

    _load_all_voices()

    logger.info(f"Pocket TTS ready with {len(pocket_voices)} voice(s): {list(pocket_voices.keys())}")
    return True


def _load_all_voices():
    """Load custom + built-in voices into pocket_voices dict."""
    # Load custom voices from safetensors files
    if os.path.isdir(VOICES_DIR):
        for fname in sorted(os.listdir(VOICES_DIR)):
            if fname.endswith(".safetensors"):
                voice_name = os.path.splitext(fname)[0]
                fpath = os.path.join(VOICES_DIR, fname)
                try:
                    state = pocket_model.get_state_for_audio_prompt(fpath)
                    pocket_voices[voice_name] = state
                    size_kb = os.path.getsize(fpath) / 1024
                    logger.info(f"  Loaded custom voice: {voice_name} ({size_kb:.0f} KB)")
                except Exception as e:
                    logger.warning(f"  Failed to load custom voice {voice_name}: {e}")

    # Load built-in voices that aren't already overridden by custom ones
    for voice_name in POCKET_BUILTIN:
        if voice_name not in pocket_voices:
            try:
                state = pocket_model.get_state_for_audio_prompt(voice_name)
                pocket_voices[voice_name] = state
                logger.info(f"  Loaded built-in voice: {voice_name}")
            except Exception as e:
                logger.warning(f"  Failed to load built-in voice {voice_name}: {e}")


# ============================================================
# Kokoro ONNX Fallback
# ============================================================

kokoro_session = None
kokoro_voices = {}
kokoro_voice_embeddings = {}
kokoro_phoneme_to_id = {}


def init_kokoro_fallback():
    """Load Kokoro ONNX model as fallback TTS."""
    global kokoro_session

    # Load phoneme mapping
    try:
        with open("phoneme_mapping.json", "r") as f:
            config = json.load(f)
            kokoro_phoneme_to_id.update(config.get("vocab", {}))
            logger.info(f"Kokoro: Loaded {len(kokoro_phoneme_to_id)} phoneme mappings")
    except Exception as e:
        logger.warning(f"Kokoro: No phoneme mapping: {e}")
        return False

    # Load voice embeddings
    try:
        import zipfile
        with zipfile.ZipFile("models/voices-v1.0.bin", "r") as zf:
            voice_files = [f for f in zf.namelist() if f.endswith('.npy')]
            for voice_file in voice_files:
                voice_name = os.path.splitext(os.path.basename(voice_file))[0]
                with zf.open(voice_file) as f:
                    embedding = np.load(f).astype(np.float32)
                    if embedding.shape == (256,):
                        pass
                    elif len(embedding.shape) == 3 and embedding.shape[2] == 256:
                        embedding = np.mean(embedding, axis=(0, 1))
                    elif len(embedding.shape) == 2 and embedding.shape[1] == 256:
                        embedding = np.mean(embedding, axis=0)
                    else:
                        continue
                kokoro_voices[voice_name] = voice_name.replace('_', ' ').title()
                kokoro_voice_embeddings[voice_name] = embedding
            logger.info(f"Kokoro: Loaded {len(kokoro_voices)} voices")
    except Exception as e:
        logger.warning(f"Kokoro: No voice embeddings: {e}")
        return False

    # Load ONNX model
    try:
        import onnxruntime as ort
        kokoro_session = ort.InferenceSession("models/kokoro-v0_19.onnx",
                                              providers=['CPUExecutionProvider'])
        logger.info("Kokoro: ONNX model loaded (CPU fallback)")
        return True
    except Exception as e:
        logger.warning(f"Kokoro: No ONNX model: {e}")
        return False


def text_to_phonemes(text: str, lang: str = 'en-us') -> str:
    """Convert text to phonemes using espeak-ng."""
    try:
        cmd = ['espeak-ng', '-q', '-x', '--ipa=3', f'-v{lang}', text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            phonemes = result.stdout.strip()
            phonemes = ' '.join(phonemes.split())
            phonemes = phonemes.replace('\u200d', '').replace('\u200c', '')
            return phonemes
        return text
    except Exception:
        return text


def text_to_tokens(text: str, voice: str = "af") -> np.ndarray:
    """Convert text to token IDs using phoneme conversion."""
    lang = 'en-us' if voice.startswith('a') else 'en-gb' if voice.startswith('b') else 'en-us'
    phonemes = text_to_phonemes(text, lang)

    tokens = [0]  # padding
    i = 0
    while i < len(phonemes):
        matched = False
        for length in range(3, 0, -1):
            if i + length <= len(phonemes):
                substring = phonemes[i:i+length]
                if substring in kokoro_phoneme_to_id:
                    tokens.append(kokoro_phoneme_to_id[substring])
                    i += length
                    matched = True
                    break
        if not matched:
            i += 1
    tokens.append(0)  # padding
    return np.array([tokens], dtype=np.int64)


def synthesize_kokoro(text: str, voice: str = "af", speed: float = 1.0):
    """Synthesize speech using Kokoro ONNX fallback."""
    if kokoro_session is None:
        return None

    if voice not in kokoro_voice_embeddings:
        voice = "af"
    if voice not in kokoro_voice_embeddings:
        return None

    try:
        style_embedding = kokoro_voice_embeddings[voice].reshape(1, 256)
        tokens = text_to_tokens(text, voice)
        inputs = {
            "tokens": tokens,
            "style": style_embedding,
            "speed": np.array([speed], dtype=np.float32)
        }
        outputs = kokoro_session.run(None, inputs)
        audio = outputs[0].squeeze()
        audio = np.clip(audio, -1, 1)
        return audio
    except Exception as e:
        logger.error(f"Kokoro synthesis failed: {e}")
        return None


# ============================================================
# Unified synthesis
# ============================================================

def synthesize_speech(text: str, voice: str = "alba", speed: float = 1.0):
    """Synthesize speech. Tries Pocket TTS first, then Kokoro fallback.

    Returns (audio_ndarray, sample_rate, backend_used).
    """
    # Try Pocket TTS
    if pocket_model is not None:
        # Resolve voice
        state = pocket_voices.get(voice)
        if state is None:
            # Try default
            state = pocket_voices.get("alba")
        if state is None and pocket_voices:
            # Use first available
            state = next(iter(pocket_voices.values()))

        if state is not None:
            try:
                with _tts_lock:
                    import torch
                    audio_tensor = pocket_model.generate_audio(state, text)
                    audio = audio_tensor.numpy()
                    if audio.ndim > 1:
                        audio = audio.squeeze()
                    # Normalize
                    peak = np.abs(audio).max()
                    if peak > 0:
                        audio = audio / max(peak, 1.0)
                    return audio, pocket_sample_rate, "pocket-tts"
            except Exception as e:
                logger.error(f"Pocket TTS synthesis failed: {e}")

    # Kokoro fallback
    kokoro_voice = voice
    # Map pocket voice names to kokoro voice names if needed
    if voice not in kokoro_voice_embeddings:
        kokoro_voice = "af"
    audio = synthesize_kokoro(text, kokoro_voice, speed)
    if audio is not None:
        return audio, 24000, "kokoro-onnx"

    # Total failure — return silence
    logger.warning("All TTS backends failed, returning silence")
    return np.zeros(24000, dtype=np.float32), 24000, "silence"


# ============================================================
# Long Text Utilities
# ============================================================

def split_text_into_sentences(text: str) -> List[str]:
    """Split text into sentences on .!? followed by whitespace or end-of-string."""
    # Split on sentence-ending punctuation followed by whitespace or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    return [s.strip() for s in parts if s.strip()]


def concatenate_audio_segments(segments: List[np.ndarray], sample_rate: int, pause_ms: int = 300) -> np.ndarray:
    """Join audio arrays with silence gaps between them."""
    if not segments:
        return np.zeros(sample_rate, dtype=np.float32)

    pause_samples = int(sample_rate * pause_ms / 1000)
    silence = np.zeros(pause_samples, dtype=np.float32)

    parts = []
    for i, seg in enumerate(segments):
        parts.append(seg)
        if i < len(segments) - 1:
            parts.append(silence)

    return np.concatenate(parts)


# ============================================================
# Dialogue Utilities
# ============================================================

def parse_dialogue_lines(text: str) -> List[dict]:
    """Parse 'Character: line' format. Unattributed lines become Narrator."""
    lines = []
    for raw_line in text.strip().split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        # Match "Character: dialogue text"
        match = re.match(r'^([A-Za-z0-9_ -]+):\s*(.+)$', raw_line)
        if match:
            lines.append({"character": match.group(1).strip(), "text": match.group(2).strip()})
        else:
            lines.append({"character": "Narrator", "text": raw_line})
    return lines


# ============================================================
# Initialize engines on startup
# ============================================================

logger.info("=" * 60)
logger.info("Unicorn Orator - Pocket TTS Engine")
logger.info("=" * 60)

pocket_ok = init_pocket_tts()
kokoro_ok = init_kokoro_fallback()

if pocket_ok:
    logger.info("Primary engine: Pocket TTS")
elif kokoro_ok:
    logger.warning("Pocket TTS unavailable — using Kokoro ONNX only")
else:
    logger.error("No TTS engine available! Service will return silence.")


# ============================================================
# API Models
# ============================================================

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "alba"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False


class LongTextRequest(BaseModel):
    text: str
    voice: Optional[str] = "alba"
    speed: Optional[float] = 1.0
    pause_ms: Optional[int] = 300


class DialogueRequest(BaseModel):
    script: str
    voice_map: Optional[Dict[str, str]] = None
    speed: Optional[float] = 1.0
    pause_ms: Optional[int] = 500


class DialogueParseRequest(BaseModel):
    script: str


class HFTokenRequest(BaseModel):
    token: str


# ============================================================
# API Endpoints — Core TTS
# ============================================================

@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest):
    try:
        logger.info(f"Synthesizing: \"{request.text[:80]}...\" voice={request.voice}")
        t0 = time.time()

        audio_data, sample_rate, backend = synthesize_speech(
            request.text,
            request.voice,
            request.speed
        )

        elapsed = time.time() - t0
        duration = len(audio_data) / sample_rate
        logger.info(f"Done: {duration:.2f}s audio in {elapsed:.2f}s "
                     f"({duration/max(elapsed,0.001):.1f}x RT) via {backend}")

        # Convert to WAV
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
        audio_bytes.seek(0)

        return StreamingResponse(
            audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"}
        )

    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# API Endpoints — Long Text
# ============================================================

@app.post("/v1/audio/long")
async def long_text_to_speech(request: LongTextRequest):
    if len(request.text) > 50000:
        raise HTTPException(status_code=400, detail="Text exceeds 50,000 character limit")

    pause_ms = max(50, min(2000, request.pause_ms or 300))

    try:
        sentences = split_text_into_sentences(request.text)
        if not sentences:
            raise HTTPException(status_code=400, detail="No text to synthesize")

        logger.info(f"Long text: {len(sentences)} sentences, {len(request.text)} chars, voice={request.voice}")
        t0 = time.time()

        segments = []
        sr = pocket_sample_rate
        for i, sentence in enumerate(sentences):
            audio_data, sample_rate, backend = synthesize_speech(sentence, request.voice, request.speed)
            segments.append(audio_data)
            sr = sample_rate
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(sentences)} sentences")

        combined = concatenate_audio_segments(segments, sr, pause_ms)

        elapsed = time.time() - t0
        duration = len(combined) / sr
        logger.info(f"Long text done: {duration:.2f}s audio in {elapsed:.2f}s "
                     f"({duration/max(elapsed,0.001):.1f}x RT)")

        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, combined, sr, format='WAV')
        audio_bytes.seek(0)

        return StreamingResponse(
            audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=long-speech.wav"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Long text TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# API Endpoints — Dialogue
# ============================================================

@app.post("/v1/audio/dialogue/parse")
async def parse_dialogue(request: DialogueParseRequest):
    lines = parse_dialogue_lines(request.script)
    if not lines:
        raise HTTPException(status_code=400, detail="No dialogue lines found")

    # Count lines per character
    char_counts = {}
    for line in lines:
        char_counts[line["character"]] = char_counts.get(line["character"], 0) + 1

    # Auto-assign voices round-robin from available Pocket TTS voices
    available = list(pocket_voices.keys()) if pocket_voices else POCKET_BUILTIN
    characters = []
    for i, (name, count) in enumerate(char_counts.items()):
        voice = available[i % len(available)]
        characters.append({"name": name, "voice": voice, "line_count": count})

    return {
        "characters": characters,
        "total_lines": len(lines),
    }


@app.post("/v1/audio/dialogue")
async def dialogue_to_speech(request: DialogueRequest):
    if len(request.script) > 50000:
        raise HTTPException(status_code=400, detail="Script exceeds 50,000 character limit")

    pause_ms = max(50, min(2000, request.pause_ms or 500))

    lines = parse_dialogue_lines(request.script)
    if not lines:
        raise HTTPException(status_code=400, detail="No dialogue lines found")

    # Build voice map — use provided overrides or auto-assign
    voice_map = {}
    if request.voice_map:
        voice_map = request.voice_map

    # Auto-assign any characters not in voice_map
    available = list(pocket_voices.keys()) if pocket_voices else POCKET_BUILTIN
    auto_idx = 0
    all_characters = list(dict.fromkeys(line["character"] for line in lines))
    for char_name in all_characters:
        if char_name not in voice_map:
            voice_map[char_name] = available[auto_idx % len(available)]
            auto_idx += 1

    try:
        logger.info(f"Dialogue: {len(lines)} lines, {len(all_characters)} characters")
        t0 = time.time()

        segments = []
        sr = pocket_sample_rate
        for i, line in enumerate(lines):
            voice = voice_map.get(line["character"], "alba")
            audio_data, sample_rate, backend = synthesize_speech(line["text"], voice, request.speed)
            segments.append(audio_data)
            sr = sample_rate

        combined = concatenate_audio_segments(segments, sr, pause_ms)

        elapsed = time.time() - t0
        duration = len(combined) / sr
        logger.info(f"Dialogue done: {duration:.2f}s audio in {elapsed:.2f}s "
                     f"({duration/max(elapsed,0.001):.1f}x RT)")

        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, combined, sr, format='WAV')
        audio_bytes.seek(0)

        return StreamingResponse(
            audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=dialogue.wav"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dialogue TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# API Endpoints — Voice Cloning
# ============================================================

@app.get("/v1/voices/clone/status")
async def clone_status():
    has_cloning = False
    reason = "Pocket TTS model not loaded"
    hf_token_set = bool(os.environ.get("HF_TOKEN", "").strip())

    if pocket_model is not None:
        has_cloning = hasattr(pocket_model, "get_state_for_audio_prompt")
        if has_cloning:
            reason = "Voice cloning available"
        else:
            reason = "Model does not support voice cloning"

    return {
        "available": has_cloning,
        "reason": reason,
        "hf_token_set": hf_token_set,
    }


@app.post("/v1/settings/hf-token")
async def save_hf_token(request: HFTokenRequest):
    token = request.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token cannot be empty")

    # Save to file so it persists
    token_path = "/app/.hf_token"
    try:
        with open(token_path, "w") as f:
            f.write(token)
        os.environ["HF_TOKEN"] = token
        logger.info("HF token saved and set in environment")
        return {"status": "saved", "note": "Container restart may be needed to reload model with cloning capabilities"}
    except Exception as e:
        logger.error(f"Failed to save HF token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/voices/clone")
async def clone_voice(name: str = Form(...), audio: UploadFile = File(...)):
    if pocket_model is None:
        raise HTTPException(status_code=503, detail="Pocket TTS model not loaded")

    if not hasattr(pocket_model, "get_state_for_audio_prompt"):
        raise HTTPException(status_code=503, detail="Voice cloning not available in this model")

    # Validate name
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', name.strip().lower())
    if not clean_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    if clean_name in POCKET_BUILTIN:
        raise HTTPException(status_code=400, detail=f"Cannot overwrite built-in voice '{clean_name}'")

    # Save uploaded file to temp
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Clone voice
        with _tts_lock:
            state = pocket_model.get_state_for_audio_prompt(tmp_path, truncate=True)

        # Export to safetensors
        safetensors_path = os.path.join(VOICES_DIR, f"{clean_name}.safetensors")

        # Try to use pocket_tts export if available, otherwise use safetensors directly
        try:
            from pocket_tts import export_model_state
            export_model_state(state, safetensors_path)
        except ImportError:
            try:
                from safetensors.torch import save_file
                save_file(state, safetensors_path)
            except ImportError:
                import torch
                torch.save(state, safetensors_path.replace('.safetensors', '.pt'))
                safetensors_path = safetensors_path.replace('.safetensors', '.pt')

        # Add to active voices
        pocket_voices[clean_name] = state
        logger.info(f"Voice cloned: {clean_name}")

        return {"status": "created", "voice": clean_name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    clean_name = name.strip().lower()

    if clean_name in POCKET_BUILTIN:
        raise HTTPException(status_code=400, detail=f"Cannot delete built-in voice '{clean_name}'")

    if clean_name not in pocket_voices:
        raise HTTPException(status_code=404, detail=f"Voice '{clean_name}' not found")

    # Remove safetensors file
    for ext in [".safetensors", ".pt"]:
        fpath = os.path.join(VOICES_DIR, f"{clean_name}{ext}")
        if os.path.isfile(fpath):
            os.unlink(fpath)
            logger.info(f"Deleted voice file: {fpath}")

    # Remove from active voices
    del pocket_voices[clean_name]
    logger.info(f"Voice removed: {clean_name}")

    return {"status": "deleted", "voice": clean_name}


# ============================================================
# API Endpoints — Info
# ============================================================

@app.get("/voices")
async def list_voices():
    all_voices = {}

    # Pocket TTS voices
    for name in pocket_voices:
        all_voices[name] = {
            "engine": "pocket-tts",
            "type": "custom" if os.path.isfile(os.path.join(VOICES_DIR, f"{name}.safetensors")) else "built-in"
        }

    # Kokoro voices (that aren't already listed)
    for name in kokoro_voices:
        if name not in all_voices:
            all_voices[name] = {"engine": "kokoro-onnx", "type": "built-in"}

    return {
        "voices": list(all_voices.keys()),
        "details": all_voices,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine": "pocket-tts" if pocket_model is not None else "kokoro-onnx" if kokoro_session is not None else "none",
        "model": "pocket-tts-100M" if pocket_model is not None else "kokoro-v0_19",
        "backend": "pocket-tts" if pocket_model is not None else "kokoro-onnx-cpu",
        "hardware": {
            "type": "cpu",
            "name": "CPU (optimized)",
        },
        "performance": "~3.5x realtime",
        "performance_note": "Pocket TTS 100M params, CPU-optimized" if pocket_model else "Kokoro ONNX CPU fallback",
        "pocket_tts_loaded": pocket_model is not None,
        "kokoro_fallback_loaded": kokoro_session is not None,
        "voices_loaded": len(pocket_voices),
        "kokoro_voices_loaded": len(kokoro_voices),
        "sample_rate": pocket_sample_rate,
    }


@app.get("/")
async def root():
    return {
        "service": "Unicorn Orator",
        "version": "3.0.0",
        "description": "Professional AI Voice Synthesis — Pocket TTS Engine",
        "engine": "pocket-tts" if pocket_model is not None else "kokoro-onnx",
        "endpoints": {
            "/v1/audio/speech": "POST - Generate speech from text",
            "/v1/audio/long": "POST - Generate speech from long text with pauses",
            "/v1/audio/dialogue/parse": "POST - Parse dialogue script",
            "/v1/audio/dialogue": "POST - Generate multi-voice dialogue",
            "/v1/voices/clone/status": "GET - Voice cloning status",
            "/v1/voices/clone": "POST - Clone a voice from audio",
            "/v1/voices/{name}": "DELETE - Delete a custom voice",
            "/voices": "GET - List available voices",
            "/health": "GET - Health check",
            "/web": "GET - Web interface"
        }
    }


@app.get("/web")
async def web_interface():
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Web interface not found")


# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
