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
        "backend": "ONNX Runtime",
        "model_loaded": session is not None,
        "voices_loaded": len(voices) > 0
    }

@app.get("/api/status")
async def api_status():
    """Return current backend status"""
    return {
        "status": "healthy",
        "device": device,  # This will be "IGPU" for Intel iGPU
        "backend": "OpenVINO" if device in ["GPU", "IGPU"] else "ONNX Runtime",
        "model": "kokoro-v0_19",
        "voices": len(voices)
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

@app.get("/dialogue")
async def dialogue_interface():
    """Dialogue generator interface"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Dialogue Generator - Unicorn Orator</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white min-h-screen p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-4xl font-bold mb-4">🎭 Dialogue Generator</h1>
        <p class="mb-8">Create multi-voice dialogues with different characters</p>
        <div class="bg-white/10 backdrop-blur rounded-lg p-6">
            <h2 class="text-xl mb-4">Coming Soon!</h2>
            <p>The dialogue generator tool server is available at port 13060</p>
            <p class="mt-4">Features:</p>
            <ul class="list-disc ml-6 mt-2">
                <li>Multiple character voices</li>
                <li>Automatic voice assignment</li>
                <li>Script parsing</li>
                <li>Export to audio</li>
            </ul>
            <a href="/web" class="inline-block mt-6 px-4 py-2 bg-purple-600 rounded hover:bg-purple-700">← Back to Main</a>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/podcast")
async def podcast_interface():
    """Podcast creator interface"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Podcast Creator - Unicorn Orator</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white min-h-screen p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-4xl font-bold mb-4">📻 Podcast Creator</h1>
        <p class="mb-8">Generate professional podcasts with intro, segments, and outro</p>
        <div class="bg-white/10 backdrop-blur rounded-lg p-6">
            <h2 class="text-xl mb-4">Coming Soon!</h2>
            <p>The podcast creator tool server is available at port 13062</p>
            <p class="mt-4">Features:</p>
            <ul class="list-disc ml-6 mt-2">
                <li>Professional intro/outro</li>
                <li>Multiple hosts and guests</li>
                <li>Background music</li>
                <li>Episode management</li>
            </ul>
            <a href="/web" class="inline-block mt-6 px-4 py-2 bg-purple-600 rounded hover:bg-purple-700">← Back to Main</a>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/admin")
async def admin_interface():
    """Admin control panel"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel - Unicorn Orator</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white min-h-screen p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-4xl font-bold mb-8">🦄 Unicorn Orator Admin Panel</h1>
        
        <div class="grid md:grid-cols-2 gap-6">
            <!-- Main Service -->
            <div class="bg-white/10 backdrop-blur rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4">Main TTS Service</h2>
                <div class="space-y-3">
                    <div class="flex justify-between items-center">
                        <span>Status:</span>
                        <span id="main-status" class="text-green-400">● Running</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span>Port:</span>
                        <span>8885</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span>Backend:</span>
                        <span id="backend-info">Intel iGPU (OpenVINO)</span>
                    </div>
                </div>
            </div>
            
            <!-- Tool Servers -->
            <div class="bg-white/10 backdrop-blur rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4">Tool Servers</h2>
                <div class="space-y-3">
                    <div class="flex justify-between items-center">
                        <span>🎭 Dialogue (13060)</span>
                        <div>
                            <span id="dialogue-status" class="mr-3">checking...</span>
                            <button onclick="toggleService('dialogue')" class="px-3 py-1 bg-purple-600 rounded hover:bg-purple-700">Toggle</button>
                        </div>
                    </div>
                    <div class="flex justify-between items-center">
                        <span>🎙️ Transcription (13061)</span>
                        <div>
                            <span id="transcription-status" class="mr-3">checking...</span>
                            <button onclick="toggleService('transcription')" class="px-3 py-1 bg-purple-600 rounded hover:bg-purple-700">Toggle</button>
                        </div>
                    </div>
                    <div class="flex justify-between items-center">
                        <span>📻 Podcast (13062)</span>
                        <div>
                            <span id="podcast-status" class="mr-3">checking...</span>
                            <button onclick="toggleService('podcast')" class="px-3 py-1 bg-purple-600 rounded hover:bg-purple-700">Toggle</button>
                        </div>
                    </div>
                </div>
                <div class="mt-4 space-y-2">
                    <button onclick="startAllTools()" class="w-full px-4 py-2 bg-green-600 rounded hover:bg-green-700">Start All Tools</button>
                    <button onclick="stopAllTools()" class="w-full px-4 py-2 bg-red-600 rounded hover:bg-red-700">Stop All Tools</button>
                </div>
            </div>
        </div>
        
        <!-- Quick Actions -->
        <div class="mt-8 bg-white/10 backdrop-blur rounded-lg p-6">
            <h2 class="text-xl font-bold mb-4">Quick Actions</h2>
            <div class="grid md:grid-cols-4 gap-4">
                <a href="/web" class="p-4 bg-gradient-to-r from-purple-600/30 to-pink-600/30 rounded-lg hover:from-purple-600/40 hover:to-pink-600/40 text-center">
                    <div class="text-2xl mb-2">🎤</div>
                    <div>TTS Interface</div>
                </a>
                <a href="/dialogue" class="p-4 bg-gradient-to-r from-blue-600/30 to-purple-600/30 rounded-lg hover:from-blue-600/40 hover:to-purple-600/40 text-center">
                    <div class="text-2xl mb-2">🎭</div>
                    <div>Dialogue Tool</div>
                </a>
                <a href="/podcast" class="p-4 bg-gradient-to-r from-pink-600/30 to-orange-600/30 rounded-lg hover:from-pink-600/40 hover:to-orange-600/40 text-center">
                    <div class="text-2xl mb-2">📻</div>
                    <div>Podcast Tool</div>
                </a>
                <a href="/transcribe" class="p-4 bg-gradient-to-r from-green-600/30 to-teal-600/30 rounded-lg hover:from-green-600/40 hover:to-teal-600/40 text-center">
                    <div class="text-2xl mb-2">🎙️</div>
                    <div>Transcription</div>
                </a>
            </div>
        </div>
        
        <!-- Logs -->
        <div class="mt-8 bg-white/10 backdrop-blur rounded-lg p-6">
            <h2 class="text-xl font-bold mb-4">Container Logs</h2>
            <div class="space-y-2">
                <button onclick="viewLogs('unicorn-orator-standalone')" class="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700">View Main Service Logs</button>
                <button onclick="viewLogs('orator-dialogue-tool')" class="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700">View Dialogue Logs</button>
                <button onclick="viewLogs('orator-transcription-tool')" class="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700">View Transcription Logs</button>
                <button onclick="viewLogs('orator-podcast-tool')" class="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700">View Podcast Logs</button>
            </div>
            <pre id="logs-output" class="mt-4 p-4 bg-black/50 rounded text-green-400 text-xs font-mono max-h-64 overflow-y-auto hidden"></pre>
        </div>
    </div>
    
    <script>
        // Check service status
        async function checkStatus() {
            // Check dialogue
            try {
                const res = await fetch('http://localhost:13060/health');
                document.getElementById('dialogue-status').textContent = res.ok ? '● Running' : '○ Stopped';
                document.getElementById('dialogue-status').className = res.ok ? 'text-green-400 mr-3' : 'text-red-400 mr-3';
            } catch {
                document.getElementById('dialogue-status').textContent = '○ Stopped';
                document.getElementById('dialogue-status').className = 'text-red-400 mr-3';
            }
            
            // Check transcription
            try {
                const res = await fetch('http://localhost:13061/health');
                document.getElementById('transcription-status').textContent = res.ok ? '● Running' : '○ Stopped';
                document.getElementById('transcription-status').className = res.ok ? 'text-green-400 mr-3' : 'text-red-400 mr-3';
            } catch {
                document.getElementById('transcription-status').textContent = '○ Stopped';
                document.getElementById('transcription-status').className = 'text-red-400 mr-3';
            }
            
            // Check podcast
            try {
                const res = await fetch('http://localhost:13062/health');
                document.getElementById('podcast-status').textContent = res.ok ? '● Running' : '○ Stopped';
                document.getElementById('podcast-status').className = res.ok ? 'text-green-400 mr-3' : 'text-red-400 mr-3';
            } catch {
                document.getElementById('podcast-status').textContent = '○ Stopped';
                document.getElementById('podcast-status').className = 'text-red-400 mr-3';
            }
            
            // Check backend
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('backend-info').textContent = data.backend === 'OpenVINO' ? 'Intel iGPU (OpenVINO)' : 'CPU (ONNX Runtime)';
            } catch {}
        }
        
        // Toggle service
        async function toggleService(service) {
            const containerMap = {
                'dialogue': 'orator-dialogue-tool',
                'transcription': 'orator-transcription-tool',
                'podcast': 'orator-podcast-tool'
            };
            
            const container = containerMap[service];
            const statusEl = document.getElementById(service + '-status');
            
            if (statusEl.textContent.includes('Running')) {
                await fetch('/api/admin/stop/' + container, {method: 'POST'});
            } else {
                await fetch('/api/admin/start/' + container, {method: 'POST'});
            }
            
            setTimeout(checkStatus, 2000);
        }
        
        // Start all tools
        async function startAllTools() {
            await fetch('/api/admin/start-all-tools', {method: 'POST'});
            setTimeout(checkStatus, 3000);
        }
        
        // Stop all tools
        async function stopAllTools() {
            await fetch('/api/admin/stop-all-tools', {method: 'POST'});
            setTimeout(checkStatus, 2000);
        }
        
        // View logs
        async function viewLogs(container) {
            const output = document.getElementById('logs-output');
            output.classList.remove('hidden');
            output.textContent = 'Fetching logs...';
            
            try {
                const res = await fetch('/api/admin/logs/' + container);
                const data = await res.json();
                output.textContent = data.logs || 'No logs available';
            } catch {
                output.textContent = 'Failed to fetch logs';
            }
        }
        
        // Check status on load
        checkStatus();
        setInterval(checkStatus, 10000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/stories")
async def stories_interface():
    """Story narration interface"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Story Narration - Unicorn Orator</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white min-h-screen p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-4xl font-bold mb-4">📖 Story Narration</h1>
        <p class="mb-8">Professional voice acting for stories and audiobooks</p>
        <div class="bg-white/10 backdrop-blur rounded-lg p-6">
            <h2 class="text-xl mb-4">Voice Acting Service</h2>
            <p>The story narration tool is available at port 13061</p>
            <p class="mt-4">Perfect for:</p>
            <ul class="list-disc ml-6 mt-2">
                <li>Children's bedtime stories</li>
                <li>Adult fiction audiobooks</li>
                <li>Fairy tales with character voices</li>
                <li>Educational stories</li>
            </ul>
            <p class="mt-4 text-sm text-purple-300">💡 Paste your story script and we'll perform the narration with appropriate voices!</p>
            <a href="/web" class="inline-block mt-6 px-4 py-2 bg-purple-600 rounded hover:bg-purple-700">← Back to Main</a>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/commercials")
async def commercials_interface():
    """Commercial voiceover interface"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Commercial Voiceover - Unicorn Orator</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white min-h-screen p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-4xl font-bold mb-4">📢 Commercial Voiceover</h1>
        <p class="mb-8">Professional voice synthesis for ads and promos</p>
        <div class="bg-white/10 backdrop-blur rounded-lg p-6">
            <h2 class="text-xl mb-4">Voice Acting Service</h2>
            <p>The commercial voiceover tool is available at port 13063</p>
            <p class="mt-4">Perfect for:</p>
            <ul class="list-disc ml-6 mt-2">
                <li>Radio commercials</li>
                <li>TV advertisements</li>
                <li>Web promos</li>
                <li>Product announcements</li>
            </ul>
            <p class="mt-4 text-sm text-purple-300">💡 Provide your ad script and we'll deliver professional voiceover!</p>
            <a href="/web" class="inline-block mt-6 px-4 py-2 bg-purple-600 rounded hover:bg-purple-700">← Back to Main</a>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)

# Admin API endpoints
@app.post("/api/admin/start/{container}")
async def start_container(container: str):
    """Start a Docker container"""
    try:
        result = subprocess.run(["docker", "start", container], capture_output=True, text=True)
        return {"success": result.returncode == 0, "message": result.stdout or result.stderr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/stop/{container}")
async def stop_container(container: str):
    """Stop a Docker container"""
    try:
        result = subprocess.run(["docker", "stop", container], capture_output=True, text=True)
        return {"success": result.returncode == 0, "message": result.stdout or result.stderr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/start-all-tools")
async def start_all_tools():
    """Start all tool servers"""
    containers = ["orator-dialogue-tool", "orator-transcription-tool", "orator-podcast-tool"]
    results = []
    for container in containers:
        try:
            result = subprocess.run(["docker", "start", container], capture_output=True, text=True)
            results.append({"container": container, "success": result.returncode == 0})
        except:
            results.append({"container": container, "success": False})
    return {"results": results}

@app.post("/api/admin/stop-all-tools")
async def stop_all_tools():
    """Stop all tool servers"""
    containers = ["orator-dialogue-tool", "orator-transcription-tool", "orator-podcast-tool"]
    results = []
    for container in containers:
        try:
            result = subprocess.run(["docker", "stop", container], capture_output=True, text=True)
            results.append({"container": container, "success": result.returncode == 0})
        except:
            results.append({"container": container, "success": False})
    return {"results": results}

@app.get("/api/admin/logs/{container}")
async def get_container_logs(container: str):
    """Get Docker container logs"""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", "100", container],
            capture_output=True, text=True
        )
        return {"logs": result.stdout or result.stderr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for web interface assets
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")