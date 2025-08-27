#!/usr/bin/env python3
"""
Open-WebUI Tool Server: Transcription Service
Provides speech-to-text using Unicorn Amanuensis (WhisperX)
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import json
import base64
import os
import io

app = FastAPI(
    title="Transcription Tool",
    description="Speech-to-text transcription with speaker diarization",
    version="1.0.0"
)

# CORS for Open-WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
AMANUENSIS_URL = os.getenv("AMANUENSIS_URL", "http://localhost:9000")

# Tool metadata for Open-WebUI
TOOL_INFO = {
    "id": "transcription_service",
    "name": "Speech Transcription",
    "description": "Convert audio to text with speaker diarization and timestamps",
    "version": "1.0.0",
    "author": "Unicorn Commander",
    "license": "MIT",
    "icon": "🎙️",
    "capabilities": ["transcription", "diarization", "timestamps", "multi-language"]
}

class Message(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    model: str = "transcription-service"
    messages: List[Message]
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)

class TranscriptionResult(BaseModel):
    text: str
    segments: Optional[List[Dict]] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    speakers: Optional[List[str]] = None

async def transcribe_audio(audio_data: bytes, language: Optional[str] = None) -> TranscriptionResult:
    """Send audio to Unicorn Amanuensis for transcription"""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Prepare multipart form data
        files = {
            "file": ("audio.wav", io.BytesIO(audio_data), "audio/wav")
        }
        
        data = {}
        if language:
            data["language"] = language
        
        # Send to Amanuensis
        response = await client.post(
            f"{AMANUENSIS_URL}/v1/audio/transcriptions",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract speaker information if available
            speakers = []
            segments = result.get("segments", [])
            
            for segment in segments:
                speaker = segment.get("speaker")
                if speaker and speaker not in speakers:
                    speakers.append(speaker)
            
            return TranscriptionResult(
                text=result.get("text", ""),
                segments=segments,
                language=result.get("language"),
                duration=result.get("duration"),
                speakers=speakers if speakers else None
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Transcription failed: {response.text}"
            )

def format_transcript(result: TranscriptionResult) -> str:
    """Format transcription result for display"""
    
    output = "🎙️ **Transcription Result**\n\n"
    
    # Basic info
    if result.language:
        output += f"**Language:** {result.language}\n"
    if result.duration:
        output += f"**Duration:** {result.duration:.1f} seconds\n"
    if result.speakers:
        output += f"**Speakers:** {', '.join(result.speakers)}\n"
    
    output += "\n**Full Transcript:**\n"
    
    # If we have segments with speaker diarization
    if result.segments and result.speakers:
        current_speaker = None
        
        for segment in result.segments:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            
            if speaker != current_speaker:
                output += f"\n**{speaker}:** "
                current_speaker = speaker
            
            output += f"{text} "
    else:
        # Just the plain text
        output += result.text
    
    # Add timestamps if available
    if result.segments and any(s.get("start") for s in result.segments):
        output += "\n\n**Detailed Segments:**\n"
        
        for i, segment in enumerate(result.segments[:10]):  # Show first 10
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "")
            
            timestamp = f"[{start:.1f}s - {end:.1f}s]"
            speaker_label = f"({speaker})" if speaker else ""
            
            output += f"{timestamp} {speaker_label} {text}\n"
        
        if len(result.segments) > 10:
            output += f"... and {len(result.segments) - 10} more segments\n"
    
    return output

@app.get("/")
async def root():
    """Tool information endpoint"""
    return TOOL_INFO

@app.post("/v1/chat/completions")
async def process_transcription(request: CompletionRequest):
    """Open-WebUI compatible completion endpoint"""
    
    # Get the last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Check if message contains audio data (base64 encoded)
    if user_message.startswith("data:audio"):
        # Extract base64 audio data
        try:
            # Remove data URL prefix
            audio_b64 = user_message.split(",")[1]
            audio_data = base64.b64decode(audio_b64)
            
            # Transcribe the audio
            result = await transcribe_audio(audio_data)
            
            # Format the response
            response_text = format_transcript(result)
            
        except Exception as e:
            response_text = f"❌ Error transcribing audio: {str(e)}"
    
    else:
        # Provide instructions
        response_text = """🎙️ **Speech Transcription Tool**

To transcribe audio:
1. Upload an audio file or record audio
2. The tool will provide:
   - Full transcript
   - Speaker diarization (who said what)
   - Timestamps for each segment
   - Language detection

**Supported formats:** WAV, MP3, M4A, FLAC, OGG
**Languages:** 100+ languages supported

**Features:**
- Speaker separation
- Word-level timestamps
- Automatic punctuation
- Multiple speakers support

Please upload or record audio to begin transcription."""
    
    return {
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(user_message),
            "completion_tokens": len(response_text),
            "total_tokens": len(user_message) + len(response_text)
        }
    }

@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """Direct transcription endpoint (OpenAI compatible)"""
    
    # Read audio file
    audio_data = await file.read()
    
    # Transcribe
    result = await transcribe_audio(audio_data, language)
    
    # Return OpenAI-compatible response
    return {
        "text": result.text,
        "segments": result.segments,
        "language": result.language,
        "duration": result.duration
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    
    # Check if Amanuensis is available
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{AMANUENSIS_URL}/health")
            amanuensis_healthy = response.status_code == 200
    except:
        amanuensis_healthy = False
    
    return {
        "status": "healthy" if amanuensis_healthy else "degraded",
        "service": "transcription_service",
        "amanuensis": "connected" if amanuensis_healthy else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "13061"))
    uvicorn.run(app, host="0.0.0.0", port=port)