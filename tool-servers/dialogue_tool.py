#!/usr/bin/env python3
"""
Open-WebUI Tool Server: Dialogue & Podcast Generator
Creates multi-voice dialogues, podcasts, and stories using Unicorn Orator
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import json
import base64
import os

app = FastAPI(
    title="Dialogue Generator Tool",
    description="Generate multi-voice dialogues and podcasts",
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
ORATOR_URL = os.getenv("ORATOR_URL", "http://localhost:8880")

# Tool metadata for Open-WebUI
TOOL_INFO = {
    "id": "dialogue_generator",
    "name": "Dialogue & Podcast Generator",
    "description": "Generate multi-voice dialogues, podcasts, and story narrations",
    "version": "1.0.0",
    "author": "Unicorn Commander",
    "license": "MIT",
    "icon": "🎭",
    "capabilities": ["dialogue", "podcast", "story", "multi-voice"]
}

class Message(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    model: str = "dialogue-generator"
    messages: List[Message]
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)
    
class DialogueLine(BaseModel):
    character: str
    text: str
    voice: Optional[str] = None
    emotion: Optional[str] = "neutral"
    pause_after: Optional[float] = 0.5

class ScriptAnalysis(BaseModel):
    type: str  # dialogue, podcast, story
    characters: Dict[str, str]  # character -> voice mapping
    lines: List[DialogueLine]

def parse_script(text: str) -> ScriptAnalysis:
    """Parse user text into dialogue format"""
    
    lines = []
    characters = {}
    script_type = "dialogue"
    
    # Detect script type
    if "podcast" in text.lower():
        script_type = "podcast"
        characters = {
            "Host": "am_michael",
            "Guest": "af_sarah",
            "Co-Host": "af_bella"
        }
    elif "story" in text.lower() or "narrat" in text.lower():
        script_type = "story"
        characters = {
            "Narrator": "am_adam",
            "Character 1": "af_sky",
            "Character 2": "am_liam"
        }
    
    # Parse lines
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for character:dialogue format
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                character = parts[0].strip()
                dialogue = parts[1].strip()
                
                # Assign voice if not already assigned
                if character not in characters:
                    # Auto-assign voices based on character number
                    voice_options = [
                        "af_heart", "am_michael", "bf_emma", "bm_george",
                        "af_sarah", "am_adam", "af_bella", "am_liam"
                    ]
                    voice_idx = len(characters) % len(voice_options)
                    characters[character] = voice_options[voice_idx]
                
                lines.append(DialogueLine(
                    character=character,
                    text=dialogue,
                    voice=characters[character]
                ))
        else:
            # Single line - treat as narrator
            if "Narrator" not in characters:
                characters["Narrator"] = "am_adam"
            
            lines.append(DialogueLine(
                character="Narrator",
                text=line,
                voice=characters["Narrator"]
            ))
    
    return ScriptAnalysis(
        type=script_type,
        characters=characters,
        lines=lines
    )

async def generate_audio(lines: List[DialogueLine]) -> str:
    """Generate audio from dialogue lines"""
    
    audio_segments = []
    
    async with httpx.AsyncClient() as client:
        for line in lines:
            # Generate speech for each line
            response = await client.post(
                f"{ORATOR_URL}/v1/audio/speech",
                json={
                    "text": line.text,
                    "voice": line.voice,
                    "speed": 1.0
                }
            )
            
            if response.status_code == 200:
                # Get audio data
                audio_data = response.content
                audio_segments.append(base64.b64encode(audio_data).decode())
    
    # For now, return the first segment (would need audio mixing for full dialogue)
    # In production, send to dialogue endpoint for proper mixing
    if audio_segments:
        return audio_segments[0]
    
    return ""

@app.get("/")
async def root():
    """Tool information endpoint"""
    return TOOL_INFO

@app.post("/v1/chat/completions")
async def generate_dialogue(request: CompletionRequest):
    """Open-WebUI compatible completion endpoint"""
    
    # Get the last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Parse the script
    script = parse_script(user_message)
    
    if not script.lines:
        return {
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Please provide a script in the format:\n\nCharacter: Dialogue\n\nFor example:\nHost: Welcome to our podcast!\nGuest: Thanks for having me!"
                },
                "finish_reason": "stop"
            }]
        }
    
    # Generate response with script analysis
    response_text = f"🎭 **{script.type.title()} Script Analysis**\n\n"
    response_text += f"**Characters:** {len(script.characters)}\n"
    
    for char, voice in script.characters.items():
        response_text += f"- {char} (voice: {voice})\n"
    
    response_text += f"\n**Lines:** {len(script.lines)}\n\n"
    
    # Show script preview
    response_text += "**Script Preview:**\n"
    for line in script.lines[:5]:  # Show first 5 lines
        response_text += f"- **{line.character}:** {line.text[:100]}...\n"
    
    if len(script.lines) > 5:
        response_text += f"... and {len(script.lines) - 5} more lines\n"
    
    # Add generation button (Open-WebUI will handle this)
    response_text += "\n**Generate Audio:** Processing dialogue...\n"
    
    # Generate audio (optional - could be done on demand)
    # audio_base64 = await generate_audio(script.lines)
    
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

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dialogue_generator"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "13060"))
    uvicorn.run(app, host="0.0.0.0", port=port)