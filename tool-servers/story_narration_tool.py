#!/usr/bin/env python3
"""
Open-WebUI Tool Server: Story Narration
Provides voice synthesis for stories - kids stories, adult fiction, audiobooks
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
    title="Story Narration Tool",
    description="Professional narration for stories, audiobooks, and fiction",
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
    "id": "story_narration",
    "name": "Story Narration",
    "description": "Professional voice narration for stories and audiobooks",
    "version": "1.0.0",
    "author": "Unicorn Commander",
    "license": "MIT",
    "icon": "📖",
    "capabilities": ["kids-stories", "adult-fiction", "audiobooks", "fairy-tales"]
}

class Message(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    model: str = "story-narration"
    messages: List[Message]
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)

class StorySegment(BaseModel):
    type: str  # narration, dialogue, description, action
    text: str
    voice: Optional[str] = None
    emotion: Optional[str] = "neutral"
    pace: Optional[str] = "normal"  # slow, normal, fast

class StoryScript(BaseModel):
    title: str
    story_type: str  # kids, adult, audiobook
    narrator_voice: str
    character_voices: Dict[str, str]
    segments: List[StorySegment]

def parse_story_script(text: str) -> StoryScript:
    """Parse story text into narration format"""
    
    segments = []
    title = "Untitled Story"
    story_type = "adult"  # default
    
    # Detect story type from keywords
    if any(word in text.lower() for word in ["kids", "children", "fairy", "bedtime"]):
        story_type = "kids"
        narrator_voice = "af_bella"  # Warm, friendly voice for kids
    else:
        narrator_voice = "am_adam"  # Professional narrator voice
    
    character_voices = {}
    voice_options = [
        "af_sarah", "am_michael", "af_sky", "am_liam",
        "bf_emma", "bm_george", "af_heart", "am_echo"
    ]
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for metadata
        if line.startswith("Title:"):
            title = line.replace("Title:", "").strip()
        elif line.startswith("Type:"):
            type_str = line.replace("Type:", "").strip().lower()
            if "kid" in type_str:
                story_type = "kids"
        
        # Parse story content
        elif ":" in line and not line.startswith("http"):
            # Character dialogue
            parts = line.split(":", 1)
            if len(parts) == 2:
                character = parts[0].strip()
                dialogue = parts[1].strip()
                
                # Assign voice if new character
                if character not in character_voices and character != "Narrator":
                    voice_idx = len(character_voices) % len(voice_options)
                    character_voices[character] = voice_options[voice_idx]
                
                # Determine emotion based on content
                emotion = "neutral"
                if "!" in dialogue:
                    emotion = "excited"
                elif "?" in dialogue:
                    emotion = "curious"
                elif any(word in dialogue.lower() for word in ["sad", "cry", "tear"]):
                    emotion = "sad"
                elif any(word in dialogue.lower() for word in ["happy", "joy", "laugh"]):
                    emotion = "happy"
                
                segments.append(StorySegment(
                    type="dialogue",
                    text=dialogue,
                    voice=character_voices.get(character, narrator_voice),
                    emotion=emotion,
                    pace="normal" if story_type == "adult" else "slow"
                ))
        else:
            # Narration
            pace = "slow" if story_type == "kids" else "normal"
            
            # Check for action sequences
            if any(word in line.lower() for word in ["suddenly", "quickly", "ran", "jumped"]):
                pace = "fast"
            
            segments.append(StorySegment(
                type="narration",
                text=line,
                voice=narrator_voice,
                emotion="neutral",
                pace=pace
            ))
    
    # Add opening if none exists
    if not segments or not any(s.text.lower().startswith("once upon") for s in segments[:3]):
        if story_type == "kids":
            opening = f"Hello children! Today's story is called {title}. Are you ready? Let's begin!"
        else:
            opening = f"You are about to hear {title}."
        
        segments.insert(0, StorySegment(
            type="narration",
            text=opening,
            voice=narrator_voice,
            emotion="warm" if story_type == "kids" else "neutral",
            pace="slow" if story_type == "kids" else "normal"
        ))
    
    # Add closing if none exists
    if story_type == "kids" and not any("the end" in s.text.lower() for s in segments[-3:]):
        segments.append(StorySegment(
            type="narration",
            text="And that's the end of our story! Did you enjoy it? Sweet dreams!",
            voice=narrator_voice,
            emotion="warm",
            pace="slow"
        ))
    
    return StoryScript(
        title=title,
        story_type=story_type,
        narrator_voice=narrator_voice,
        character_voices=character_voices,
        segments=segments
    )

def format_story_response(script: StoryScript) -> str:
    """Format story script for display"""
    
    output = f"📖 **Story Narration Ready**\n\n"
    output += f"**Title:** {script.title}\n"
    story_type_text = "Children's Story" if script.story_type == "kids" else "Adult Fiction"
    output += f"**Type:** {story_type_text}\n"
    output += f"**Narrator:** {script.narrator_voice}\n"
    
    if script.character_voices:
        output += f"**Characters:** {len(script.character_voices)}\n"
        for char, voice in list(script.character_voices.items())[:5]:
            output += f"  - {char}: {voice}\n"
    
    output += f"\n**Segments:** {len(script.segments)}\n\n"
    
    # Show preview
    output += "**Preview:**\n"
    for i, segment in enumerate(script.segments[:5]):
        if segment.type == "dialogue":
            output += f"💬 {segment.text[:80]}...\n"
        else:
            output += f"📝 {segment.text[:80]}...\n"
    
    if len(script.segments) > 5:
        output += f"... and {len(script.segments) - 5} more segments\n"
    
    output += "\n**Narration Settings:**\n"
    if script.story_type == "kids":
        output += "- Slower pace for easy listening\n"
        output += "- Warm, expressive narration\n"
        output += "- Character voices with personality\n"
    else:
        output += "- Professional audiobook quality\n"
        output += "- Natural pacing and rhythm\n"
        output += "- Distinct character voices\n"
    
    return output

@app.get("/")
async def root():
    """Tool information endpoint"""
    return TOOL_INFO

@app.post("/v1/chat/completions")
async def process_story(request: CompletionRequest):
    """Open-WebUI compatible completion endpoint"""
    
    # Get the last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Check if asking for help
    if len(user_message) < 100 and "help" in user_message.lower():
        response_text = """📖 **Story Narration Tool**

Professional voice narration for stories and audiobooks.

**How to use:**
1. Paste your story or script
2. Include character names with dialogue (Name: "dialogue")
3. Add "Type: kids" for children's stories
4. The tool will assign appropriate voices

**Format Example:**
```
Title: The Magic Forest
Type: kids

Once upon a time, in a magical forest...

Rabbit: "Hello! Welcome to our forest!"

Fox: "Would you like to explore with us?"

The children were excited to begin their adventure...
```

**Features:**
- Automatic voice assignment for characters
- Kids stories: Warm, slower narration
- Adult fiction: Professional audiobook style
- Emotion detection for expressive reading
- Multiple character voices
"""
    else:
        # Parse and process the story
        script = parse_story_script(user_message)
        response_text = format_story_response(script)
    
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
    return {"status": "healthy", "service": "story_narration"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "13061"))
    uvicorn.run(app, host="0.0.0.0", port=port)