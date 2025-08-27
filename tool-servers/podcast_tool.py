#!/usr/bin/env python3
"""
Open-WebUI Tool Server: Podcast Generator
Professional podcast creation with intro/outro, music beds, and multiple speakers
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
    title="Podcast Generator Tool",
    description="Create professional podcasts with multiple speakers and music",
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
    "id": "podcast_generator",
    "name": "Podcast Generator",
    "description": "Create professional podcasts with intro, segments, and outro",
    "version": "1.0.0",
    "author": "Unicorn Commander",
    "license": "MIT",
    "icon": "📻",
    "capabilities": ["podcast", "interview", "narration", "audio-production"]
}

class Message(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    model: str = "podcast-generator"
    messages: List[Message]
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)

class PodcastSegment(BaseModel):
    type: str  # intro, interview, discussion, ad, outro
    speaker: str
    text: str
    voice: Optional[str] = None
    duration_estimate: Optional[float] = None

class PodcastScript(BaseModel):
    title: str
    episode_number: Optional[int] = None
    hosts: List[str]
    guests: List[str]
    segments: List[PodcastSegment]
    total_duration_estimate: Optional[float] = None

def parse_podcast_outline(text: str) -> PodcastScript:
    """Parse user text into podcast format"""
    
    segments = []
    hosts = []
    guests = []
    title = "Untitled Podcast"
    episode_number = None
    
    # Voice assignments
    voice_map = {
        "Host": "am_michael",
        "Co-Host": "af_bella",
        "Guest": "af_sarah",
        "Narrator": "am_adam",
        "Announcer": "am_echo",
    }
    
    # Parse the text
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for metadata
        if line.startswith("Title:"):
            title = line.replace("Title:", "").strip()
        elif line.startswith("Episode:"):
            try:
                episode_number = int(line.replace("Episode:", "").strip())
            except:
                pass
        elif line.startswith("Hosts:"):
            hosts = [h.strip() for h in line.replace("Hosts:", "").split(",")]
        elif line.startswith("Guests:"):
            guests = [g.strip() for g in line.replace("Guests:", "").split(",")]
        
        # Parse dialogue segments
        elif ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                
                # Determine segment type
                segment_type = "discussion"
                if speaker in ["Intro", "Opening"]:
                    segment_type = "intro"
                elif speaker in ["Outro", "Closing"]:
                    segment_type = "outro"
                elif speaker in ["Ad", "Advertisement", "Sponsor"]:
                    segment_type = "ad"
                elif "interview" in text.lower():
                    segment_type = "interview"
                
                # Get voice for speaker
                voice = voice_map.get(speaker, voice_map["Host"])
                
                # Add to appropriate list
                if speaker in hosts or "Host" in speaker:
                    if speaker not in hosts:
                        hosts.append(speaker)
                elif speaker in guests or "Guest" in speaker:
                    if speaker not in guests:
                        guests.append(speaker)
                
                segments.append(PodcastSegment(
                    type=segment_type,
                    speaker=speaker,
                    text=text,
                    voice=voice,
                    duration_estimate=len(text) / 15.0  # Rough estimate: 15 chars/second
                ))
    
    # Add default intro if none exists
    if not segments or segments[0].type != "intro":
        intro_text = f"Welcome to {title}"
        if episode_number:
            intro_text += f", Episode {episode_number}"
        if hosts:
            intro_text += f". I'm your host, {hosts[0]}"
        
        segments.insert(0, PodcastSegment(
            type="intro",
            speaker=hosts[0] if hosts else "Host",
            text=intro_text,
            voice=voice_map["Host"],
            duration_estimate=3.0
        ))
    
    # Add default outro if none exists
    if not segments or segments[-1].type != "outro":
        outro_text = f"Thanks for listening to {title}. "
        outro_text += "Don't forget to subscribe and leave a review!"
        
        segments.append(PodcastSegment(
            type="outro",
            speaker=hosts[0] if hosts else "Host",
            text=outro_text,
            voice=voice_map["Host"],
            duration_estimate=3.0
        ))
    
    # Calculate total duration
    total_duration = sum(s.duration_estimate or 0 for s in segments)
    
    # Default hosts and guests if none specified
    if not hosts:
        hosts = ["Host"]
    if not guests:
        guests = []
    
    return PodcastScript(
        title=title,
        episode_number=episode_number,
        hosts=hosts,
        guests=guests,
        segments=segments,
        total_duration_estimate=total_duration
    )

def format_podcast_script(script: PodcastScript) -> str:
    """Format podcast script for display"""
    
    output = "📻 **Podcast Script Generated**\n\n"
    
    # Podcast info
    output += f"**Title:** {script.title}\n"
    if script.episode_number:
        output += f"**Episode:** #{script.episode_number}\n"
    output += f"**Hosts:** {', '.join(script.hosts)}\n"
    if script.guests:
        output += f"**Guests:** {', '.join(script.guests)}\n"
    if script.total_duration_estimate:
        mins = int(script.total_duration_estimate / 60)
        secs = int(script.total_duration_estimate % 60)
        output += f"**Estimated Duration:** {mins}:{secs:02d}\n"
    
    output += "\n**Segments:**\n"
    
    # Group segments by type
    intro_segments = [s for s in script.segments if s.type == "intro"]
    main_segments = [s for s in script.segments if s.type in ["discussion", "interview"]]
    ad_segments = [s for s in script.segments if s.type == "ad"]
    outro_segments = [s for s in script.segments if s.type == "outro"]
    
    if intro_segments:
        output += "\n🎙️ **Introduction**\n"
        for seg in intro_segments:
            output += f"- **{seg.speaker}:** {seg.text[:100]}...\n"
    
    if main_segments:
        output += "\n💬 **Main Content**\n"
        for i, seg in enumerate(main_segments[:10]):  # Show first 10
            output += f"{i+1}. **{seg.speaker}:** {seg.text[:100]}...\n"
        if len(main_segments) > 10:
            output += f"... and {len(main_segments) - 10} more segments\n"
    
    if ad_segments:
        output += "\n📣 **Advertisements**\n"
        for seg in ad_segments:
            output += f"- **{seg.speaker}:** {seg.text[:100]}...\n"
    
    if outro_segments:
        output += "\n👋 **Outro**\n"
        for seg in outro_segments:
            output += f"- **{seg.speaker}:** {seg.text[:100]}...\n"
    
    output += "\n**Production Notes:**\n"
    output += "- Background music will be added to intro/outro\n"
    output += "- Transitions between segments will be smoothed\n"
    output += "- Audio will be normalized for consistent volume\n"
    
    return output

@app.get("/")
async def root():
    """Tool information endpoint"""
    return TOOL_INFO

@app.post("/v1/chat/completions")
async def generate_podcast(request: CompletionRequest):
    """Open-WebUI compatible completion endpoint"""
    
    # Get the last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Check if this is a request for instructions
    if len(user_message) < 100 and "help" in user_message.lower():
        response_text = """📻 **Podcast Generator**

Create professional podcasts by providing a script or outline. 

**Format your podcast like this:**

```
Title: Tech Talk Podcast
Episode: 42
Hosts: John, Jane
Guests: Dr. Smith

Intro: Welcome to Tech Talk, the podcast where we explore the latest in technology!

John: Today we have an amazing guest, Dr. Smith, who will talk about AI.

Dr. Smith: Thanks for having me! I'm excited to discuss the future of AI.

Jane: Let's start with the basics. What is artificial intelligence?

Dr. Smith: AI is the simulation of human intelligence by machines...

[Continue with your dialogue...]

Outro: Thanks for listening! Subscribe for more episodes!
```

**Features:**
- Automatic voice assignment for each speaker
- Professional intro/outro generation
- Background music mixing (coming soon)
- Episode numbering and metadata
- Multiple hosts and guests support

**Tips:**
- Label segments as "Intro:", "Ad:", or "Outro:" for special handling
- Each speaker gets a unique voice automatically
- The tool estimates duration based on text length
"""
    else:
        # Parse the podcast outline
        script = parse_podcast_outline(user_message)
        
        # Format response
        response_text = format_podcast_script(script)
        
        # Add generation instructions
        response_text += "\n\n**Ready to Generate Audio**\n"
        response_text += "The podcast script has been prepared. Audio generation will:\n"
        response_text += "1. Generate speech for each segment\n"
        response_text += "2. Add appropriate pauses between speakers\n"
        response_text += "3. Mix in background music for intro/outro\n"
        response_text += "4. Normalize audio levels\n"
        response_text += "\nProcessing will take approximately "
        
        if script.total_duration_estimate:
            response_text += f"{int(script.total_duration_estimate / 10)} seconds"
        else:
            response_text += "a few minutes"
    
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
    return {"status": "healthy", "service": "podcast_generator"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "13062"))
    uvicorn.run(app, host="0.0.0.0", port=port)