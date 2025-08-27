#!/usr/bin/env python3
"""
Open-WebUI Tool Server: Commercial Voiceover
Professional voice synthesis for ads, commercials, and promotional content
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
    title="Commercial Voiceover Tool",
    description="Professional voiceover for commercials and advertisements",
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
    "id": "commercial_voiceover",
    "name": "Commercial Voiceover",
    "description": "Professional voice synthesis for ads and commercials",
    "version": "1.0.0",
    "author": "Unicorn Commander",
    "license": "MIT",
    "icon": "📢",
    "capabilities": ["commercials", "radio-ads", "promos", "announcements"]
}

class Message(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    model: str = "commercial-voiceover"
    messages: List[Message]
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)

class CommercialSegment(BaseModel):
    type: str  # hook, product, benefit, cta, disclaimer
    text: str
    voice: str
    energy: str  # low, medium, high, urgent
    speed: float  # 0.8 to 1.5

class CommercialScript(BaseModel):
    product_name: str
    commercial_type: str  # radio, tv, web, podcast
    duration: str  # 15s, 30s, 60s
    style: str  # hard-sell, soft-sell, conversational, dramatic
    segments: List[CommercialSegment]

def parse_commercial_script(text: str) -> CommercialScript:
    """Parse commercial script into voiceover format"""
    
    segments = []
    product_name = "Product"
    commercial_type = "radio"
    duration = "30s"
    style = "conversational"
    
    # Voice selection based on style
    voice_map = {
        "hard-sell": "am_echo",  # Strong, commanding voice
        "soft-sell": "af_bella",  # Warm, friendly voice
        "conversational": "am_michael",  # Natural, relatable voice
        "dramatic": "am_adam",  # Deep, dramatic voice
        "energetic": "af_sarah",  # Upbeat, enthusiastic voice
        "professional": "bf_emma",  # Clear, professional voice
    }
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for metadata
        if line.startswith("Product:"):
            product_name = line.replace("Product:", "").strip()
        elif line.startswith("Type:"):
            commercial_type = line.replace("Type:", "").strip().lower()
        elif line.startswith("Duration:"):
            duration = line.replace("Duration:", "").strip()
        elif line.startswith("Style:"):
            style = line.replace("Style:", "").strip().lower()
        
        # Parse segments with special markers
        elif line.startswith("[") and "]" in line:
            # Format: [TYPE] text or [TYPE:VOICE] text
            bracket_end = line.index("]")
            bracket_content = line[1:bracket_end]
            text = line[bracket_end+1:].strip()
            
            if ":" in bracket_content:
                segment_type, voice_hint = bracket_content.split(":", 1)
                voice = voice_map.get(voice_hint.lower(), voice_map.get(style, "am_michael"))
            else:
                segment_type = bracket_content.lower()
                voice = voice_map.get(style, "am_michael")
            
            # Determine energy and speed based on segment type
            if segment_type == "hook":
                energy = "high"
                speed = 1.1
            elif segment_type == "cta":  # Call to action
                energy = "urgent"
                speed = 1.2
            elif segment_type == "disclaimer":
                energy = "low"
                speed = 1.3  # Fast for disclaimers
            elif segment_type == "benefit":
                energy = "medium"
                speed = 1.0
            else:
                energy = "medium"
                speed = 1.0
            
            segments.append(CommercialSegment(
                type=segment_type,
                text=text,
                voice=voice,
                energy=energy,
                speed=speed
            ))
        
        else:
            # Regular line - determine type from content
            segment_type = "product"
            energy = "medium"
            speed = 1.0
            
            # Check for common patterns
            if any(word in line.lower() for word in ["limited time", "now", "today", "hurry"]):
                segment_type = "cta"
                energy = "urgent"
                speed = 1.1
            elif any(word in line.lower() for word in ["save", "free", "discount", "offer"]):
                segment_type = "benefit"
                energy = "high"
                speed = 1.0
            elif line.startswith("*") or "terms" in line.lower():
                segment_type = "disclaimer"
                energy = "low"
                speed = 1.3
            elif len(segments) == 0:
                segment_type = "hook"
                energy = "high"
                speed = 1.0
            
            voice = voice_map.get(style, "am_michael")
            
            segments.append(CommercialSegment(
                type=segment_type,
                text=line,
                voice=voice,
                energy=energy,
                speed=speed
            ))
    
    # Add default segments if missing
    if not segments:
        segments.append(CommercialSegment(
            type="hook",
            text=f"Introducing {product_name}!",
            voice=voice_map.get(style, "am_michael"),
            energy="high",
            speed=1.0
        ))
    
    return CommercialScript(
        product_name=product_name,
        commercial_type=commercial_type,
        duration=duration,
        style=style,
        segments=segments
    )

def format_commercial_response(script: CommercialScript) -> str:
    """Format commercial script for display"""
    
    output = f"📢 **Commercial Voiceover Ready**\n\n"
    output += f"**Product:** {script.product_name}\n"
    output += f"**Type:** {script.commercial_type.title()}\n"
    output += f"**Duration:** {script.duration}\n"
    output += f"**Style:** {script.style.title()}\n\n"
    
    output += f"**Script Breakdown:**\n"
    
    # Group segments by type
    hooks = [s for s in script.segments if s.type == "hook"]
    benefits = [s for s in script.segments if s.type == "benefit"]
    ctas = [s for s in script.segments if s.type == "cta"]
    disclaimers = [s for s in script.segments if s.type == "disclaimer"]
    
    if hooks:
        output += "\n🎯 **Hook:**\n"
        for seg in hooks:
            output += f"- {seg.text}\n"
    
    if benefits:
        output += "\n✨ **Benefits:**\n"
        for seg in benefits[:3]:
            output += f"- {seg.text}\n"
    
    if ctas:
        output += "\n📣 **Call to Action:**\n"
        for seg in ctas:
            output += f"- {seg.text}\n"
    
    if disclaimers:
        output += "\n📝 **Disclaimers:**\n"
        for seg in disclaimers:
            output += f"- {seg.text[:50]}...\n"
    
    output += "\n**Voice Direction:**\n"
    if script.style == "hard-sell":
        output += "- Urgent, commanding delivery\n"
        output += "- High energy throughout\n"
        output += "- Strong emphasis on benefits\n"
    elif script.style == "soft-sell":
        output += "- Warm, friendly tone\n"
        output += "- Building trust and connection\n"
        output += "- Gentle persuasion\n"
    elif script.style == "conversational":
        output += "- Natural, authentic delivery\n"
        output += "- Like talking to a friend\n"
        output += "- Relatable and genuine\n"
    else:
        output += "- Professional delivery\n"
        output += "- Clear and articulate\n"
        output += "- Appropriate pacing\n"
    
    return output

@app.get("/")
async def root():
    """Tool information endpoint"""
    return TOOL_INFO

@app.post("/v1/chat/completions")
async def process_commercial(request: CompletionRequest):
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
        response_text = """📢 **Commercial Voiceover Tool**

Professional voiceover for commercials, ads, and promos.

**Script Format:**
```
Product: Amazing Product
Type: radio
Duration: 30s
Style: energetic

[HOOK] Are you tired of ordinary products?

[BENEFIT] Amazing Product changes everything with revolutionary technology!

[CTA] Call now and save 50%!

[DISCLAIMER] *Terms and conditions apply
```

**Segment Types:**
- `[HOOK]` - Opening attention grabber
- `[BENEFIT]` - Product benefits
- `[CTA]` - Call to action
- `[DISCLAIMER]` - Legal/terms

**Styles:**
- **hard-sell** - Urgent, commanding
- **soft-sell** - Warm, friendly
- **conversational** - Natural, relatable
- **dramatic** - Deep, impactful
- **energetic** - Upbeat, exciting

**Tips:**
- Keep it concise for impact
- Use energy markers for emphasis
- Include clear call to action
- Add disclaimers at the end
"""
    else:
        # Parse and process the commercial
        script = parse_commercial_script(user_message)
        response_text = format_commercial_response(script)
    
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
    return {"status": "healthy", "service": "commercial_voiceover"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "13063"))
    uvicorn.run(app, host="0.0.0.0", port=port)