# 🦄 Unicorn Speech Suite - Application Split Proposal

## Current Situation
Unicorn Orator currently combines both STT (WhisperX) and TTS (Kokoro) services. As we add more features like transcription UI, diarization visualization, and NPU support, the question arises: should we split these into separate, focused applications?

## Proposed Architecture

### 🎙️ **Unicorn Scribe** - Professional Transcription Platform
**Focus**: Speech-to-Text, Transcription, Diarization

#### Features:
- **Web Interface**:
  - Real-time audio recording and transcription
  - File upload (supports MP3, WAV, M4A, etc.)
  - Live waveform visualization
  - Speaker diarization timeline
  - Speaker labeling and identification
  - Word-level timestamp display
  - Search within transcripts
  
- **Export Options**:
  - SRT/VTT subtitles
  - JSON with timestamps
  - Plain text
  - Word/PDF documents
  - Speaker-separated transcripts

- **Advanced Features**:
  - Batch processing
  - Language detection
  - Translation capabilities
  - Custom vocabulary
  - API for integrations

- **Hardware Support**:
  - AMD NPU (with/without diarization)
  - Intel iGPU variants
  - NVIDIA GPU acceleration
  - CPU fallback

### 🔊 **Unicorn Orator** - Professional Voice Synthesis Platform
**Focus**: Text-to-Speech, Voice Generation

#### Features (Keep Current):
- Beautiful voice selection interface
- Real-time preview
- Speed/pitch controls
- Multiple voice options
- Download capabilities

#### Enhancements:
- AMD NPU support for Kokoro
- Voice cloning (future)
- Emotion controls (future)
- SSML support

## Shared Infrastructure

### Backend Services (Docker)
Both applications would share the same backend containers:
```yaml
# Shared services
services:
  whisperx:       # Used by Unicorn Scribe
    image: unicorn-whisperx:latest
    
  kokoro-tts:     # Used by Unicorn Orator
    image: unicorn-kokoro:latest
```

### Benefits of Sharing:
1. **Resource Efficiency**: One set of models loaded in memory
2. **Unified Deployment**: Single docker-compose for both apps
3. **Consistent APIs**: Same endpoints for both applications
4. **Hardware Detection**: Share the same detection system

## Implementation Plan

### Phase 1: Create Unicorn Scribe
1. Create new repository: `Unicorn-Commander/Unicorn-Scribe`
2. Design transcription-focused UI
3. Implement diarization visualization
4. Add file upload and processing
5. Create export functionality

### Phase 2: Refactor Unicorn Orator
1. Keep TTS-focused interface
2. Remove STT API documentation (move to Scribe)
3. Enhance voice synthesis features
4. Add NPU support for Kokoro

### Phase 3: Create Unified Deployment
1. Create `Unicorn-Commander/Unicorn-Speech-Suite`
2. Combine both apps with shared backend
3. Single installer for complete suite
4. Option to install individually

## User Experience

### For End Users:
- **Unicorn Scribe**: "I need to transcribe audio/video"
- **Unicorn Orator**: "I need to generate speech"
- **Speech Suite**: "I need both capabilities"

### URLs:
- Scribe: `http://localhost:9001` (new port)
- Orator: `http://localhost:8880` (keep current)
- Shared API: `http://localhost:9000` (WhisperX)

## Advantages of Splitting

1. **Focused UX**: Each app optimized for its purpose
2. **Cleaner Interfaces**: No feature bloat
3. **Independent Development**: Teams can work separately
4. **Better Branding**: Clear purpose for each app
5. **Modular Deployment**: Users can choose what they need

## Disadvantages of Splitting

1. **More Repositories**: Additional maintenance
2. **Potential Duplication**: Some shared code
3. **Complex Deployment**: For users who want both

## Alternative: Enhanced Unicorn Orator

Keep everything in one app but with better organization:
- Tab 1: Voice Synthesis (TTS)
- Tab 2: Transcription (STT)
- Tab 3: Settings
- Tab 4: API Documentation

## Recommendation

**Split the applications** for better focus and user experience, but maintain shared backend services. Create:

1. **Unicorn Scribe** - New transcription-focused app
2. **Unicorn Orator** - Keep as voice synthesis app
3. **Unicorn Speech Suite** - Combined deployment option

This gives users flexibility while maintaining clean, focused interfaces.

## Next Steps

1. Decision on split vs. unified approach
2. If split:
   - Create Unicorn Scribe repository
   - Design transcription UI
   - Extract WhisperX interface code
3. If unified:
   - Design tabbed interface
   - Integrate transcription features
   - Update navigation

What do you think? Should we create Unicorn Scribe as a separate, transcription-focused application?