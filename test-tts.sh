#!/bin/bash

# Unicorn Orator TTS Test Script
echo "🔊 Testing Text-to-Speech Service..."

# Check if service is running
if ! curl -s http://localhost:8880/health > /dev/null 2>&1; then
    echo "❌ TTS service is not responding. Is it running?"
    echo "Start with: docker-compose up -d kokoro-tts"
    exit 1
fi

echo "✓ TTS service is running"

# Test text-to-speech
echo ""
echo "Generating speech..."
curl -X POST http://localhost:8880/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Welcome to Unicorn Orator. This is a test of the text to speech system. The quality is exceptional!",
        "voice": "af",
        "speed": 1.0
    }' \
    --output test-output.wav 2>/dev/null

if [ -f "test-output.wav" ] && [ -s "test-output.wav" ]; then
    echo "✓ Audio generated: test-output.wav"
    echo ""
    
    # Get file info
    file_size=$(ls -lh test-output.wav | awk '{print $5}')
    echo "📊 File size: $file_size"
    
    # Try to play if command exists
    if command -v aplay &> /dev/null; then
        echo "Playing audio..."
        aplay test-output.wav 2>/dev/null
    elif command -v afplay &> /dev/null; then
        echo "Playing audio..."
        afplay test-output.wav
    else
        echo "ℹ️ Audio player not found. File saved as test-output.wav"
    fi
else
    echo "❌ Failed to generate audio"
    exit 1
fi

echo ""
echo "✅ TTS test complete!"