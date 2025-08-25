#!/bin/bash

# Unicorn Orator STT Test Script
echo "🎙️ Testing Speech-to-Text Service..."

# Check if service is running
if ! curl -s http://localhost:9000/health > /dev/null 2>&1; then
    echo "❌ STT service is not responding. Is it running?"
    echo "Start with: docker-compose up -d whisperx"
    exit 1
fi

echo "✓ STT service is running"

# Create test audio if it doesn't exist
if [ ! -f "test-audio.wav" ]; then
    echo "Creating test audio file..."
    # Use text-to-speech to create test audio
    if curl -s http://localhost:8880/health > /dev/null 2>&1; then
        curl -X POST http://localhost:8880/v1/audio/speech \
            -H "Content-Type: application/json" \
            -d '{"text": "Hello, this is a test of Unicorn Orator speech to text capabilities.", "voice": "af"}' \
            --output test-audio.wav 2>/dev/null
        echo "✓ Test audio created using TTS"
    else
        echo "⚠️ No test audio available and TTS is not running"
        echo "Please provide a test-audio.wav file"
        exit 1
    fi
fi

# Test transcription
echo ""
echo "Testing transcription..."
response=$(curl -s -X POST http://localhost:9000/v1/audio/transcriptions \
    -F "file=@test-audio.wav" \
    -F "response_format=json")

if [ -z "$response" ]; then
    echo "❌ No response from STT service"
    exit 1
fi

# Parse and display result
echo ""
echo "📝 Transcription Result:"
echo "$response" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('text', 'No text found'))" 2>/dev/null || echo "$response"

echo ""
echo "✅ STT test complete!"