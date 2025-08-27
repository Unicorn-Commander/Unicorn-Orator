#!/bin/bash

# Download Kokoro TTS models
echo "Downloading Kokoro TTS models..."

# Create models directory if it doesn't exist
mkdir -p models

cd models

# Download the Kokoro v0.19 ONNX model
if [ ! -f "kokoro-v0_19.onnx" ]; then
    echo "Downloading Kokoro v0.19 model..."
    wget -q --show-progress https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
else
    echo "Kokoro model already exists"
fi

# Download voice embeddings
if [ ! -f "voices-v1.0.bin" ]; then
    echo "Downloading voice embeddings..."
    wget -q --show-progress https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin
else
    echo "Voice embeddings already exist"
fi

echo "Model download complete!"
ls -lah