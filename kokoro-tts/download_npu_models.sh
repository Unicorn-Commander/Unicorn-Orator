#!/bin/bash
set -e

# Download Kokoro models for AMD Phoenix NPU
# This script downloads models from official sources before Docker build

echo "🦄 Unicorn Orator - AMD NPU Model Download"
echo "=========================================="
echo ""

# Create models directory
mkdir -p models
cd models

echo "📦 Step 1: Downloading voice embeddings..."
echo "   Source: Kokoro ONNX official release"
echo "   File: voices-v1.0.bin (26.9 MB)"
echo ""

if [ -f "voices-v1.0.bin" ] && [ $(stat -c%s "voices-v1.0.bin") -gt 1000000 ]; then
    echo "✅ voices-v1.0.bin already exists ($(stat -c%s voices-v1.0.bin | numfmt --to=iec-i)B)"
else
    # Try official kokoro-onnx GitHub release
    echo "⬇️  Downloading from GitHub..."
    if curl -L --fail --progress-bar \
        -o voices-v1.0.bin \
        'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin'; then

        # Verify size
        size=$(stat -c%s "voices-v1.0.bin")
        if [ $size -gt 1000000 ]; then
            echo "✅ voices-v1.0.bin downloaded ($(echo $size | numfmt --to=iec-i)B)"
        else
            echo "❌ Download failed - file too small ($size bytes)"
            rm -f voices-v1.0.bin
            exit 1
        fi
    else
        echo "❌ Failed to download voices-v1.0.bin"
        echo ""
        echo "Please download manually:"
        echo "  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
        echo "  Save as: $(pwd)/voices-v1.0.bin"
        exit 1
    fi
fi

echo ""
echo "📦 Step 2: Downloading standard Kokoro model..."
echo "   Source: Kokoro ONNX official release"
echo "   File: kokoro-v0_19.onnx (311 MB)"
echo ""

if [ -f "kokoro-v0_19.onnx" ] && [ $(stat -c%s "kokoro-v0_19.onnx") -gt 100000000 ]; then
    echo "✅ kokoro-v0_19.onnx already exists ($(stat -c%s kokoro-v0_19.onnx | numfmt --to=iec-i)B)"
else
    echo "⬇️  Downloading from GitHub..."
    if curl -L --fail --progress-bar \
        -o kokoro-v0_19.onnx \
        'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx'; then

        # Verify size
        size=$(stat -c%s "kokoro-v0_19.onnx")
        if [ $size -gt 100000000 ]; then
            echo "✅ kokoro-v0_19.onnx downloaded ($(echo $size | numfmt --to=iec-i)B)"
        else
            echo "❌ Download failed - file too small ($size bytes)"
            rm -f kokoro-v0_19.onnx
            exit 1
        fi
    else
        echo "❌ Failed to download kokoro-v0_19.onnx"
        exit 1
    fi
fi

echo ""
echo "📦 Step 3: NPU-optimized model (optional)..."
echo "   File: kokoro-npu-quantized-int8.onnx"
echo ""

# For now, we'll use the standard model and let NPU runtime optimize it
# Future: Add NPU-specific quantized model when available
if [ -f "kokoro-npu-quantized-int8.onnx" ]; then
    echo "✅ NPU-optimized model found ($(stat -c%s kokoro-npu-quantized-int8.onnx | numfmt --to=iec-i)B)"
else
    echo "ℹ️  NPU-optimized model not available"
    echo "   Using standard model - NPU runtime will optimize automatically"
    echo ""
    echo "   📝 To add NPU-optimized model (if you have it):"
    echo "      cp your-npu-model.onnx kokoro-npu-quantized-int8.onnx"
fi

echo ""
echo "✅ Model Download Complete!"
echo "============================"
echo ""
echo "Downloaded files:"
ls -lh *.bin *.onnx 2>/dev/null | awk '{print "  📄", $9, "-", $5}'
echo ""

total_size=$(du -sh . | awk '{print $1}')
echo "Total size: $total_size"
echo ""
echo "✨ Models are ready for Docker build!"
echo ""
echo "Next steps:"
echo "  1. docker compose -f docker-compose.yml build"
echo "  2. docker compose up -d"
echo "  3. Open http://localhost:8880"
echo ""
