#!/bin/bash

# Build script for Intel iGPU optimized Unicorn Orator
set -e

echo "🦄 Building Unicorn Orator for Intel iGPU..."

# Build the Docker image
cd kokoro-tts
docker build -f Dockerfile.intel-igpu -t unicorn-orator:intel-igpu .

echo "✅ Build complete!"
echo ""
echo "🧪 Testing the image..."

# Test the image
docker run --rm --name unicorn-orator-test \
  --device /dev/dri:/dev/dri \
  --group-add 44 --group-add 993 \
  -p 8886:8880 \
  -e DEVICE=IGPU \
  -e OPENVINO_PRECISION=FP16 \
  -d unicorn-orator:intel-igpu

echo "⏳ Waiting for service to start..."
sleep 15

# Check health
HEALTH=$(curl -s http://localhost:8886/health | jq -r .backend 2>/dev/null || echo "failed")

if [[ "$HEALTH" == *"OpenVINO"* ]]; then
  echo "✅ Intel iGPU acceleration confirmed: $HEALTH"
else
  echo "❌ Intel iGPU acceleration failed: $HEALTH"
fi

# Cleanup
docker stop unicorn-orator-test

echo ""
echo "🚀 Ready to push to DockerHub:"
echo "  docker tag unicorn-orator:intel-igpu magicunicorn/unicorn-orator:intel-igpu-v2.0"
echo "  docker push magicunicorn/unicorn-orator:intel-igpu-v2.0"
echo ""
echo "📋 DockerHub repository: https://hub.docker.com/r/magicunicorn/unicorn-orator"