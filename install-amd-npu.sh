#!/bin/bash
set -e

# Unicorn Orator - Automated Installation for AMD Phoenix NPU
# Compatible with: AMD Ryzen 9 8945HS, Ryzen AI 300 Series
# Requires: Ubuntu 24.10+, NPU drivers installed

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║        🦄  Unicorn Orator - AMD NPU Installation  🦄             ║"
echo "║                                                                  ║"
echo "║         Professional AI Voice Synthesis Platform                ║"
echo "║         Optimized for AMD Phoenix NPU (XDNA1)                   ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will:"
echo "  ✓ Download Kokoro TTS models (~320 MB)"
echo "  ✓ Build Docker image with NPU support"
echo "  ✓ Start Unicorn Orator service on port 8880"
echo ""
echo "Prerequisites:"
echo "  • AMD Ryzen AI CPU with Phoenix NPU"
echo "  • NPU drivers installed (/dev/accel/accel0)"
echo "  • Docker and Docker Compose installed"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."
echo ""

# Check if NPU device exists
if [ ! -e "/dev/accel/accel0" ]; then
    echo "❌ ERROR: NPU device not found at /dev/accel/accel0"
    echo ""
    echo "Please install AMD NPU drivers first:"
    echo "  • XRT (Xilinx Runtime)"
    echo "  • NPU firmware"
    echo ""
    echo "See: https://github.com/amd/xdna-driver"
    exit 1
else
    echo "✅ NPU device found: /dev/accel/accel0"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker not found"
    echo ""
    echo "Please install Docker first:"
    echo "  curl -fsSL https://get.docker.com | sh"
    echo "  sudo usermod -aG docker $USER"
    exit 1
else
    echo "✅ Docker: $(docker --version | cut -d' ' -f3)"
fi

# Check Docker Compose
if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
    echo "❌ ERROR: Docker Compose not found"
    echo ""
    echo "Docker Compose is required (included in Docker 20.10+)"
    exit 1
else
    echo "✅ Docker Compose: $(docker compose version --short)"
fi

# Check user is in docker group
if ! groups | grep -q docker; then
    echo "⚠️  WARNING: User not in docker group"
    echo ""
    echo "Run this to add yourself to docker group:"
    echo "  sudo usermod -aG docker $USER"
    echo "  newgrp docker"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "✅ All prerequisites met!"
echo ""

# Navigate to kokoro-tts directory
cd kokoro-tts

# Download models
echo "================================================"
echo "📥 Step 1: Downloading Models"
echo "================================================"
echo ""

chmod +x download_npu_models.sh
./download_npu_models.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model download failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "🔨 Step 2: Building Docker Image"
echo "================================================"
echo ""
echo "This may take 5-10 minutes..."
echo ""

# Build with NPU support
if docker compose -f ../docker-compose.yml build; then
    echo ""
    echo "✅ Docker image built successfully!"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "🚀 Step 3: Starting Service"
echo "================================================"
echo ""

# Start service
if docker compose -f ../docker-compose.yml up -d; then
    echo ""
    echo "✅ Service started successfully!"
else
    echo ""
    echo "❌ Service start failed!"
    exit 1
fi

# Wait for service to be ready
echo ""
echo "⏳ Waiting for service to initialize..."
sleep 5

# Check health
for i in {1..12}; do
    if curl -s http://localhost:8880/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi
    echo "   Attempt $i/12..."
    sleep 5
done

echo ""
echo "================================================"
echo "🎉 Installation Complete!"
echo "================================================"
echo ""

# Show status
health_response=$(curl -s http://localhost:8880/health 2>/dev/null || echo '{}')

echo "Service Status:"
echo "  • WebGUI: http://localhost:8880"
echo "  • API: http://localhost:8880/v1/audio/speech"
echo "  • Health: http://localhost:8880/health"
echo ""

# Check if NPU is enabled
if echo "$health_response" | grep -q '"npu_enabled":true'; then
    echo "🎯 NPU Acceleration: ENABLED ✅"
    echo "   Expected performance: 13x realtime"
else
    echo "⚠️  NPU Acceleration: Not Active"
    echo "   Running in CPU mode (slower)"
    echo ""
    echo "   Troubleshooting:"
    echo "   • Check NPU device permissions: ls -la /dev/accel/accel0"
    echo "   • Verify XRT installation: xrt-smi examine"
    echo "   • Check container logs: docker compose logs unicorn-orator"
fi

echo ""
echo "Available voices: 50+ professional voices"
echo "Supported languages: English (multiple accents)"
echo ""
echo "Quick Test:"
echo "  curl -X POST http://localhost:8880/v1/audio/speech \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\":\"Hello from Unicorn Orator!\",\"voice\":\"af_heart\"}' \\"
echo "    -o test.wav"
echo ""
echo "View logs:"
echo "  docker compose -f ../docker-compose.yml logs -f"
echo ""
echo "Stop service:"
echo "  docker compose -f ../docker-compose.yml down"
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🦄 Enjoy professional AI voice synthesis with AMD NPU! 🦄       ║"
echo "║                                                                  ║"
echo "║  Magic Unicorn Unconventional Technology & Stuff Inc            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
