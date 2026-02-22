#!/bin/bash
set -e

echo "🦄 Installing Unicorn Orator (TTS Service)"
echo "============================================"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "kokoro-tts" ]; then
    echo -e "${RED}❌ Error: kokoro-tts directory not found${NC}"
    echo "Please run this script from the Unicorn-Orator directory"
    exit 1
fi

echo ""
echo "📦 Step 1: Installing unicorn-npu-core library"
echo "-----------------------------------------------"

# Check if unicorn-npu-core exists
CORE_PATH="/home/ucadmin/UC-1/unicorn-npu-core"
if [ ! -d "$CORE_PATH" ]; then
    echo -e "${RED}❌ unicorn-npu-core not found at: $CORE_PATH${NC}"
    echo "Please ensure unicorn-npu-core is cloned/created first"
    exit 1
fi

# Install unicorn-npu-core in development mode
cd "$CORE_PATH"
pip install -e . --break-system-packages
cd - > /dev/null

echo -e "${GREEN}✅ unicorn-npu-core installed${NC}"

echo ""
echo "📦 Step 2: Setting up NPU on host system"
echo "-----------------------------------------------"

# Run NPU host setup (via core library)
python3 -c "from unicorn_npu.scripts.install_host import main; main()" || true

echo ""
echo "📦 Step 3: Installing Orator dependencies"
echo "-----------------------------------------------"

# Determine which requirements file to use
if [ -f "kokoro-tts/requirements.npu.txt" ]; then
    REQUIREMENTS_FILE="kokoro-tts/requirements.npu.txt"
    echo "Using NPU requirements"
else
    REQUIREMENTS_FILE="kokoro-tts/requirements.txt"
    echo "Using default requirements"
fi

# Install Orator requirements
cd kokoro-tts
pip install -r $(basename "$REQUIREMENTS_FILE") --break-system-packages
cd ..

echo -e "${GREEN}✅ Dependencies installed${NC}"

echo ""
echo "📦 Step 4: Downloading Kokoro models (optional)"
echo "-----------------------------------------------"

# Check if models directory exists
if [ ! -d "kokoro-tts/models" ]; then
    mkdir -p kokoro-tts/models
    echo "Created models directory"
fi

# Check if models are already downloaded
if [ -f "kokoro-tts/models/kokoro-v0_19.onnx" ]; then
    echo -e "${GREEN}✅ Kokoro model already exists${NC}"
else
    read -p "Download Kokoro TTS model? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading Kokoro model..."
        cd kokoro-tts
        python3 -c "from kokoro_onnx import Kokoro; k = Kokoro('kokoro-v0_19', lang='a')" || echo "Model download initiated"
        cd ..
    else
        echo "Skipping model download"
    fi
fi

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Orator installation complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Log out and log back in (for group changes)"
echo "2. Set NPU to performance mode:"
echo "   python3 -c \"from unicorn_npu import NPUDevice; npu = NPUDevice(); npu.set_power_mode('performance')\""
echo "3. Start the service:"
echo "   cd kokoro-tts && python3 server.py"
echo ""
