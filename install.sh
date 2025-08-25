#!/bin/bash

# Unicorn Orator Installation Script
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║           🦄 Unicorn Orator Installer 🦄            ║${NC}"
echo -e "${PURPLE}║       Professional AI Speech Processing Platform     ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed${NC}"

# Check GPU (optional)
echo ""
echo "Checking for GPU acceleration..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    GPU_AVAILABLE="cuda"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
elif [ -d "/dev/dri" ] && [ -e "/dev/dri/renderD128" ]; then
    echo -e "${BLUE}✓ Intel GPU detected${NC}"
    INTEL_GPU_AVAILABLE="true"
else
    echo -e "${YELLOW}⚠ No GPU detected - will use CPU mode${NC}"
    GPU_AVAILABLE="cpu"
fi

# Create .env file
if [ ! -f .env ]; then
    echo ""
    echo "Creating configuration file..."
    cp .env.template .env
    
    # Configure based on detected hardware
    if [ "$GPU_AVAILABLE" = "cuda" ]; then
        sed -i 's/WHISPERX_DEVICE=cpu/WHISPERX_DEVICE=cuda/' .env
        sed -i 's/WHISPERX_COMPUTE_TYPE=int8/WHISPERX_COMPUTE_TYPE=float16/' .env
        sed -i 's/KOKORO_DEVICE=CPU/KOKORO_DEVICE=CUDA/' .env
        echo -e "${GREEN}✓ Configured for NVIDIA GPU acceleration${NC}"
    elif [ "$INTEL_GPU_AVAILABLE" = "true" ]; then
        sed -i 's/KOKORO_DEVICE=CPU/KOKORO_DEVICE=IGPU/' .env
        echo -e "${BLUE}✓ Configured Kokoro for Intel GPU acceleration${NC}"
    fi
    
    # Ask for domain configuration
    echo ""
    echo -e "${BLUE}Remote Access Configuration${NC}"
    echo "Enter domain/IP for remote access (or press Enter for localhost):"
    read -p "> " EXTERNAL_HOST
    
    if [ -n "$EXTERNAL_HOST" ] && [ "$EXTERNAL_HOST" != "localhost" ]; then
        sed -i "s/EXTERNAL_HOST=localhost/EXTERNAL_HOST=$EXTERNAL_HOST/" .env
        
        read -p "Use HTTPS? (y/N): " USE_HTTPS
        if [[ "$USE_HTTPS" =~ ^[Yy]$ ]]; then
            sed -i "s/EXTERNAL_PROTOCOL=http/EXTERNAL_PROTOCOL=https/" .env
        fi
        echo -e "${GREEN}✓ Remote access configured${NC}"
    fi
    
    # Ask for HuggingFace token (optional)
    echo ""
    echo -e "${BLUE}Optional: HuggingFace Token${NC}"
    echo "Enter your HF token for speaker diarization (or press Enter to skip):"
    echo "Get a token at: https://huggingface.co/settings/tokens"
    read -p "> " HF_TOKEN
    
    if [ -n "$HF_TOKEN" ]; then
        sed -i "s/HF_TOKEN=/HF_TOKEN=$HF_TOKEN/" .env
        echo -e "${GREEN}✓ HuggingFace token configured${NC}"
    fi
else
    echo -e "${GREEN}✓ Configuration file exists${NC}"
fi

# Create required directories
echo ""
echo "Creating required directories..."
mkdir -p volumes/whisperx_models
mkdir rp volumes/kokoro_models
echo -e "${GREEN}✓ Directories created${NC}"

# Build services
echo ""
echo -e "${BLUE}Building services...${NC}"
docker-compose build

# Start services
echo ""
echo -e "${BLUE}Starting Unicorn Orator...${NC}"
docker-compose up -d

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         🎉 Installation Complete! 🎉                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services are starting up..."
echo ""
echo -e "${PURPLE}Access Points:${NC}"

# Load .env to show correct URLs
source .env
if [ "$EXTERNAL_HOST" != "localhost" ] && [ -n "$EXTERNAL_HOST" ]; then
    echo "  🎨 Web Interface: ${BLUE}${EXTERNAL_PROTOCOL:-http}://${EXTERNAL_HOST}:8880${NC}"
    echo "  🎙️ STT API: ${BLUE}${EXTERNAL_PROTOCOL:-http}://${EXTERNAL_HOST}:9000${NC}"
    echo "  🔊 TTS API: ${BLUE}${EXTERNAL_PROTOCOL:-http}://${EXTERNAL_HOST}:8880${NC}"
else
    echo "  🎨 Web Interface: ${BLUE}http://localhost:8880${NC}"
    echo "  🎙️ STT API: ${BLUE}http://localhost:9000${NC}"
    echo "  🔊 TTS API: ${BLUE}http://localhost:8880${NC}"
fi

echo ""
echo -e "${YELLOW}Note: First startup may take a few minutes while models download${NC}"
echo ""
echo "Useful commands:"
echo "  View logs:    ${GREEN}docker-compose logs -f${NC}"
echo "  Stop services: ${GREEN}docker-compose down${NC}"
echo "  Test STT:     ${GREEN}./test-stt.sh${NC}"
echo "  Test TTS:     ${GREEN}./test-tts.sh${NC}"
echo ""
echo -e "${PURPLE}🦄 Enjoy Unicorn Orator! 🦄${NC}"