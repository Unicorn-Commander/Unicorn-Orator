#!/bin/bash

# Unicorn Orator v2 - Universal Hardware Support Installer
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
INSTALL_DIR="$(pwd)"
CONFIG_FILE=".orator.yml"
HARDWARE_REPORT="hardware_detection.json"

echo -e "${PURPLE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║     🦄 Unicorn Orator v2.0 - Universal Installer 🦄      ║${NC}"
echo -e "${PURPLE}║         Intelligent Hardware-Optimized Speech AI          ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        echo "Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed${NC}"
        echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check Python (for hardware detection)
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}⚠️ Python 3 not found - hardware detection limited${NC}"
        PYTHON_AVAILABLE=false
    else
        PYTHON_AVAILABLE=true
    fi
    
    echo -e "${GREEN}✅ Prerequisites check complete${NC}"
    echo ""
}

# Function to detect hardware
detect_hardware() {
    echo -e "${CYAN}🔍 Detecting available hardware...${NC}"
    echo ""
    
    if [ "$PYTHON_AVAILABLE" = true ]; then
        # Run Python hardware detection
        python3 hardware-detect/detect.py
        
        if [ -f "$HARDWARE_REPORT" ]; then
            # Parse recommendations from JSON
            WHISPERX_BACKEND=$(python3 -c "import json; print(json.load(open('$HARDWARE_REPORT'))['recommendations']['whisperx']['backend'])")
            KOKORO_BACKEND=$(python3 -c "import json; print(json.load(open('$HARDWARE_REPORT'))['recommendations']['kokoro']['backend'])")
            WHISPERX_VARIANT=$(python3 -c "import json; print(json.load(open('$HARDWARE_REPORT'))['recommendations']['whisperx'].get('variant', 'full'))")
        fi
    else
        # Fallback detection using shell commands
        echo -e "${YELLOW}Using basic hardware detection...${NC}"
        
        # Check for NVIDIA GPU
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
            echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
            WHISPERX_BACKEND="cuda"
            KOKORO_BACKEND="cuda"
        # Check for Intel GPU
        elif [ -d "/dev/dri" ] && [ -e "/dev/dri/renderD128" ]; then
            echo -e "${BLUE}✅ Intel GPU detected${NC}"
            WHISPERX_BACKEND="igpu"
            KOKORO_BACKEND="igpu"
        else
            echo -e "${YELLOW}⚠️ No GPU detected - using CPU${NC}"
            WHISPERX_BACKEND="cpu"
            KOKORO_BACKEND="cpu"
        fi
        WHISPERX_VARIANT="full"
    fi
    
    echo ""
    echo -e "${GREEN}Recommended Configuration:${NC}"
    echo -e "  WhisperX: ${CYAN}$WHISPERX_BACKEND${NC} (variant: $WHISPERX_VARIANT)"
    echo -e "  Kokoro: ${CYAN}$KOKORO_BACKEND${NC}"
    echo ""
}

# Function to prompt for configuration
configure_installation() {
    echo -e "${BLUE}Configuration Options${NC}"
    echo ""
    
    # Ask if user wants to use recommended configuration
    read -p "Use recommended configuration? (Y/n): " USE_RECOMMENDED
    if [[ ! "$USE_RECOMMENDED" =~ ^[Nn]$ ]]; then
        echo -e "${GREEN}✅ Using recommended configuration${NC}"
    else
        # Manual configuration
        echo ""
        echo "Available WhisperX backends:"
        echo "  1) cpu - Universal CPU support"
        echo "  2) cuda - NVIDIA GPU acceleration"
        echo "  3) npu - AMD NPU acceleration (Ryzen AI)"
        echo "  4) igpu - Intel iGPU acceleration"
        read -p "Select WhisperX backend (1-4): " WHISPERX_CHOICE
        
        case $WHISPERX_CHOICE in
            1) WHISPERX_BACKEND="cpu" ;;
            2) WHISPERX_BACKEND="cuda" ;;
            3) WHISPERX_BACKEND="npu" ;;
            4) WHISPERX_BACKEND="igpu" ;;
            *) WHISPERX_BACKEND="cpu" ;;
        esac
        
        # Ask about diarization for NPU/iGPU
        if [[ "$WHISPERX_BACKEND" == "npu" ]] || [[ "$WHISPERX_BACKEND" == "igpu" ]]; then
            read -p "Enable speaker diarization? (Y/n): " ENABLE_DIARIZATION
            if [[ "$ENABLE_DIARIZATION" =~ ^[Nn]$ ]]; then
                WHISPERX_VARIANT="lite"
            else
                WHISPERX_VARIANT="full"
            fi
        fi
        
        echo ""
        echo "Available Kokoro backends:"
        echo "  1) cpu - Universal CPU support"
        echo "  2) cuda - NVIDIA GPU acceleration"
        echo "  3) npu - AMD NPU acceleration"
        echo "  4) igpu - Intel iGPU acceleration"
        read -p "Select Kokoro backend (1-4): " KOKORO_CHOICE
        
        case $KOKORO_CHOICE in
            1) KOKORO_BACKEND="cpu" ;;
            2) KOKORO_BACKEND="cuda" ;;
            3) KOKORO_BACKEND="npu" ;;
            4) KOKORO_BACKEND="igpu" ;;
            *) KOKORO_BACKEND="cpu" ;;
        esac
    fi
    
    # Additional options
    echo ""
    read -p "Enable automatic fallback to CPU on errors? (Y/n): " ENABLE_FALLBACK
    if [[ ! "$ENABLE_FALLBACK" =~ ^[Nn]$ ]]; then
        FALLBACK_TO_CPU=true
    else
        FALLBACK_TO_CPU=false
    fi
    
    # HuggingFace token for diarization
    if [[ "$WHISPERX_VARIANT" == "full" ]]; then
        echo ""
        echo -e "${BLUE}Optional: HuggingFace Token for speaker diarization${NC}"
        echo "Get a token at: https://huggingface.co/settings/tokens"
        read -p "HF Token (press Enter to skip): " HF_TOKEN
    fi
    
    # Domain configuration
    echo ""
    echo -e "${BLUE}Remote Access Configuration${NC}"
    read -p "Domain/IP for remote access (press Enter for localhost): " EXTERNAL_HOST
    EXTERNAL_HOST=${EXTERNAL_HOST:-localhost}
    
    if [[ "$EXTERNAL_HOST" != "localhost" ]]; then
        read -p "Use HTTPS? (y/N): " USE_HTTPS
        if [[ "$USE_HTTPS" =~ ^[Yy]$ ]]; then
            EXTERNAL_PROTOCOL="https"
        else
            EXTERNAL_PROTOCOL="http"
        fi
    else
        EXTERNAL_PROTOCOL="http"
    fi
}

# Function to generate configuration files
generate_configs() {
    echo ""
    echo -e "${BLUE}Generating configuration files...${NC}"
    
    # Create .env file
    cat > .env << EOF
# Unicorn Orator v2 Configuration
# Generated on $(date)

# Hardware Configuration
WHISPERX_BACKEND=$WHISPERX_BACKEND
WHISPERX_VARIANT=$WHISPERX_VARIANT
KOKORO_BACKEND=$KOKORO_BACKEND

# Feature Flags
WHISPERX_ENABLE_DIARIZATION=$([ "$WHISPERX_VARIANT" == "full" ] && echo "true" || echo "false")
FALLBACK_TO_CPU=$FALLBACK_TO_CPU

# Model Configuration
WHISPER_MODEL=${WHISPER_MODEL:-base}
KOKORO_VOICE=${KOKORO_VOICE:-af}

# API Tokens
HF_TOKEN=$HF_TOKEN

# Network Configuration
EXTERNAL_HOST=$EXTERNAL_HOST
EXTERNAL_PROTOCOL=$EXTERNAL_PROTOCOL

# Performance Tuning (auto-configured based on hardware)
$(if [[ "$WHISPERX_BACKEND" == "npu" ]]; then
    echo "NPU_BATCH_SIZE=8"
elif [[ "$WHISPERX_BACKEND" == "igpu" ]]; then
    echo "IGPU_BATCH_SIZE=4"
elif [[ "$WHISPERX_BACKEND" == "cuda" ]]; then
    echo "CUDA_BATCH_SIZE=16"
else
    echo "CPU_BATCH_SIZE=1"
fi)
EOF
    
    # Create YAML configuration
    cat > $CONFIG_FILE << EOF
# Unicorn Orator Configuration
version: 2.0

hardware:
  detection: auto
  whisperx:
    backend: $WHISPERX_BACKEND
    variant: $WHISPERX_VARIANT
  kokoro:
    backend: $KOKORO_BACKEND

preferences:
  fallback_to_cpu: $FALLBACK_TO_CPU
  prefer_npu: $([ "$WHISPERX_BACKEND" == "npu" ] && echo "true" || echo "false")

network:
  external_host: $EXTERNAL_HOST
  external_protocol: $EXTERNAL_PROTOCOL

performance:
  batch_size: auto
  compute_precision: $([ "$WHISPERX_BACKEND" == "cuda" ] && echo "fp16" || echo "int8")
EOF
    
    echo -e "${GREEN}✅ Configuration files generated${NC}"
}

# Function to select and build appropriate Docker images
build_services() {
    echo ""
    echo -e "${BLUE}Building optimized Docker images...${NC}"
    
    # Determine which docker-compose file to use
    COMPOSE_FILE="docker-compose.yml"
    
    # Create override file for hardware-specific configuration
    cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  whisperx:
    build: 
      context: ./whisperx/$WHISPERX_BACKEND
      dockerfile: Dockerfile$([ "$WHISPERX_VARIANT" == "lite" ] && echo ".lite" || echo "")
    environment:
      - BACKEND=$WHISPERX_BACKEND
      - ENABLE_DIARIZATION=$([ "$WHISPERX_VARIANT" == "full" ] && echo "true" || echo "false")
$(if [[ "$WHISPERX_BACKEND" == "npu" ]]; then
    echo "    devices:"
    echo "      - /dev/npu0:/dev/npu0"
elif [[ "$WHISPERX_BACKEND" == "igpu" ]] || [[ "$WHISPERX_BACKEND" == "cuda" ]]; then
    echo "    devices:"
    echo "      - /dev/dri:/dev/dri"
fi)

  kokoro-tts:
    build:
      context: ./kokoro-tts/$KOKORO_BACKEND
      dockerfile: Dockerfile
    environment:
      - BACKEND=$KOKORO_BACKEND
$(if [[ "$KOKORO_BACKEND" == "npu" ]]; then
    echo "    devices:"
    echo "      - /dev/npu0:/dev/npu0"
elif [[ "$KOKORO_BACKEND" == "igpu" ]] || [[ "$KOKORO_BACKEND" == "cuda" ]]; then
    echo "    devices:"
    echo "      - /dev/dri:/dev/dri"
fi)
EOF
    
    # Build the services
    echo -e "${CYAN}Building WhisperX ($WHISPERX_BACKEND-$WHISPERX_VARIANT)...${NC}"
    docker-compose build whisperx
    
    echo -e "${CYAN}Building Kokoro ($KOKORO_BACKEND)...${NC}"
    docker-compose build kokoro-tts
    
    echo -e "${GREEN}✅ Docker images built successfully${NC}"
}

# Function to start services
start_services() {
    echo ""
    echo -e "${BLUE}Starting Unicorn Orator services...${NC}"
    
    docker-compose up -d
    
    echo -e "${GREEN}✅ Services started${NC}"
}

# Function to display final information
show_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           🎉 Installation Complete! 🎉                   ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${PURPLE}Configuration Summary:${NC}"
    echo "  WhisperX: $WHISPERX_BACKEND ($WHISPERX_VARIANT)"
    echo "  Kokoro: $KOKORO_BACKEND"
    echo ""
    echo -e "${PURPLE}Access Points:${NC}"
    if [[ "$EXTERNAL_HOST" != "localhost" ]]; then
        echo "  🎨 Web Interface: ${BLUE}${EXTERNAL_PROTOCOL}://${EXTERNAL_HOST}:8880${NC}"
        echo "  🎙️ STT API: ${BLUE}${EXTERNAL_PROTOCOL}://${EXTERNAL_HOST}:9000${NC}"
        echo "  🔊 TTS API: ${BLUE}${EXTERNAL_PROTOCOL}://${EXTERNAL_HOST}:8880${NC}"
    else
        echo "  🎨 Web Interface: ${BLUE}http://localhost:8880${NC}"
        echo "  🎙️ STT API: ${BLUE}http://localhost:9000${NC}"
        echo "  🔊 TTS API: ${BLUE}http://localhost:8880${NC}"
    fi
    echo ""
    echo -e "${YELLOW}Note: First startup may take a few minutes while models download${NC}"
    echo ""
    echo "Useful commands:"
    echo "  View logs: ${GREEN}docker-compose logs -f${NC}"
    echo "  Stop services: ${GREEN}docker-compose down${NC}"
    echo "  Test STT: ${GREEN}./test-stt.sh${NC}"
    echo "  Test TTS: ${GREEN}./test-tts.sh${NC}"
    echo "  Reconfigure: ${GREEN}./install-v2.sh --reconfigure${NC}"
    echo ""
    
    # Show performance expectations
    echo -e "${CYAN}Expected Performance:${NC}"
    case $WHISPERX_BACKEND in
        npu)
            echo "  WhisperX: ~5x realtime (AMD NPU optimized)"
            ;;
        igpu)
            echo "  WhisperX: ~4x realtime (Intel iGPU optimized)"
            ;;
        cuda)
            echo "  WhisperX: ~10x realtime (NVIDIA GPU accelerated)"
            ;;
        cpu)
            echo "  WhisperX: ~2x realtime (CPU optimized)"
            ;;
    esac
    
    case $KOKORO_BACKEND in
        npu)
            echo "  Kokoro: ~3x realtime (AMD NPU optimized)"
            ;;
        igpu)
            echo "  Kokoro: ~3x realtime (Intel iGPU optimized)"
            ;;
        cuda)
            echo "  Kokoro: ~5x realtime (NVIDIA GPU accelerated)"
            ;;
        cpu)
            echo "  Kokoro: ~1x realtime (CPU baseline)"
            ;;
    esac
    
    echo ""
    echo -e "${PURPLE}🦄 Enjoy Unicorn Orator v2.0! 🦄${NC}"
}

# Main installation flow
main() {
    # Parse arguments
    if [[ "$1" == "--reconfigure" ]]; then
        echo -e "${BLUE}Reconfiguration mode...${NC}"
        RECONFIGURE=true
    fi
    
    # Run installation steps
    check_prerequisites
    detect_hardware
    configure_installation
    generate_configs
    
    if [[ "$RECONFIGURE" != true ]]; then
        build_services
        start_services
    fi
    
    show_summary
}

# Run main function
main "$@"