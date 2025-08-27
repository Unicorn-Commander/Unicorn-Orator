#!/bin/bash

# Unicorn Orator - Multi-Hardware Build Script
# Automatically detects hardware and builds optimized containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
HARDWARE=""
FORCE_BUILD=""
VERBOSE=""
TAG_SUFFIX=""

usage() {
    echo "Unicorn Orator Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --hardware HARDWARE   Force specific hardware (auto, cpu, igpu, npu)"
    echo "  -f, --force              Force rebuild even if image exists"
    echo "  -v, --verbose            Enable verbose output"
    echo "  -t, --tag TAG            Additional tag suffix"
    echo "  --help                   Show this help message"
    echo ""
    echo "Hardware options:"
    echo "  auto     - Auto-detect best available hardware (default)"
    echo "  cpu      - CPU-only build (universal compatibility)"
    echo "  igpu     - Intel iGPU optimized (OpenVINO)"
    echo "  npu      - AMD NPU optimized (custom runtime)"
    echo ""
    echo "Examples:"
    echo "  $0                       # Auto-detect and build"
    echo "  $0 --hardware igpu       # Force Intel iGPU build"
    echo "  $0 --hardware cpu --tag production"
    echo ""
}

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--hardware)
            HARDWARE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_BUILD="1"
            shift
            ;;
        -v|--verbose)
            VERBOSE="1"
            shift
            ;;
        -t|--tag)
            TAG_SUFFIX="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Hardware detection functions
detect_intel_gpu() {
    if [ -d "/dev/dri" ] && [ -n "$(find /dev/dri -name 'renderD*' 2>/dev/null)" ]; then
        if lspci 2>/dev/null | grep -E "(Intel.*Graphics|Intel.*VGA)" > /dev/null; then
            return 0
        fi
    fi
    return 1
}

detect_amd_npu() {
    # Check for AMD Ryzen AI NPU
    if lspci 2>/dev/null | grep -i "amd.*npu\|ryzen.*ai" > /dev/null; then
        return 0
    fi
    # Check for specific AMD processors with NPU
    if grep -E "(7940HS|7840HS|8945HS|8845HS)" /proc/cpuinfo > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

auto_detect_hardware() {
    log "Auto-detecting optimal hardware configuration..."
    
    if detect_amd_npu; then
        log_success "AMD NPU detected (Ryzen AI processor)"
        echo "npu"
    elif detect_intel_gpu; then
        log_success "Intel iGPU detected"
        echo "igpu"
    else
        log_warning "No specialized hardware detected, falling back to CPU"
        echo "cpu"
    fi
}

# Determine hardware target
if [ -z "$HARDWARE" ] || [ "$HARDWARE" = "auto" ]; then
    HARDWARE=$(auto_detect_hardware)
else
    log "Using specified hardware: $HARDWARE"
fi

# Validate hardware choice
case $HARDWARE in
    cpu|igpu|npu)
        ;;
    *)
        log_error "Invalid hardware option: $HARDWARE"
        log_error "Valid options: cpu, igpu, npu, auto"
        exit 1
        ;;
esac

# Set build parameters based on hardware
case $HARDWARE in
    cpu)
        DOCKERFILE="Dockerfile.cpu"
        IMAGE_NAME="unicorn-orator:cpu"
        BUILD_CONTEXT="kokoro-tts"
        ;;
    igpu)
        DOCKERFILE="Dockerfile.intel-igpu"
        IMAGE_NAME="unicorn-orator:intel-igpu"
        BUILD_CONTEXT="kokoro-tts"
        ;;
    npu)
        DOCKERFILE="Dockerfile.amd-npu"
        IMAGE_NAME="unicorn-orator:amd-npu"
        BUILD_CONTEXT="kokoro-tts"
        ;;
esac

# Add tag suffix if specified
if [ -n "$TAG_SUFFIX" ]; then
    IMAGE_NAME="${IMAGE_NAME}-${TAG_SUFFIX}"
fi

# Check if image already exists
if [ -z "$FORCE_BUILD" ] && docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    log_warning "Image $IMAGE_NAME already exists. Use --force to rebuild."
    log "To run the existing image:"
    echo "  docker run -d --name unicorn-orator -p 8880:8880 $IMAGE_NAME"
    exit 0
fi

# Pre-build checks
log "Performing pre-build checks..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$BUILD_CONTEXT/$DOCKERFILE" ]; then
    log_error "Dockerfile not found: $BUILD_CONTEXT/$DOCKERFILE"
    exit 1
fi

# Check if models directory exists and has content
if [ ! -d "$BUILD_CONTEXT/models" ] || [ -z "$(ls -A $BUILD_CONTEXT/models)" ]; then
    log_warning "Models directory is empty. Downloading required models..."
    cd "$BUILD_CONTEXT"
    ./download_models.sh
    cd ..
fi

# Start building
log "Building Unicorn Orator for $HARDWARE..."
log "Image: $IMAGE_NAME"
log "Dockerfile: $DOCKERFILE"

BUILD_ARGS=""
if [ "$VERBOSE" = "1" ]; then
    BUILD_ARGS="--progress=plain"
fi

# Build the Docker image
log "Starting Docker build..."
if docker build $BUILD_ARGS -f "$BUILD_CONTEXT/$DOCKERFILE" -t "$IMAGE_NAME" "$BUILD_CONTEXT"; then
    log_success "Build completed successfully!"
    
    # Tag as latest for this hardware type
    LATEST_TAG="unicorn-orator:${HARDWARE}-latest"
    docker tag "$IMAGE_NAME" "$LATEST_TAG"
    log_success "Tagged as: $LATEST_TAG"
    
    # Show image info
    echo ""
    log "Image Information:"
    docker images | grep "unicorn-orator" | head -5
    
    echo ""
    log_success "Unicorn Orator ($HARDWARE) build complete!"
    echo ""
    echo "To run the service:"
    echo ""
    
    case $HARDWARE in
        igpu)
            echo "  docker run -d --name unicorn-orator \\"
            echo "    -p 8880:8880 \\"
            echo "    -e DEVICE=IGPU \\"
            echo "    --device=/dev/dri:/dev/dri \\"
            echo "    --group-add=44 --group-add=993 \\"
            echo "    --security-opt=seccomp:unconfined \\"
            echo "    $IMAGE_NAME"
            ;;
        npu)
            echo "  docker run -d --name unicorn-orator \\"
            echo "    -p 8880:8880 \\"
            echo "    -e DEVICE=NPU \\"
            echo "    $IMAGE_NAME"
            ;;
        cpu)
            echo "  docker run -d --name unicorn-orator \\"
            echo "    -p 8880:8880 \\"
            echo "    -e DEVICE=CPU \\"
            echo "    $IMAGE_NAME"
            ;;
    esac
    
    echo ""
    echo "Or use docker-compose:"
    echo "  docker-compose -f kokoro-tts/docker-compose.yml up unicorn-orator-$HARDWARE -d"
    echo ""
    echo "Access the service at: http://localhost:8880"
    
else
    log_error "Build failed!"
    exit 1
fi