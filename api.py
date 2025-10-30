"""
Unicorn-Orator: Multi-Platform Text-to-Speech API

Automatically detects and uses the best available backend:
- XDNA2 for Strix Point NPU
- XDNA1 for Phoenix/Hawk Point NPU
- CPU for fallback support
"""

from fastapi import FastAPI, HTTPException
import logging
import sys
import os

# Add runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'runtime'))

from runtime.platform_detector import get_platform, get_platform_info, Platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unicorn-Orator",
    description="Multi-platform Text-to-Speech Service with NPU acceleration",
    version="2.0.0"
)

# Detect platform and load appropriate backend
platform = get_platform()
platform_info = get_platform_info()

logger.info(f"Initializing Unicorn-Orator on {platform.value} backend")
logger.info(f"Platform info: {platform_info}")

# Import platform-specific server
if platform == Platform.XDNA2:
    logger.info("Loading XDNA2 backend...")
    # TODO: Import xdna2 implementation when ready
    # For now, fall back to XDNA1
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xdna1'))
    from xdna1.server import app as backend_app
    backend_type = "XDNA1 (XDNA2 pending implementation)"
elif platform == Platform.XDNA1:
    logger.info("Loading XDNA1 backend...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xdna1'))
    from xdna1.server import app as backend_app
    backend_type = "XDNA1"
else:  # CPU
    logger.info("Loading CPU backend...")
    # TODO: Import CPU implementation when ready
    # For now, fall back to XDNA1 (which supports CPU mode)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xdna1'))
    from xdna1.server import app as backend_app
    backend_type = "CPU"

# Mount backend routes
app.mount("/", backend_app)


@app.get("/platform")
async def get_platform_endpoint():
    """Get current platform information"""
    return {
        "service": "Unicorn-Orator",
        "version": "2.0.0",
        "platform": platform_info,
        "backend": backend_type
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Unicorn-Orator",
        "description": "Multi-platform Text-to-Speech Service",
        "version": "2.0.0",
        "platform": platform.value,
        "backend": backend_type,
        "endpoints": {
            "/v1/audio/speech": "POST - Generate speech",
            "/voices": "GET - List available voices",
            "/health": "GET - Health check",
            "/platform": "GET - Platform information"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
