"""
Platform detection for multi-backend STT support.

Detects available hardware and selects optimal backend:
- XDNA2 (Strix Point) - Highest priority for new NPU
- XDNA1 (Phoenix/Hawk Point) - Legacy NPU support
- CPU - Fallback for systems without NPU
"""

import os
import logging
import subprocess
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Available compute platforms"""
    XDNA2 = "xdna2"  # Strix Point NPU
    XDNA1 = "xdna1"  # Phoenix/Hawk Point NPU
    CPU = "cpu"      # CPU fallback


class PlatformDetector:
    """Detects and selects optimal compute platform"""

    def __init__(self):
        self._detected_platform: Optional[Platform] = None

    def detect(self) -> Platform:
        """
        Detect available platform in priority order:
        1. XDNA2 (if available)
        2. XDNA1 (if available)
        3. CPU (always available)
        """
        if self._detected_platform:
            return self._detected_platform

        # Check for manual override
        override = os.environ.get("NPU_PLATFORM")
        if override:
            try:
                platform = Platform(override.lower())
                logger.info(f"Using manually specified platform: {platform.value}")
                self._detected_platform = platform
                return platform
            except ValueError:
                logger.warning(f"Invalid NPU_PLATFORM override: {override}")

        # Auto-detect XDNA2 (Strix Point)
        if self._has_xdna2():
            logger.info("Detected XDNA2 NPU (Strix Point)")
            self._detected_platform = Platform.XDNA2
            return Platform.XDNA2

        # Auto-detect XDNA1 (Phoenix/Hawk Point)
        if self._has_xdna1():
            logger.info("Detected XDNA1 NPU (Phoenix/Hawk Point)")
            self._detected_platform = Platform.XDNA1
            return Platform.XDNA1

        # Fallback to CPU
        logger.info("No NPU detected, using CPU backend")
        self._detected_platform = Platform.CPU
        return Platform.CPU

    def _has_xdna2(self) -> bool:
        """Check for XDNA2 NPU availability"""
        try:
            # Check for XDNA2 driver
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Look for AMD Strix Point NPU device ID
            # Device ID: 1502:1502 (Strix Point NPU)
            if "1502:1502" in result.stdout:
                return True

            # Check for XDNA2 runtime
            xdna2_runtime = os.path.exists("/opt/xilinx/xrt/bin/xbutil")
            if xdna2_runtime:
                return True

        except Exception as e:
            logger.debug(f"XDNA2 detection failed: {e}")

        return False

    def _has_xdna1(self) -> bool:
        """Check for XDNA1 NPU availability"""
        try:
            # Check for XDNA1 driver
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Look for AMD Phoenix/Hawk Point NPU device IDs
            # Phoenix: 1502:17f0
            # Hawk Point: Similar device ID range
            xdna1_ids = ["1502:17f0", "1502:17f1", "1502:17f2"]
            if any(dev_id in result.stdout for dev_id in xdna1_ids):
                return True

        except Exception as e:
            logger.debug(f"XDNA1 detection failed: {e}")

        return False

    def get_backend_path(self) -> str:
        """Get the path to backend implementation"""
        platform = self.detect()
        return platform.value

    def get_info(self) -> dict:
        """Get platform information"""
        platform = self.detect()
        return {
            "platform": platform.value,
            "backend_path": self.get_backend_path(),
            "has_npu": platform in [Platform.XDNA1, Platform.XDNA2],
            "npu_generation": "XDNA2" if platform == Platform.XDNA2 else "XDNA1" if platform == Platform.XDNA1 else None
        }


# Global detector instance
_detector = PlatformDetector()


def get_platform() -> Platform:
    """Get detected platform"""
    return _detector.detect()


def get_platform_info() -> dict:
    """Get platform information"""
    return _detector.get_info()
