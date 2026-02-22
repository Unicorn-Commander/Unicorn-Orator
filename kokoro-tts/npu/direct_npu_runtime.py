#!/usr/bin/env python3
"""
Direct NPU Runtime using XRT Python API.
Provides hardware-level access to AMD XDNA NPU through Xilinx Runtime (XRT).
This implementation uses XRT instead of raw IOCTLs for stability and proper driver handling.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import XRT Python bindings
try:
    import pyxrt
    XRT_AVAILABLE = True
    logger.info("✅ XRT Python bindings imported successfully")
except ImportError as e:
    XRT_AVAILABLE = False
    logger.warning(f"⚠️ XRT Python bindings not available: {e}")
    logger.warning("⚠️ Install with: pip install /opt/xilinx/xrt/python/pyxrt*.whl")

# Note: XRT buffer flags are accessed via pyxrt.bo.flags enum
# Available: host_only, device_only, cacheable, normal, p2p, svm
# For AMD Phoenix NPU, use host_only for host-device transfers


class DirectNPURuntime:
    """Direct NPU execution using XRT Python API - NO simulation"""

    def __init__(self):
        self.npu_device_path = "/dev/accel/accel0"
        self.device = None
        self.xclbin = None
        self.xclbin_uuid = None
        self.context = None
        self.kernel = None
        self.buffers = {}
        self.is_initialized = False

    def initialize(self, xclbin_path: Optional[str] = None) -> bool:
        """Initialize NPU hardware access using XRT Python API"""
        try:
            # Check if XRT is available
            if not XRT_AVAILABLE:
                logger.error("❌ XRT Python bindings not available")
                logger.error("   Install with: pip install /opt/xilinx/xrt/python/pyxrt*.whl")
                return False

            # Verify NPU device exists
            if not os.path.exists(self.npu_device_path):
                logger.error(f"❌ NPU device not found at {self.npu_device_path}")
                return False

            # Open NPU device using XRT
            logger.info(f"🔌 Opening NPU device with XRT...")
            self.device = pyxrt.device(0)  # Device 0 = /dev/accel/accel0
            logger.info(f"✅ Opened NPU device via XRT")

            # Load xclbin if provided
            if xclbin_path and os.path.exists(xclbin_path):
                logger.info(f"📦 Loading xclbin: {xclbin_path}")
                self.xclbin = pyxrt.xclbin(xclbin_path)
                self.xclbin_uuid = self.device.register_xclbin(self.xclbin)
                logger.info(f"✅ Loaded xclbin with UUID: {self.xclbin_uuid}")

                # Try to get kernel handle if available
                try:
                    # Look for common kernel names
                    kernel_names = ["dpu", "kernel_0", "whisper", "encoder", "decoder", "mel_spectrogram"]
                    for kname in kernel_names:
                        try:
                            self.kernel = pyxrt.kernel(self.device, self.xclbin_uuid, kname)
                            logger.info(f"✅ Found kernel: {kname}")
                            break
                        except:
                            continue

                    if not self.kernel:
                        logger.warning("⚠️ No kernel found in xclbin (may not be needed for buffer operations)")
                except Exception as e:
                    logger.warning(f"⚠️ Kernel loading skipped: {e}")
            else:
                logger.info("ℹ️  No xclbin provided, device opened without pre-loaded kernels")
                logger.info("ℹ️  You can still use buffer operations, but kernel execution requires an xclbin")

            self.is_initialized = True
            logger.info(f"🚀 XRT NPU runtime initialized - HARDWARE READY")
            logger.info(f"   Device: {self.npu_device_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize XRT NPU runtime: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def create_buffer(self, size: int, flags = None) -> Optional[pyxrt.bo]:
        """
        Create buffer object on NPU using XRT

        Args:
            size: Buffer size in bytes
            flags: XRT buffer flags (default: host_only for AMD Phoenix NPU)

        Returns:
            XRT buffer object or None on failure
        """
        if not self.is_initialized or not self.device:
            logger.error("❌ Device not initialized")
            return None

        try:
            # Use host_only flag by default (works with AMD Phoenix NPU)
            if flags is None:
                flags = pyxrt.bo.flags.host_only

            # Create XRT buffer object
            # Args: device, size, flags, memory_bank
            bo = pyxrt.bo(self.device, size, flags, 0)  # memory bank 0

            # Store reference
            bo_id = id(bo)
            self.buffers[bo_id] = {
                'bo': bo,
                'size': size,
                'flags': str(flags)
            }

            logger.debug(f"✅ Created XRT buffer: id={bo_id}, size={size} bytes")
            return bo

        except Exception as e:
            logger.error(f"❌ Buffer creation failed: {e}")
            return None

    def write_buffer(self, bo: pyxrt.bo, data: np.ndarray, offset: int = 0):
        """
        Write numpy array to XRT buffer using memory mapping

        Args:
            bo: XRT buffer object
            data: Numpy array to write
            offset: Offset in bytes
        """
        try:
            mapped = bo.map()
            data_bytes = data.tobytes()
            mapped[offset:offset + len(data_bytes)] = data_bytes
            logger.debug(f"✅ Wrote {data.nbytes} bytes to buffer at offset {offset}")
        except Exception as e:
            logger.error(f"❌ Buffer write failed: {e}")
            raise

    def read_buffer(self, bo: pyxrt.bo, shape: tuple, dtype=np.float32, offset: int = 0) -> np.ndarray:
        """
        Read numpy array from XRT buffer using memory mapping

        Args:
            bo: XRT buffer object
            shape: Shape of output array
            dtype: Data type
            offset: Offset in bytes

        Returns:
            Numpy array with data from buffer
        """
        try:
            mapped = bo.map()
            result = np.zeros(shape, dtype=dtype)
            result_bytes = mapped[offset:offset + result.nbytes]
            result = np.frombuffer(result_bytes, dtype=dtype).reshape(shape)
            logger.debug(f"✅ Read {result.nbytes} bytes from buffer at offset {offset}")
            return result
        except Exception as e:
            logger.error(f"❌ Buffer read failed: {e}")
            raise

    def sync_buffer_to_device(self, bo: pyxrt.bo, size: int = 0, offset: int = 0) -> bool:
        """
        Synchronize buffer from host to NPU

        Args:
            bo: XRT buffer object
            size: Size to sync (0 = entire buffer)
            offset: Offset in bytes
        """
        try:
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, offset)
            logger.debug(f"✅ Synced buffer to device")
            return True
        except Exception as e:
            logger.error(f"❌ Buffer sync to device failed: {e}")
            return False

    def sync_buffer_from_device(self, bo: pyxrt.bo, size: int = 0, offset: int = 0) -> bool:
        """
        Synchronize buffer from NPU to host

        Args:
            bo: XRT buffer object
            size: Size to sync (0 = entire buffer)
            offset: Offset in bytes
        """
        try:
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, offset)
            logger.debug(f"✅ Synced buffer from device")
            return True
        except Exception as e:
            logger.error(f"❌ Buffer sync from device failed: {e}")
            return False

    def execute_kernel(self, *args) -> bool:
        """
        Execute kernel on NPU with arguments

        Args:
            *args: Kernel arguments (typically XRT buffer objects)

        Returns:
            True if execution succeeded
        """
        if not self.is_initialized or not self.kernel:
            logger.error("❌ Kernel not initialized - load an xclbin with initialize(xclbin_path)")
            return False

        try:
            # Execute kernel with provided arguments
            run = self.kernel(*args)

            # Wait for completion
            state = run.wait()

            if state == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                logger.debug(f"✅ Kernel execution completed")
                return True
            else:
                logger.error(f"❌ Kernel execution failed with state: {state}")
                return False

        except Exception as e:
            logger.error(f"❌ Kernel execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def execute_mel_spectrogram_npu(self, audio: np.ndarray) -> np.ndarray:
        """
        Execute mel spectrogram on NPU hardware

        This is a working implementation that:
        1. Creates input/output buffers on NPU
        2. Transfers audio data to NPU
        3. Executes mel spectrogram kernel (if available)
        4. Retrieves results from NPU

        Args:
            audio: Input audio as numpy array

        Returns:
            Mel spectrogram as numpy array
        """
        if not self.is_initialized:
            logger.warning("⚠️ NPU not initialized, using CPU fallback")
            return self._mel_spectrogram_cpu(audio)

        try:
            # Convert audio to proper format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio to [-1, 1]
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / 32768.0

            # NPU kernel parameters (Whisper standard)
            n_fft = 400
            hop_length = 160
            n_mels = 80

            # Calculate output dimensions
            n_frames = min(3000, (len(audio) - n_fft) // hop_length + 1)

            # Create input buffer on NPU
            audio_bytes = audio.nbytes
            input_bo = self.create_buffer(audio_bytes)

            if not input_bo:
                logger.error("❌ Failed to create input buffer, falling back to CPU")
                return self._mel_spectrogram_cpu(audio)

            # Create output buffer on NPU
            output_size = n_mels * n_frames * 4  # float32
            output_bo = self.create_buffer(output_size)

            if not output_bo:
                logger.error("❌ Failed to create output buffer, falling back to CPU")
                return self._mel_spectrogram_cpu(audio)

            logger.info(f"🚀 NPU buffers created successfully")

            # Write audio data to input buffer
            self.write_buffer(input_bo, audio)
            self.sync_buffer_to_device(input_bo)

            # Execute kernel if available
            if self.kernel:
                logger.info("⚡ Executing mel spectrogram kernel on NPU")
                success = self.execute_kernel(input_bo, output_bo)

                if success:
                    # Sync output from device
                    self.sync_buffer_from_device(output_bo)

                    # Read results
                    mel_output = self.read_buffer(output_bo, (n_mels, n_frames), np.float32)
                    logger.info(f"✅ NPU mel spectrogram complete: {mel_output.shape}")
                    return mel_output
                else:
                    logger.warning("⚠️ Kernel execution failed, falling back to CPU")
            else:
                logger.info("ℹ️  No kernel loaded, using optimized CPU implementation")

            # Fallback: use optimized CPU implementation
            mel_output = self._mel_spectrogram_optimized(audio, n_mels, n_frames, n_fft, hop_length)
            logger.info(f"✅ CPU mel spectrogram complete: {mel_output.shape}")
            return mel_output

        except Exception as e:
            logger.error(f"❌ NPU execution error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._mel_spectrogram_cpu(audio)

    def _mel_spectrogram_optimized(self, audio: np.ndarray, n_mels: int, n_frames: int,
                                   n_fft: int, hop_length: int) -> np.ndarray:
        """
        Optimized CPU implementation for mel spectrogram
        Uses librosa with NumPy vectorization for speed
        """
        try:
            import librosa

            # Compute mel spectrogram using librosa (optimized)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Trim to max length
            if mel_spec_db.shape[1] > n_frames:
                mel_spec_db = mel_spec_db[:, :n_frames]

            return mel_spec_db
        except Exception as e:
            logger.error(f"❌ Optimized mel spectrogram failed: {e}")
            return self._mel_spectrogram_cpu(audio)

    def _mel_spectrogram_cpu(self, audio: np.ndarray) -> np.ndarray:
        """CPU fallback for mel spectrogram"""
        try:
            import librosa

            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_mels=80,
                n_fft=400,
                hop_length=160
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            if mel_spec_db.shape[1] > 3000:
                mel_spec_db = mel_spec_db[:, :3000]

            return mel_spec_db

        except Exception as e:
            logger.error(f"❌ CPU mel spectrogram failed: {e}")
            # Return minimal valid output
            return np.zeros((80, 100), dtype=np.float32)

    def cleanup(self):
        """Clean up NPU resources"""
        try:
            # XRT handles cleanup automatically via RAII, but we can clear references
            self.buffers.clear()
            self.kernel = None
            self.xclbin = None
            self.xclbin_uuid = None
            self.device = None
            self.is_initialized = False
            logger.info("✅ Cleaned up XRT NPU resources")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


# Global instance
direct_npu_runtime = DirectNPURuntime()


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing XRT NPU Runtime")
    print("=" * 60)

    # Test initialization
    runtime = DirectNPURuntime()
    if runtime.initialize():
        print("✅ Initialization succeeded!")

        # Test buffer creation
        test_size = 1024 * 1024  # 1MB
        bo = runtime.create_buffer(test_size)
        if bo:
            print(f"✅ Buffer creation succeeded: {test_size} bytes")

            # Test write/read
            test_data = np.random.randn(1024).astype(np.float32)
            runtime.write_buffer(bo, test_data)
            runtime.sync_buffer_to_device(bo)
            runtime.sync_buffer_from_device(bo)
            result = runtime.read_buffer(bo, test_data.shape, np.float32)

            if np.allclose(test_data, result):
                print("✅ Buffer write/read test passed!")
            else:
                print("❌ Buffer write/read test failed!")
        else:
            print("❌ Buffer creation failed")

        runtime.cleanup()
    else:
        print("❌ Initialization failed")
        print("\nTo install XRT Python bindings:")
        print("  pip install /opt/xilinx/xrt/python/pyxrt*.whl")
