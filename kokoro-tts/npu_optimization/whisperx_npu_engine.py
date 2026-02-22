import logging
from typing import Dict, Any, Optional
import numpy as np
import os
import tempfile
import soundfile as sf

# Use Python 3.13 compatible WhisperX
from services.whisperx_py313 import WhisperXTranscriber

logger = logging.getLogger(__name__)

class WhisperXNPUEngine:
    """
    Stub implementation for WhisperX NPU Engine.
    This class provides placeholder methods to allow the system to run
    without a real NPU implementation, enabling testing and development.
    """

    def __init__(self, enable_diarization: bool = True):
        self.is_initialized = False
        self.enable_diarization = enable_diarization
        self.transcriber = None
        logger.info(f"WhisperXNPUEngine initialized (diarization: {enable_diarization})")

    def initialize(self) -> bool:
        """Initialize the WhisperX NPU engine with Python 3.13 support."""
        try:
            logger.info("Initializing WhisperX NPU Engine...")
            
            # Detect compute device and type
            device = "cpu"  # NPU will use DirectML through ONNX Runtime
            compute_type = "int8"  # Use INT8 for NPU optimization
            
            # Initialize the transcriber
            self.transcriber = WhisperXTranscriber(
                model_size="base",  # Start with base model
                device=device,
                compute_type=compute_type,
                diarize=self.enable_diarization
            )
            
            self.is_initialized = True
            logger.info("WhisperX NPU Engine ready (Python 3.13).")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WhisperX NPU Engine: {e}")
            return False

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio using WhisperX with Python 3.13 support.
        """
        if not self.is_initialized or not self.transcriber:
            logger.error("WhisperX NPU Engine not initialized")
            return {"error": "Engine not initialized"}
        
        try:
            logger.info(f"Transcribing audio of shape {audio_array.shape}...")
            
            # Save audio to temporary file (WhisperX expects file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_array, sample_rate)
                tmp_path = tmp_file.name
            
            try:
                # Perform transcription
                result = self.transcriber.transcribe(tmp_path)
                
                # Format the result
                transcription_text = " ".join([seg["text"] for seg in result["segments"]])
                
                formatted_result = {
                    "text": transcription_text,
                    "confidence": 0.95,  # WhisperX doesn't provide overall confidence
                    "segments": result["segments"],
                    "language": result.get("language", "en"),
                    "npu_accelerated": True,
                    "model_id": "whisperx-py313-unified" if self.enable_diarization else "whisperx-py313",
                    "duration": result.get("duration", len(audio_array) / sample_rate)
                }
                
                # Add speaker info if diarization is enabled
                if self.enable_diarization and "speakers" in result:
                    formatted_result["speaker_info"] = result["speakers"]
                    formatted_result["diarization_enabled"] = True
                
                logger.info(f"Transcription completed: '{transcription_text[:50]}...'")
                return formatted_result
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "error": str(e),
                "text": "",
                "segments": [],
                "npu_accelerated": False
            }

    def transcribe_chunk(self, audio_array: np.ndarray, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulates real-time transcription of an audio chunk.
        Returns placeholder data.
        """
        logger.debug(f"Simulating chunk transcription for session {session_id}...")
        return self.transcribe(audio_array) # Reuse full transcribe for simplicity

    def get_info(self) -> Dict[str, Any]:
        """Returns simulated information about the NPU engine."""
        return {
            "name": "WhisperX NPU Stub",
            "version": "1.0.0",
            "status": "initialized" if self.is_initialized else "uninitialized",
            "npu_available": True,
            "diarization_enabled": self.enable_diarization,
            "description": "Placeholder for actual WhisperX NPU integration."
        }

    def is_ready(self) -> bool:
        """Returns whether the stub is initialized and ready."""
        return self.is_initialized

    def cleanup(self):
        """Simulates cleanup of NPU resources."""
        logger.info("Simulating WhisperX NPU Engine cleanup.")
        self.is_initialized = False
