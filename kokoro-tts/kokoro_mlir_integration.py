#!/usr/bin/env python3
"""
Kokoro MLIR-AIE NPU Integration

This module integrates the Kokoro TTS model with MLIR-AIE NPU acceleration.
"""

import sys
import os
import logging
import numpy as np

# Add the kokoro-onnx package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kokoro-onnx", "src"))

from kokoro_onnx import Kokoro
from kokoro_mlir_npu import KokoroNPUAcceleratorMLIR
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class KokoroMLIRNPUIntegration:
    """Complete integration of Kokoro TTS with MLIR-AIE NPU acceleration"""
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Initialize Kokoro MLIR-AIE NPU integration
        
        Args:
            model_path: Path to Kokoro ONNX model
            voices_path: Path to voices file
        """
        self.model_path = model_path
        self.voices_path = voices_path
        
        # Initialize MLIR-AIE NPU accelerator
        self.mlir_accelerator = KokoroNPUAcceleratorMLIR()
        self.acceleration_enabled = self.mlir_accelerator.acceleration_enabled
        
        # Initialize standard Kokoro for comparison and voice handling
        self.kokoro_standard = Kokoro(model_path, voices_path)
        
        # Create NPU-accelerated ONNX session
        self._create_npu_session()
        
        if self.acceleration_enabled:
            logger.info("🎉 Kokoro MLIR-AIE NPU integration ready")
        else:
            logger.warning("⚠️ MLIR-AIE NPU not available, using CPU fallback")
    
    def _create_npu_session(self):
        """Create ONNX Runtime session optimized for NPU operations"""
        try:
            # Create session with CPU provider (we'll intercept matrix ops for NPU)
            providers = ['CPUExecutionProvider']
            self.npu_session = ort.InferenceSession(self.model_path, providers=providers)
            
            if self.acceleration_enabled:
                # Wrap session for NPU acceleration
                self.npu_session = self._wrap_session_for_mlir_npu(self.npu_session)
                logger.info("🚀 Created MLIR-AIE NPU-accelerated session")
            else:
                logger.info("Created CPU-only session")
                
        except Exception as e:
            logger.error(f"Failed to create NPU session: {e}")
            raise
    
    def _wrap_session_for_mlir_npu(self, session):
        """Wrap ONNX session to use MLIR-AIE NPU for matrix operations"""
        original_run = session.run
        
        def mlir_npu_accelerated_run(output_names, input_feed, run_options=None):
            """MLIR-AIE NPU-accelerated inference run"""
            return self.mlir_accelerator.accelerated_inference(
                lambda: original_run(output_names, input_feed, run_options),
                input_feed
            )
        
        # Replace the run method
        session.run = mlir_npu_accelerated_run
        return session
    
    def _handle_npu_optimized_model(self, session, tokens, style, speed):
        """Handle models that don't need tokens input"""
        input_names = [inp.name for inp in session.get_inputs()]
        
        if 'tokens' not in input_names:
            # Model doesn't need tokens, just use style and speed
            input_feed = {
                'style': style,
                'speed': speed
            }
        else:
            # Standard model with all inputs
            input_feed = {
                'tokens': tokens,
                'style': style, 
                'speed': speed
            }
        
        return session.run(None, input_feed)
    
    def create_audio(self, text: str, voice: str, speed: float = 1.0, 
                    lang: str = "en-us") -> tuple[np.ndarray, int]:
        """
        Create audio using MLIR-AIE NPU acceleration
        
        Args:
            text: Text to synthesize
            voice: Voice name
            speed: Speaking speed
            lang: Language code
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            if self.acceleration_enabled:
                logger.info(f"🚀 Generating audio with MLIR-AIE NPU acceleration")
                logger.info(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                logger.info(f"Voice: {voice}, Speed: {speed}")
                
                # Use the standard Kokoro API but with NPU-accelerated inference
                return self._create_audio_npu_accelerated(text, voice, speed, lang)
            else:
                logger.info("Using CPU fallback for audio generation")
                return self.kokoro_standard.create(text, voice, speed, lang)
                
        except Exception as e:
            logger.error(f"MLIR-AIE NPU audio generation failed: {e}")
            logger.info("Falling back to standard CPU generation")
            return self.kokoro_standard.create(text, voice, speed, lang)
    
    def _create_audio_npu_accelerated(self, text: str, voice: str, 
                                    speed: float, lang: str) -> tuple[np.ndarray, int]:
        """Create audio using NPU-accelerated inference"""
        import time
        
        # Get voice style
        if isinstance(voice, str):
            voice_style = self.kokoro_standard.get_voice_style(voice)
        else:
            voice_style = voice
        
        start_time = time.time()
        
        # Convert text to phonemes
        if text.strip():
            phonemes = self.kokoro_standard.tokenizer.phonemize(text, lang)
        else:
            raise ValueError("Empty text provided")
        
        # Tokenize phonemes
        tokens = np.array(self.kokoro_standard.tokenizer.tokenize(phonemes), dtype=np.int64)
        
        # Prepare inputs for NPU inference
        voice_for_length = voice_style[len(tokens)]
        tokens_padded = [[0, *tokens, 0]]
        
        input_feed = {
            'tokens': tokens_padded,
            'style': voice_for_length,
            'speed': np.ones(1, dtype=np.float32) * speed
        }
        
        # Run NPU-accelerated inference
        result = self._handle_npu_optimized_model(self.npu_session, tokens_padded, voice_for_length, np.ones(1, dtype=np.float32) * speed)
        audio = result[0]
        
        # Debug: Check audio type and shape
        logger.info(f"   Raw audio result type: {type(audio)}")
        if hasattr(audio, 'shape'):
            logger.info(f"   Raw audio shape: {audio.shape}")
        else:
            logger.info(f"   Raw audio value: {audio}")
            
        # Ensure audio is an array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        if audio.ndim == 0:
            logger.warning("Audio result is scalar, this might indicate an issue")
            audio = np.array([audio])  # Convert scalar to 1D array
        
        generation_time = time.time() - start_time
        sample_rate = 24000  # Kokoro sample rate
        
        logger.info(f"✅ MLIR-AIE NPU audio generation completed")
        logger.info(f"   Generation time: {generation_time:.3f}s")
        logger.info(f"   Audio length: {len(audio)/sample_rate:.2f}s ({len(audio)} samples)")
        audio_duration = len(audio) / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"   Real-time factor: {rtf:.3f}")
        
        return audio, sample_rate
    
    def get_voices(self) -> list[str]:
        """Get available voices"""
        return self.kokoro_standard.get_voices()
    
    def get_acceleration_status(self) -> dict:
        """Get detailed acceleration status"""
        base_status = self.mlir_accelerator.get_acceleration_status()
        return {
            **base_status,
            "model_path": self.model_path,
            "voices_available": len(self.get_voices()),
            "session_ready": hasattr(self, 'npu_session') and self.npu_session is not None
        }


def create_kokoro_mlir_npu_integration(model_path: str, voices_path: str):
    """
    Create Kokoro MLIR-AIE NPU integration
    
    Args:
        model_path: Path to Kokoro ONNX model
        voices_path: Path to voices file
        
    Returns:
        KokoroMLIRNPUIntegration instance
    """
    integration = KokoroMLIRNPUIntegration(model_path, voices_path)
    
    # Print status
    status = integration.get_acceleration_status()
    print("🔍 Kokoro MLIR-AIE NPU Integration Status:")
    for key, value in status.items():
        if key != 'kernel_info':  # Don't print nested dict
            print(f"  {key}: {value}")
    
    return integration


if __name__ == "__main__":
    # Test the MLIR-AIE NPU integration
    print("🧪 Testing Kokoro MLIR-AIE NPU Integration...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "kokoro-v1.0.onnx")
    voices_path = os.path.join(base_dir, "voices-v1.0.bin")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(voices_path):
        print(f"❌ Voices file not found: {voices_path}")
        sys.exit(1)
    
    try:
        # Create integration
        integration = create_kokoro_mlir_npu_integration(model_path, voices_path)
        
        # Test audio generation
        test_text = "Hello! This is a test of MLIR-AIE NPU acceleration for Kokoro TTS."
        test_voice = "af_bella"
        
        print(f"\n🎤 Generating audio...")
        print(f"Text: '{test_text}'")
        print(f"Voice: {test_voice}")
        
        audio, sample_rate = integration.create_audio(test_text, test_voice)
        
        print(f"\n✅ Audio generation successful!")
        print(f"   Audio length: {len(audio)/sample_rate:.2f}s")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Samples: {len(audio)}")
        
        # Test multiple voices
        voices = integration.get_voices()
        print(f"\n🎭 Available voices: {len(voices)}")
        print(f"   Sample voices: {voices[:5]}")
        
        print(f"\n🎉 Kokoro MLIR-AIE NPU integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)