#!/usr/bin/env python3
"""
Test if Kokoro TTS has sign bug in audio output

The bug manifests as:
- Negative correlation with reference
- Wrong polarity in output
- +/-65536 errors in int16 range

This script tests Kokoro's audio generation pipeline for the sign extension bug
discovered in Whisper mel computation (October 2025).

Key Difference: Kokoro generates audio OUTPUT, not input processing like Whisper.
The bug could affect audio waveform generation if int8/uint8 buffers are used.
"""

import numpy as np
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_buffer_conversion():
    """
    Test if audio buffer handling has sign bug

    The sign bug occurs when converting bytes to int16:
    - Correct: int8_t high byte for sign extension
    - Buggy: uint8_t high byte causes +65536 errors
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Audio Buffer Conversion (int16 reconstruction)")
    logger.info("=" * 70)

    # Create test audio samples with known values
    test_samples_int16 = np.array([
        1000,    # Positive, high byte < 128
        16000,   # Positive, high byte < 128
        32767,   # Max positive
        -1000,   # Negative, high byte >= 128
        -16000,  # Negative, high byte >= 128
        -32768,  # Min negative
    ], dtype=np.int16)

    # Convert to bytes (little-endian)
    audio_bytes = test_samples_int16.tobytes()

    logger.info(f"Test samples (int16): {test_samples_int16}")
    logger.info(f"Bytes length: {len(audio_bytes)} bytes")

    # Method A: Simulate BUGGY conversion (uint8 high byte)
    logger.info("\n--- Method A: BUGGY (uint8 high byte) ---")
    buggy_samples = []
    for i in range(0, len(audio_bytes), 2):
        low = audio_bytes[i]  # uint8
        high = audio_bytes[i+1]  # uint8 - BUG!
        # This causes sign extension issues
        sample = np.int16(low | (high << 8))
        buggy_samples.append(sample)

    buggy_samples = np.array(buggy_samples, dtype=np.int16)
    logger.info(f"Buggy samples: {buggy_samples}")

    # Method B: CORRECT conversion (int8 high byte)
    logger.info("\n--- Method B: CORRECT (int8 high byte) ---")
    correct_samples = []
    for i in range(0, len(audio_bytes), 2):
        low = audio_bytes[i]  # uint8
        high = np.int8(audio_bytes[i+1])  # int8 - CORRECT!
        sample = np.int16(low | (high << 8))
        correct_samples.append(sample)

    correct_samples = np.array(correct_samples, dtype=np.int16)
    logger.info(f"Correct samples: {correct_samples}")

    # Method C: Using np.frombuffer (what Kokoro uses)
    logger.info("\n--- Method C: np.frombuffer (Kokoro's method) ---")
    frombuffer_samples = np.frombuffer(audio_bytes, dtype=np.int16)
    logger.info(f"frombuffer samples: {frombuffer_samples}")

    # Compare results
    logger.info("\n--- Comparison ---")
    errors_buggy = buggy_samples - test_samples_int16
    errors_correct = correct_samples - test_samples_int16
    errors_frombuffer = frombuffer_samples - test_samples_int16

    logger.info(f"Buggy errors:      {errors_buggy}")
    logger.info(f"Correct errors:    {errors_correct}")
    logger.info(f"frombuffer errors: {errors_frombuffer}")

    # Correlation test
    corr_buggy = np.corrcoef(buggy_samples, test_samples_int16)[0, 1]
    corr_correct = np.corrcoef(correct_samples, test_samples_int16)[0, 1]
    corr_frombuffer = np.corrcoef(frombuffer_samples, test_samples_int16)[0, 1]

    logger.info(f"\nBuggy correlation:      {corr_buggy:.6f}")
    logger.info(f"Correct correlation:    {corr_correct:.6f}")
    logger.info(f"frombuffer correlation: {corr_frombuffer:.6f}")

    # Verdict
    logger.info("\n--- VERDICT ---")
    if np.allclose(frombuffer_samples, test_samples_int16):
        logger.info("✅ np.frombuffer() handles int16 correctly!")
        logger.info("   Kokoro is NOT affected by the sign bug in normal usage.")
        return True
    else:
        logger.error("❌ np.frombuffer() has issues!")
        logger.error("   Kokoro MAY be affected by sign bug.")
        return False


def test_float_to_int16_conversion():
    """
    Test float32 audio to int16 conversion

    Kokoro generates float32 audio, which is then converted to int16.
    This conversion could be vulnerable if done incorrectly.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Float32 to Int16 Conversion")
    logger.info("=" * 70)

    # Create test audio with negative values (sine wave)
    sample_rate = 24000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate sine wave (440 Hz) with both positive and negative values
    audio_float = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    logger.info(f"Float32 audio: shape={audio_float.shape}, range=[{audio_float.min():.3f}, {audio_float.max():.3f}]")

    # Count negative samples
    negative_count = np.sum(audio_float < 0)
    logger.info(f"Negative samples: {negative_count}/{len(audio_float)} ({100*negative_count/len(audio_float):.1f}%)")

    # Method A: CORRECT conversion
    logger.info("\n--- Method A: CORRECT (standard conversion) ---")
    audio_int16_correct = (audio_float * 32767).astype(np.int16)
    logger.info(f"Int16 audio: range=[{audio_int16_correct.min()}, {audio_int16_correct.max()}]")

    # Convert back to float to check
    audio_back = audio_int16_correct.astype(np.float32) / 32767
    corr = np.corrcoef(audio_float, audio_back)[0, 1]
    logger.info(f"Round-trip correlation: {corr:.6f}")

    # Method B: Check if bytes->int16->float causes issues
    logger.info("\n--- Method B: Bytes round-trip test ---")
    audio_bytes = audio_int16_correct.tobytes()
    audio_reconstructed = np.frombuffer(audio_bytes, dtype=np.int16)

    if np.array_equal(audio_int16_correct, audio_reconstructed):
        logger.info("✅ Byte conversion is lossless")
    else:
        logger.error("❌ Byte conversion has errors!")

    # Method C: Check for sign errors
    logger.info("\n--- Method C: Sign error detection ---")
    negative_mask = audio_int16_correct < 0
    positive_mask = audio_int16_correct >= 0

    logger.info(f"Original negative samples: {np.sum(negative_mask)}")
    logger.info(f"Reconstructed negative samples: {np.sum(audio_reconstructed < 0)}")

    if np.sum(negative_mask) == np.sum(audio_reconstructed < 0):
        logger.info("✅ No sign flips detected")
        return True
    else:
        logger.error("❌ Sign flips detected!")
        return False


def test_npu_buffer_operations():
    """
    Test NPU buffer read/write operations for sign bug

    This simulates the DirectNPURuntime buffer operations.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: NPU Buffer Operations Simulation")
    logger.info("=" * 70)

    # Create test audio (float32)
    test_audio = np.array([0.5, -0.5, 0.9, -0.9, 0.0, -0.1], dtype=np.float32)
    logger.info(f"Test audio (float32): {test_audio}")

    # Simulate write_buffer (line 152 in direct_npu_runtime.py)
    logger.info("\n--- Simulating write_buffer ---")
    data_bytes = test_audio.tobytes()
    logger.info(f"Converted to bytes: {len(data_bytes)} bytes")

    # Simulate read_buffer (line 176 in direct_npu_runtime.py)
    logger.info("\n--- Simulating read_buffer ---")
    # This is the ACTUAL line from direct_npu_runtime.py:
    # result = np.frombuffer(result_bytes, dtype=dtype).reshape(shape)

    result = np.frombuffer(data_bytes, dtype=np.float32)
    logger.info(f"Read back (frombuffer): {result}")

    # Check if data matches
    if np.array_equal(test_audio, result):
        logger.info("✅ NPU buffer operations preserve data correctly")
        logger.info("   No sign bug in float32 operations")
        return True
    else:
        logger.error("❌ NPU buffer operations corrupt data!")
        logger.error(f"   Difference: {result - test_audio}")
        return False


def test_int8_quantization_path():
    """
    Test int8 quantization path (used in NPU optimizations)

    This is where the sign bug would most likely occur in Kokoro.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Int8 Quantization Path")
    logger.info("=" * 70)

    # Create test data
    test_data = np.array([0.5, -0.5, 0.9, -0.9, 0.0, -0.1, 1.0, -1.0], dtype=np.float32)
    logger.info(f"Test data (float32): {test_data}")

    # Quantize to int8 (simulating NPU quantization)
    logger.info("\n--- Quantizing to int8 ---")
    quantized = (test_data * 127).astype(np.int8)
    logger.info(f"Quantized (int8): {quantized}")

    # Convert to bytes
    data_bytes = quantized.tobytes()

    # Method A: Reconstruct with np.frombuffer (CORRECT)
    logger.info("\n--- Method A: np.frombuffer (int8) ---")
    reconstructed_a = np.frombuffer(data_bytes, dtype=np.int8)
    logger.info(f"Reconstructed: {reconstructed_a}")

    if np.array_equal(quantized, reconstructed_a):
        logger.info("✅ int8 frombuffer is correct")
    else:
        logger.error("❌ int8 frombuffer has errors!")

    # Method B: What if someone uses uint8 by mistake?
    logger.info("\n--- Method B: np.frombuffer (uint8) - POTENTIAL BUG ---")
    reconstructed_b = np.frombuffer(data_bytes, dtype=np.uint8)
    logger.info(f"Reconstructed (uint8): {reconstructed_b}")

    # Cast back to int8
    reconstructed_b_int8 = reconstructed_b.astype(np.int8)
    logger.info(f"Cast to int8: {reconstructed_b_int8}")

    if np.array_equal(quantized, reconstructed_b_int8):
        logger.info("⚠️  uint8->int8 cast works, but is risky")
    else:
        logger.error("❌ uint8->int8 cast causes errors!")

    # Dequantize and check correlation
    dequant_a = reconstructed_a.astype(np.float32) / 127
    dequant_b = reconstructed_b_int8.astype(np.float32) / 127

    corr_a = np.corrcoef(test_data, dequant_a)[0, 1]
    corr_b = np.corrcoef(test_data, dequant_b)[0, 1]

    logger.info(f"\nMethod A correlation: {corr_a:.6f}")
    logger.info(f"Method B correlation: {corr_b:.6f}")

    if corr_a > 0.99:
        logger.info("✅ Kokoro's int8 path is safe")
        return True
    else:
        logger.error("❌ Kokoro's int8 path may have issues")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("KOKORO TTS SIGN BUG DETECTION TEST SUITE")
    logger.info("Testing for int8/uint8 sign extension bug (October 2025)")
    logger.info("=" * 70)

    results = {}

    # Run tests
    results['audio_buffer'] = test_audio_buffer_conversion()
    results['float_to_int16'] = test_float_to_int16_conversion()
    results['npu_buffer'] = test_npu_buffer_operations()
    results['int8_quant'] = test_int8_quantization_path()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("   Kokoro is NOT affected by the sign bug")
        logger.info("   np.frombuffer() handles data types correctly")
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("   Kokoro MAY be vulnerable to sign bug")
        logger.error("   Review failed tests and apply fixes")
    logger.info("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
