"""
Integration tests for BF16 workaround in Unicorn-Orator XDNA2

Tests the BF16 signed value workaround with TTS neural network operations.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from xdna2.utils.bf16_workaround import BF16WorkaroundManager, matmul_bf16_safe


class TestBF16Workaround:
    """Test BF16 workaround functionality"""

    def test_workaround_manager_init(self):
        """Test BF16WorkaroundManager initialization"""
        manager = BF16WorkaroundManager()
        assert manager.epsilon == 1e-8
        assert manager.stats['total_calls'] == 0

    def test_prepare_inputs_positive_range(self):
        """Test input preparation with positive values"""
        manager = BF16WorkaroundManager()
        A = np.random.uniform(0, 1, (100, 100)).astype(np.float32)
        B = np.random.uniform(0, 1, (100, 100)).astype(np.float32)

        (A_scaled, B_scaled), metadata = manager.prepare_inputs(A, B)

        # Check scaled values are in [0, 1]
        assert np.all(A_scaled >= 0) and np.all(A_scaled <= 1)
        assert np.all(B_scaled >= 0) and np.all(B_scaled <= 1)

        # Check metadata
        assert 'scales' in metadata
        assert 'offsets' in metadata
        assert len(metadata['scales']) == 2
        assert len(metadata['offsets']) == 2

    def test_prepare_inputs_negative_range(self):
        """Test input preparation with negative values"""
        manager = BF16WorkaroundManager()
        A = np.random.uniform(-2, 2, (100, 100)).astype(np.float32)
        B = np.random.uniform(-1, 1, (100, 100)).astype(np.float32)

        (A_scaled, B_scaled), metadata = manager.prepare_inputs(A, B)

        # Check scaled values are in [0, 1]
        assert np.all(A_scaled >= 0) and np.all(A_scaled <= 1)
        assert np.all(B_scaled >= 0) and np.all(B_scaled <= 1)

        # Verify range
        assert np.min(A_scaled) < 0.1  # Should be close to 0
        assert np.max(A_scaled) > 0.9  # Should be close to 1

    def test_reconstruct_output_matmul(self):
        """Test output reconstruction for matrix multiplication"""
        manager = BF16WorkaroundManager()

        # Create test matrices
        A = np.random.uniform(-2, 2, (50, 50)).astype(np.float32)
        B = np.random.uniform(-2, 2, (50, 50)).astype(np.float32)

        # Prepare inputs
        (A_scaled, B_scaled), metadata = manager.prepare_inputs(A, B)

        # Simulate NPU execution
        C_scaled = A_scaled @ B_scaled

        # Reconstruct output
        C_reconstructed = manager.reconstruct_output(C_scaled, metadata, operation='matmul')

        # Compare with reference
        C_reference = A @ B

        # Calculate error
        error = np.mean(np.abs(C_reconstructed - C_reference))
        rel_error = error / (np.mean(np.abs(C_reference)) + 1e-8) * 100

        print(f"Absolute error: {error:.6f}")
        print(f"Relative error: {rel_error:.2f}%")

        # Should be much better than 789% error without workaround
        assert rel_error < 20  # Allow up to 20% error (still way better than 789%)

    def test_matmul_bf16_safe_with_workaround(self):
        """Test matmul_bf16_safe function with workaround enabled"""
        A = np.random.uniform(-2, 2, (128, 128)).astype(np.float32)
        B = np.random.uniform(-2, 2, (128, 128)).astype(np.float32)

        # Execute with workaround
        C = matmul_bf16_safe(A, B, use_workaround=True)

        # Compare with reference
        C_reference = A @ B

        error = np.mean(np.abs(C - C_reference))
        rel_error = error / (np.mean(np.abs(C_reference)) + 1e-8) * 100

        print(f"matmul_bf16_safe error: {rel_error:.2f}%")

        # Should be < 20% error
        assert rel_error < 20

    def test_matmul_bf16_safe_without_workaround(self):
        """Test matmul_bf16_safe function without workaround (CPU fallback)"""
        A = np.random.uniform(-2, 2, (128, 128)).astype(np.float32)
        B = np.random.uniform(-2, 2, (128, 128)).astype(np.float32)

        # Execute without workaround (uses CPU)
        C = matmul_bf16_safe(A, B, use_workaround=False)

        # Compare with reference
        C_reference = A @ B

        error = np.mean(np.abs(C - C_reference))

        # CPU should be exact (within floating point precision)
        assert error < 1e-5

    def test_stats_tracking(self):
        """Test statistics tracking"""
        manager = BF16WorkaroundManager()

        # Run multiple operations
        for i in range(5):
            A = np.random.randn(50, 50).astype(np.float32)
            B = np.random.randn(50, 50).astype(np.float32)
            manager.prepare_inputs(A, B)

        stats = manager.get_stats()

        assert stats['total_calls'] == 5
        assert stats['max_input_range'] > 0
        assert stats['min_input_range'] > 0

    def test_reset_stats(self):
        """Test statistics reset"""
        manager = BF16WorkaroundManager()

        # Run operations
        A = np.random.randn(50, 50).astype(np.float32)
        B = np.random.randn(50, 50).astype(np.float32)
        manager.prepare_inputs(A, B)

        # Reset stats
        manager.reset_stats()

        stats = manager.get_stats()
        assert stats['total_calls'] == 0


class TestTTSIntegration:
    """Test BF16 workaround integration with TTS operations"""

    def test_embedding_layer(self):
        """Test embedding layer with BF16 workaround"""
        manager = BF16WorkaroundManager()

        # Simulate token embedding lookup
        # In real TTS: tokens @ embedding_matrix
        tokens_one_hot = np.random.randn(1, 100, 512).astype(np.float32)
        embedding_matrix = np.random.randn(512, 256).astype(np.float32) * 0.1

        # For each token, multiply with embedding matrix
        embeddings = []
        for i in range(tokens_one_hot.shape[1]):
            token = tokens_one_hot[0, i:i+1, :]  # [1, 512]

            # Use BF16 safe matmul
            (token_scaled, emb_scaled), metadata = manager.prepare_inputs(
                token, embedding_matrix
            )
            result_scaled = token_scaled @ emb_scaled
            result = manager.reconstruct_output(result_scaled, metadata, 'matmul')

            embeddings.append(result)

        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape == (100, 1, 256)

    def test_attention_mechanism(self):
        """Test attention mechanism with BF16 workaround"""
        manager = BF16WorkaroundManager()

        # Simulate attention: Q @ K^T
        seq_len = 50
        d_model = 256

        Q = np.random.randn(1, seq_len, d_model).astype(np.float32) * 0.1
        K = np.random.randn(1, seq_len, d_model).astype(np.float32) * 0.1

        # Reshape for matmul
        Q_2d = Q.reshape(-1, d_model)  # [seq_len, d_model]
        K_2d = K.reshape(-1, d_model)  # [seq_len, d_model]

        # Compute attention scores with workaround
        (Q_scaled, K_scaled), metadata = manager.prepare_inputs(Q_2d, K_2d.T)
        attn_scaled = Q_scaled @ K_scaled
        attn_scores = manager.reconstruct_output(attn_scaled, metadata, 'matmul')

        assert attn_scores.shape == (seq_len, seq_len)

    def test_feed_forward_network(self):
        """Test feed-forward network with BF16 workaround"""
        manager = BF16WorkaroundManager()

        # Simulate FFN: x @ W1 + b1 -> ReLU -> @ W2 + b2
        batch_size = 1
        seq_len = 100
        d_model = 256
        d_ff = 1024

        x = np.random.randn(batch_size * seq_len, d_model).astype(np.float32) * 0.1
        W1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.1
        W2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.1

        # First linear layer with BF16 workaround
        (x_scaled, W1_scaled), meta1 = manager.prepare_inputs(x, W1)
        h_scaled = x_scaled @ W1_scaled
        h = manager.reconstruct_output(h_scaled, meta1, 'matmul')

        # ReLU activation
        h = np.maximum(0, h)

        # Second linear layer with BF16 workaround
        (h_scaled, W2_scaled), meta2 = manager.prepare_inputs(h, W2)
        out_scaled = h_scaled @ W2_scaled
        out = manager.reconstruct_output(out_scaled, meta2, 'matmul')

        assert out.shape == (batch_size * seq_len, d_model)

    def test_full_tts_pipeline_simulation(self):
        """Test full TTS pipeline with BF16 workaround"""
        manager = BF16WorkaroundManager()

        # Simulate simplified TTS pipeline
        batch_size = 1
        seq_len = 50
        d_model = 256
        vocab_size = 1000

        # 1. Token embedding
        tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
        token_emb_matrix = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.1

        # Convert tokens to one-hot
        tokens_one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)
        for i, t in enumerate(tokens[0]):
            tokens_one_hot[i, t] = 1.0

        # Embed tokens with BF16 workaround
        (tok_scaled, emb_scaled), meta = manager.prepare_inputs(tokens_one_hot, token_emb_matrix)
        embeddings_scaled = tok_scaled @ emb_scaled
        embeddings = manager.reconstruct_output(embeddings_scaled, meta, 'matmul')

        assert embeddings.shape == (seq_len, d_model)

        # 2. Transformer layer (simplified)
        # Self-attention + FFN already tested above

        # 3. Output projection
        out_proj = np.random.randn(d_model, 80).astype(np.float32) * 0.1  # 80 mel bins

        (emb_scaled, proj_scaled), meta = manager.prepare_inputs(embeddings, out_proj)
        mel_scaled = emb_scaled @ proj_scaled
        mel_spectrogram = manager.reconstruct_output(mel_scaled, meta, 'matmul')

        assert mel_spectrogram.shape == (seq_len, 80)

        print(f"Full pipeline test passed!")
        print(f"Total BF16 operations: {manager.get_stats()['total_calls']}")


class TestPerformance:
    """Test performance characteristics of BF16 workaround"""

    def test_overhead_measurement(self):
        """Measure overhead of BF16 workaround"""
        import time

        manager = BF16WorkaroundManager()

        # Test matrices
        sizes = [128, 256, 512]

        for size in sizes:
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Time with workaround
            start = time.time()
            for _ in range(10):
                (A_s, B_s), meta = manager.prepare_inputs(A, B)
                C_s = A_s @ B_s
                C = manager.reconstruct_output(C_s, meta, 'matmul')
            time_with = (time.time() - start) / 10

            # Time without workaround
            start = time.time()
            for _ in range(10):
                C = A @ B
            time_without = (time.time() - start) / 10

            overhead = (time_with - time_without) / time_without * 100

            print(f"Size {size}x{size}: overhead = {overhead:.2f}%")

            # Overhead should be < 10% for large matrices
            if size >= 256:
                assert overhead < 15  # Allow 15% overhead

    def test_accuracy_vs_size(self):
        """Test accuracy across different matrix sizes"""
        manager = BF16WorkaroundManager()

        sizes = [64, 128, 256, 512]
        errors = []

        for size in sizes:
            A = np.random.uniform(-2, 2, (size, size)).astype(np.float32)
            B = np.random.uniform(-2, 2, (size, size)).astype(np.float32)

            C_ref = A @ B

            (A_s, B_s), meta = manager.prepare_inputs(A, B)
            C_s = A_s @ B_s
            C = manager.reconstruct_output(C_s, meta, 'matmul')

            error = np.mean(np.abs(C - C_ref))
            rel_error = error / (np.mean(np.abs(C_ref)) + 1e-8) * 100

            errors.append(rel_error)

            print(f"Size {size}x{size}: relative error = {rel_error:.2f}%")

        # All errors should be < 20%
        assert all(e < 20 for e in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
