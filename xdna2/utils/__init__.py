"""
XDNA2 Utilities for Unicorn-Orator

Utilities for XDNA2 NPU-accelerated TTS including BF16 workaround.
"""

from .bf16_workaround import (
    BF16WorkaroundManager,
    matmul_bf16_safe
)

__all__ = [
    'BF16WorkaroundManager',
    'matmul_bf16_safe'
]
