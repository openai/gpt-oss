"""JAX/Flax implementation for gpt-oss inference.

This package provides a JAX-based inference implementation for gpt-oss models,
optimized for CPU execution on Apple Silicon (ARM64) and x86-64 platforms.

Key features:
- BF16 precision throughout
- Non-quantized KV caching for efficient autoregressive generation
- Supports both SafeTensors and Orbax checkpoint formats
- MXFP4 weight decompression for MoE expert weights
"""

__all__ = [
    'ModelConfig',
    'Transformer',
    'generate',
    'get_tokenizer',
    'WeightLoader',
    'OrbaxWeightLoader',
]
