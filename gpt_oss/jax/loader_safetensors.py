"""SafeTensors weight loading with MXFP4 decompression for gpt-oss-20b.

This module handles loading weights from SafeTensors format and decompressing
MXFP4 quantized tensors (used for MoE expert weights).

MXFP4 Format:
- 4-bit floating point with block-based exponent scaling
- 16-value FP4 lookup table
- 2 FP4 values packed per uint8 byte
- Scale factors biased by 127
"""

import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from flax import traverse_util
from tqdm import tqdm

# Handle both module import and direct execution
try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig

# FP4 lookup table (16 values: 8 positive, 8 negative)
FP4_VALUES = np.array([
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)


def decompress_mxfp4(
    blocks: np.ndarray,
    scales: np.ndarray,
    target_shape: tuple
) -> jnp.ndarray:
    """Decompress MXFP4 quantized tensor to BF16.

    MXFP4 uses 4-bit floating point with block-based exponent scaling:
    1. Each uint8 byte contains 2 FP4 values (4 bits each)
    2. FP4 nibbles index into a 16-value lookup table (mantissas)
    3. Scales provide per-block exponent scaling (biased by 127)
    4. Final value = mantissa * 2^(scale - 127)

    Args:
        blocks: MXFP4 blocks, shape [num_experts, out_dim, groups, 16]
                Each uint8 contains 2 packed FP4 values
                groups = in_dim // 32, where 32 = 16 bytes * 2 FP4 values per byte
        scales: Exponent scales, shape [num_experts, out_dim, groups]
                Values are biased by 127
        target_shape: Expected output shape [num_experts, out_dim, in_dim]

    Returns:
        Decompressed BF16 tensor of shape target_shape

    Example:
        >>> blocks = np.array([[[[0x12, 0x34]]]], dtype=np.uint8)  # 1 expert, 1 row, 1 group, 2 bytes
        >>> scales = np.array([[[127]]], dtype=np.uint8)  # scale = 0 (unbiased)
        >>> output = decompress_mxfp4(blocks, scales, (1, 1, 4))
        >>> # Unpacks nibbles: [0x1, 0x2, 0x3, 0x4] → [+0.5, +1.0, +1.5, +2.0]
    """
    assert blocks.dtype == np.uint8, f"decompress_mxfp4: blocks must be uint8, got {blocks.dtype}"
    assert scales.dtype == np.uint8, f"decompress_mxfp4: scales must be uint8, got {scales.dtype}"

    # blocks: [num_experts, out_dim, groups, 16]
    # scales: [num_experts, out_dim, groups]
    # target: [num_experts, out_dim, in_dim] where in_dim = groups * 32
    assert len(blocks.shape) == 4, f"decompress_mxfp4: blocks must be 4D, got shape {blocks.shape}"
    assert len(scales.shape) == 3, f"decompress_mxfp4: scales must be 3D, got shape {scales.shape}"
    assert len(target_shape) == 3, f"decompress_mxfp4: target_shape must be 3D, got {target_shape}"

    num_experts, out_dim, groups, block_size = blocks.shape
    expected_in_dim = target_shape[2]

    assert block_size == 16, f"decompress_mxfp4: expected block_size=16, got {block_size}"
    assert groups * block_size * 2 == expected_in_dim, \
        f"decompress_mxfp4: groups * 32 = {groups * 32} != target in_dim {expected_in_dim}"
    assert scales.shape == (num_experts, out_dim, groups), \
        f"decompress_mxfp4: scales shape {scales.shape} != expected {(num_experts, out_dim, groups)}"

    # Unpack nibbles: each uint8 → 2 FP4 values
    # Low nibble (bits 0-3), high nibble (bits 4-7)
    idx_lo = (blocks & 0x0F).astype(np.int32)  # [num_experts, out_dim, groups, 16]
    idx_hi = (blocks >> 4).astype(np.int32)    # [num_experts, out_dim, groups, 16]

    # Lookup mantissas from FP4 table
    mantissas_lo = FP4_VALUES[idx_lo]  # [num_experts, out_dim, groups, 16]
    mantissas_hi = FP4_VALUES[idx_hi]  # [num_experts, out_dim, groups, 16]

    # Interleave mantissas: [lo[0], hi[0], lo[1], hi[1], ...]
    # This matches the PyTorch implementation's packing convention
    mantissas = np.empty((num_experts, out_dim, groups, block_size * 2), dtype=np.float32)
    mantissas[:, :, :, 0::2] = mantissas_lo
    mantissas[:, :, :, 1::2] = mantissas_hi
    # Shape: [num_experts, out_dim, groups, 32]

    # Apply exponent scaling: value = mantissa * 2^(scale - 127)
    # PyTorch does: scales.reshape(rows_total, 1) which broadcasts across the 32-value dimension
    # We need scales [num_experts, out_dim, groups] → [num_experts, out_dim, groups, 1]
    exponents = scales.astype(np.int32) - 127  # Unbias: [num_experts, out_dim, groups]
    exponents = exponents[:, :, :, np.newaxis]  # [num_experts, out_dim, groups, 1]

    # Use ldexp for efficient 2^exp scaling (broadcasts across last dim)
    # mantissas: [num_experts, out_dim, groups, 32]
    # exponents: [num_experts, out_dim, groups, 1]
    # → output: [num_experts, out_dim, groups, 32]
    output = np.ldexp(mantissas, exponents)

    # Flatten groups dimension: [num_experts, out_dim, groups, 32] → [num_experts, out_dim, groups*32]
    output = output.reshape(num_experts, out_dim, groups * block_size * 2)

    # Convert to BF16 (stay in NumPy, convert to JAX later for efficiency)
    output_bf16 = output.astype(np.float32)  # BF16 not natively supported in NumPy
    return jnp.array(output_bf16, dtype=jnp.bfloat16)


def decompress_mxfp4_2d(
    blocks: np.ndarray,
    scales: np.ndarray,
    target_shape: tuple
) -> jnp.ndarray:
    """Decompress MXFP4 quantized 2D tensor to BF16.

    Similar to decompress_mxfp4 but for 2D tensors (used for some weight matrices).

    Args:
        blocks: MXFP4 blocks, shape [out_dim, in_dim // 2]
        scales: Exponent scales, shape [out_dim]
        target_shape: Expected output shape [out_dim, in_dim]

    Returns:
        Decompressed BF16 tensor of shape target_shape
    """
    assert blocks.dtype == np.uint8, f"decompress_mxfp4_2d: blocks must be uint8, got {blocks.dtype}"
    assert scales.dtype == np.uint8, f"decompress_mxfp4_2d: scales must be uint8, got {scales.dtype}"

    out_dim, packed_in = blocks.shape
    expected_in = target_shape[1]

    assert packed_in * 2 == expected_in, \
        f"decompress_mxfp4_2d: packed dimension {packed_in} * 2 != target in {expected_in}"
    assert scales.shape == (out_dim,), \
        f"decompress_mxfp4_2d: scales shape {scales.shape} != expected ({out_dim},)"

    # Unpack nibbles
    idx_lo = (blocks & 0x0F).astype(np.int32)
    idx_hi = (blocks >> 4).astype(np.int32)

    # Lookup mantissas
    mantissas_lo = FP4_VALUES[idx_lo]
    mantissas_hi = FP4_VALUES[idx_hi]

    # Interleave mantissas
    mantissas = np.empty((out_dim, expected_in), dtype=np.float32)
    mantissas[:, 0::2] = mantissas_lo
    mantissas[:, 1::2] = mantissas_hi

    # Apply exponent scaling
    exponents = scales.astype(np.int32) - 127
    exponents = exponents[:, np.newaxis]
    output = np.ldexp(mantissas, exponents)

    return jnp.array(output, dtype=jnp.bfloat16)


def create_param_name_mapping(num_layers: int = 24) -> Dict[str, Any]:
    """Create parameter name mapping from PyTorch checkpoint to Flax parameters.

    PyTorch uses:
        - Linear layers: weight [out, in], bias [out]
        - Naming: block.{n}.attn.qkv.weight, block.{n}.mlp.gate.bias, etc.

    Flax uses:
        - Dense layers: kernel [in, out], bias [out]  ← TRANSPOSE REQUIRED!
        - Naming: params/block_{n}/attn/qkv/kernel, params/block_{n}/mlp/gate/bias, etc.

    Returns:
        Mapping from Flax path tuples to checkpoint names or (blocks_name, scales_name) for MXFP4.

    Example:
        >>> mapping = create_param_name_mapping(num_layers=24)
        >>> mapping[('embedding', 'embedding')]
        'embedding.weight'
        >>> mapping[('block_0', 'attn', 'qkv', 'kernel')]
        ('block.0.attn.qkv.weight', True)  # True means transpose
        >>> mapping[('block_0', 'mlp', 'mlp1_weight')]
        ('block.0.mlp.mlp1_weight.blocks', 'block.0.mlp.mlp1_weight.scales')  # MXFP4
    """
    mapping = {}

    # Embedding (no transpose, it's an Embed layer not Dense)
    mapping[('embedding', 'embedding')] = 'embedding.weight'

    # Transformer blocks
    for layer_idx in range(num_layers):
        flax_prefix = f'block_{layer_idx}'
        torch_prefix = f'block.{layer_idx}'

        # Attention
        # - norm.scale (no transpose)
        mapping[(flax_prefix, 'attn', 'norm', 'scale')] = f'{torch_prefix}.attn.norm.scale'
        # - qkv: Dense layer → TRANSPOSE
        mapping[(flax_prefix, 'attn', 'qkv', 'kernel')] = (f'{torch_prefix}.attn.qkv.weight', True)
        mapping[(flax_prefix, 'attn', 'qkv', 'bias')] = f'{torch_prefix}.attn.qkv.bias'
        # - sinks (no transpose)
        mapping[(flax_prefix, 'attn', 'sinks')] = f'{torch_prefix}.attn.sinks'
        # - out: Dense layer → TRANSPOSE
        mapping[(flax_prefix, 'attn', 'out', 'kernel')] = (f'{torch_prefix}.attn.out.weight', True)
        mapping[(flax_prefix, 'attn', 'out', 'bias')] = f'{torch_prefix}.attn.out.bias'

        # MLP
        # - norm.scale (no transpose)
        mapping[(flax_prefix, 'mlp', 'norm', 'scale')] = f'{torch_prefix}.mlp.norm.scale'
        # - gate: Dense layer → TRANSPOSE
        mapping[(flax_prefix, 'mlp', 'gate', 'kernel')] = (f'{torch_prefix}.mlp.gate.weight', True)
        mapping[(flax_prefix, 'mlp', 'gate', 'bias')] = f'{torch_prefix}.mlp.gate.bias'
        # - mlp1_weight: MXFP4 (3D tensor, no transpose - already correct shape)
        mapping[(flax_prefix, 'mlp', 'mlp1_weight')] = (
            f'{torch_prefix}.mlp.mlp1_weight.blocks',
            f'{torch_prefix}.mlp.mlp1_weight.scales'
        )
        mapping[(flax_prefix, 'mlp', 'mlp1_bias')] = f'{torch_prefix}.mlp.mlp1_bias'
        # - mlp2_weight: MXFP4 (3D tensor, no transpose)
        mapping[(flax_prefix, 'mlp', 'mlp2_weight')] = (
            f'{torch_prefix}.mlp.mlp2_weight.blocks',
            f'{torch_prefix}.mlp.mlp2_weight.scales'
        )
        mapping[(flax_prefix, 'mlp', 'mlp2_bias')] = f'{torch_prefix}.mlp.mlp2_bias'

    # Final norm
    mapping[('norm', 'scale')] = 'norm.scale'

    # Unembedding: Dense layer → TRANSPOSE
    mapping[('unembedding', 'kernel')] = ('unembedding.weight', True)

    return mapping


class WeightLoader:
    """Load weights from SafeTensors checkpoint into Flax parameter tree.

    Handles:
    - MXFP4 decompression for MoE expert weights
    - Parameter name mapping (PyTorch → Flax)
    - Transpose for Dense layers (PyTorch [out, in] → Flax [in, out])
    - Memory-mapped loading for large checkpoints
    - Progress bar for loading feedback

    Example:
        >>> loader = WeightLoader('gpt-oss-20b/original/')
        >>> config = ModelConfig(...)
        >>> params = loader.load_params(config)
        >>> model = Transformer(config=config)
        >>> logits = model.apply({'params': params}, tokens)
    """

    def __init__(self, checkpoint_path: str):
        """Initialize weight loader.

        Args:
            checkpoint_path: Path to directory containing model.safetensors
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Find all .safetensors files in directory
        safetensor_files = list(self.checkpoint_path.glob('*.safetensors'))
        assert len(safetensor_files) > 0, \
            f"WeightLoader: No .safetensors files found in {checkpoint_path}"

        # Build mapping from tensor name to file
        self.tensor_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(str(safetensor_file), framework='np', device='cpu') as f:
                for key in f.keys():
                    self.tensor_to_file[key] = safetensor_file

        # Keep file handles open for faster access (avoid repeated file opens)
        self.file_handles = {}
        for safetensor_file in safetensor_files:
            self.file_handles[safetensor_file] = safe_open(
                str(safetensor_file), framework='np', device='cpu'
            )

        print(f"WeightLoader: Found {len(self.tensor_to_file)} tensors in {len(safetensor_files)} file(s)")

    def _get_tensor(self, name: str) -> np.ndarray:
        """Load a single tensor from checkpoint (memory-mapped).

        Uses pre-opened file handles to avoid repeated file opens.
        """
        assert name in self.tensor_to_file, \
            f"WeightLoader._get_tensor: Tensor '{name}' not found in checkpoint"

        safetensor_file = self.tensor_to_file[name]
        return self.file_handles[safetensor_file].get_tensor(name)

    def _get_mxfp4_tensor_3d(
        self,
        blocks_name: str,
        scales_name: str
    ) -> jnp.ndarray:
        """Load and decompress MXFP4 3D tensor (for MoE expert weights).

        Args:
            blocks_name: Name of blocks tensor (uint8)
            scales_name: Name of scales tensor (uint8)

        Returns:
            Decompressed BF16 tensor [num_experts, out_dim, in_dim]
        """
        blocks = self._get_tensor(blocks_name)
        scales = self._get_tensor(scales_name)

        # MXFP4 blocks shape: [num_experts, out_dim, in_dim // 32, 16]
        # Target shape: [num_experts, out_dim, in_dim]
        num_experts, out_dim, groups, block_size = blocks.shape
        assert block_size == 16, \
            f"WeightLoader: Expected MXFP4 block_size=16, got {block_size}"

        in_dim = groups * block_size * 2  # Each uint8 packs 2 FP4 values
        target_shape = (num_experts, out_dim, in_dim)

        return decompress_mxfp4(blocks, scales, target_shape)

    def load_params(self, config: ModelConfig, show_progress: bool = True) -> Dict[str, Any]:
        """Load all model parameters from checkpoint.

        Args:
            config: Model configuration
            show_progress: Show progress bar during loading

        Returns:
            Flax parameter dictionary suitable for model.apply({'params': params}, ...)
        """
        import time

        # Create parameter name mapping
        param_mapping = create_param_name_mapping(num_layers=config.num_hidden_layers)

        # Load all parameters with progress bar
        flat_params = {}

        items = list(param_mapping.items())
        iterator = tqdm(items, desc="Loading weights", disable=not show_progress)

        # Timing stats
        time_io = 0.0
        time_decompress = 0.0
        time_jax_convert = 0.0

        for idx, (flax_path, checkpoint_spec) in enumerate(iterator):
            # Log timing for first few parameters
            t_start = time.time()

            # Handle different checkpoint specs
            if isinstance(checkpoint_spec, tuple) and len(checkpoint_spec) == 2:
                if isinstance(checkpoint_spec[1], bool):
                    # (checkpoint_name, transpose_flag)
                    checkpoint_name, should_transpose = checkpoint_spec

                    t_io_start = time.time()
                    tensor = self._get_tensor(checkpoint_name)
                    time_io += time.time() - t_io_start

                    if should_transpose:
                        # PyTorch Linear [out, in] → Flax Dense [in, out]
                        tensor = tensor.T

                    t_jax_start = time.time()
                    flat_params[flax_path] = jnp.array(tensor, dtype=jnp.bfloat16)
                    time_jax_convert += time.time() - t_jax_start

                else:
                    # (blocks_name, scales_name) - MXFP4
                    blocks_name, scales_name = checkpoint_spec

                    t_io_start = time.time()
                    blocks = self._get_tensor(blocks_name)
                    scales = self._get_tensor(scales_name)
                    time_io += time.time() - t_io_start

                    t_decompress_start = time.time()
                    flat_params[flax_path] = self._get_mxfp4_tensor_3d(blocks_name, scales_name)
                    time_decompress += time.time() - t_decompress_start

            else:
                # Simple checkpoint name (no transpose)
                t_io_start = time.time()
                tensor = self._get_tensor(checkpoint_spec)
                time_io += time.time() - t_io_start

                t_jax_start = time.time()
                flat_params[flax_path] = jnp.array(tensor, dtype=jnp.bfloat16)
                time_jax_convert += time.time() - t_jax_start

            # Log first few parameters
            if idx < 3:
                t_total = time.time() - t_start
                print(f"\n[Param {idx}] {flax_path}: {t_total:.3f}s")

        # Convert flat dict to nested dict for Flax
        params = traverse_util.unflatten_dict(flat_params)

        if show_progress:
            print(f"\n✓ Loaded {len(flat_params)} parameters")
            print(f"Timing breakdown:")
            print(f"  I/O (SafeTensors): {time_io:.2f}s")
            print(f"  MXFP4 decompress: {time_decompress:.2f}s")
            print(f"  JAX conversion: {time_jax_convert:.2f}s")

        return params


if __name__ == "__main__":
    """Test MXFP4 decompression with known values."""
    print("Testing MXFP4 decompression...")
    print("=" * 80)

    # Test 1: 4D case (matches actual MoE weight structure)
    print("\nTest 1: 4D tensor (MoE weights)")
    print("-" * 80)

    # Create test data: 2 experts, 4 rows, 1 group, 16 bytes (→ 32 values)
    # Expert 0: all 0x11 (nibbles 0x1, 0x1) → FP4[1] = +0.5
    # Expert 1: all 0x22 (nibbles 0x2, 0x2) → FP4[2] = +1.0
    blocks_3d = np.full((2, 4, 1, 16), 0x11, dtype=np.uint8)  # Expert 0
    blocks_3d[1, :, :, :] = 0x22  # Expert 1

    # Scales: all 127 (unbiased scale = 0, so no scaling)
    # Shape: [num_experts, out_dim, groups]
    scales_3d = np.full((2, 4, 1), 127, dtype=np.uint8)

    output_3d = decompress_mxfp4(blocks_3d, scales_3d, (2, 4, 32))

    print(f"Input blocks shape: {blocks_3d.shape}")
    print(f"Input scales shape: {scales_3d.shape}")
    print(f"Output shape: {output_3d.shape}")
    print(f"Expected output: Expert 0 all +0.5, Expert 1 all +1.0")
    print(f"Expert 0 values (first 4): {output_3d[0, 0, :4]}")
    print(f"Expert 1 values (first 4): {output_3d[1, 0, :4]}")

    # Validate
    assert output_3d.shape == (2, 4, 32), f"Wrong output shape: {output_3d.shape}"
    assert np.allclose(output_3d[0], 0.5, atol=1e-3), "Expert 0 should be all +0.5"
    assert np.allclose(output_3d[1], 1.0, atol=1e-3), "Expert 1 should be all +1.0"
    print("✓ Test 1 passed")

    # Test 2: Exponent scaling
    print("\nTest 2: Exponent scaling")
    print("-" * 80)

    # Same blocks, but scale expert 0 by 2^1 = 2, expert 1 by 2^-1 = 0.5
    scales_scaled = np.array([
        [[128], [128], [128], [128]],  # Expert 0: scale = 1 → multiply by 2
        [[126], [126], [126], [126]],  # Expert 1: scale = -1 → multiply by 0.5
    ], dtype=np.uint8)

    output_scaled = decompress_mxfp4(blocks_3d, scales_scaled, (2, 4, 32))

    print(f"Expected output: Expert 0 all +1.0 (0.5 * 2), Expert 1 all +0.5 (1.0 * 0.5)")
    print(f"Expert 0 values (first 4): {output_scaled[0, 0, :4]}")
    print(f"Expert 1 values (first 4): {output_scaled[1, 0, :4]}")

    assert np.allclose(output_scaled[0], 1.0, atol=1e-3), "Expert 0 should be all +1.0"
    assert np.allclose(output_scaled[1], 0.5, atol=1e-3), "Expert 1 should be all +0.5"
    print("✓ Test 2 passed")

    # Test 3: 2D tensor
    print("\nTest 3: 2D tensor")
    print("-" * 80)

    blocks_2d = np.array([
        [0x33, 0x33],  # Row 0: nibbles 0x3 → FP4[3] = +1.5
        [0x44, 0x44],  # Row 1: nibbles 0x4 → FP4[4] = +2.0
    ], dtype=np.uint8)
    scales_2d = np.array([127, 127], dtype=np.uint8)

    output_2d = decompress_mxfp4_2d(blocks_2d, scales_2d, (2, 4))

    print(f"Input blocks shape: {blocks_2d.shape}")
    print(f"Output shape: {output_2d.shape}")
    print(f"Row 0 values: {output_2d[0]}")
    print(f"Row 1 values: {output_2d[1]}")

    assert output_2d.shape == (2, 4), f"Wrong output shape: {output_2d.shape}"
    assert np.allclose(output_2d[0], 1.5, atol=1e-3), "Row 0 should be all +1.5"
    assert np.allclose(output_2d[1], 2.0, atol=1e-3), "Row 1 should be all +2.0"
    print("✓ Test 3 passed")

    # Test 4: Negative values
    print("\nTest 4: Negative values")
    print("-" * 80)

    # Use high nibbles (0x8-0xF) for negative values
    # 0x88 → nibbles 0x8, 0x8 → FP4[8] = -0.0
    # 0x99 → nibbles 0x9, 0x9 → FP4[9] = -0.5
    blocks_neg = np.full((1, 1, 1, 16), 0x99, dtype=np.uint8)  # 1 expert, 1 row, 1 group, 16 bytes
    scales_neg = np.full((1, 1, 1), 127, dtype=np.uint8)

    output_neg = decompress_mxfp4(blocks_neg, scales_neg, (1, 1, 32))

    print(f"Expected output: all -0.5")
    print(f"Values (first 4): {output_neg[0, 0, :4]}")

    assert np.allclose(output_neg[0, 0], -0.5, atol=1e-3), "Should be all -0.5"
    print("✓ Test 4 passed")

    print("\n" + "=" * 80)
    print("All MXFP4 decompression tests passed!")
