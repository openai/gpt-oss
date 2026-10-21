"""Orbax checkpoint loader for GPT-OSS models.

This module provides fast weight loading from pre-converted Orbax checkpoints.
Much faster than loading SafeTensors (5s vs 90s).

Supports both gpt-oss-20b and gpt-oss-120b models, with MXFP4 quantized
weight unpacking.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec, Mesh

# Import MXFP4 unpacking utilities
from .mx_formats import unpack_quantized_param_tree


def translate_orbax_to_model_structure(orbax_params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Orbax checkpoint structure to JAX model structure.

    Orbax checkpoint uses:
        - embed_tokens/embedding
        - layers/0/...
        - lm_head/kernel
        - norm/scale

    JAX model expects:
        - embedding/embedding
        - block_0/...
        - unembedding/kernel
        - norm/scale

    Args:
        orbax_params: Parameters from Orbax checkpoint

    Returns:
        Translated parameters matching JAX model structure
    """
    translated = {}

    for key, value in orbax_params.items():
        if key == 'embed_tokens':
            # Map embed_tokens → embedding
            translated['embedding'] = value
        elif key == 'layers':
            # Map layers/N → block_N
            for layer_idx, layer_params in value.items():
                block_key = f'block_{layer_idx}'
                translated[block_key] = layer_params
        elif key == 'lm_head':
            # Map lm_head → unembedding
            translated['unembedding'] = value
        elif key == 'norm':
            # Keep norm as-is
            translated['norm'] = value
        else:
            # Keep other keys as-is
            translated[key] = value

    return translated


class OrbaxWeightLoader:
    """Load weights from Orbax checkpoint format.

    Orbax checkpoints are pre-converted from SafeTensors and load much faster
    (5-6s vs 90s for SafeTensors + MXFP4 decompression).

    Usage:
        loader = OrbaxWeightLoader('/path/to/orbax/checkpoint')
        params = loader.load_params()
    """

    def __init__(self, checkpoint_path: str):
        """Initialize loader.

        Args:
            checkpoint_path: Path to Orbax checkpoint directory (should contain '0' subdirectory)
        """
        self.checkpoint_path = Path(checkpoint_path).resolve()  # Use absolute path for Orbax
        assert self.checkpoint_path.exists(), \
            f"Checkpoint path not found: {self.checkpoint_path}"

        # Load quantization metadata if present
        self.quantization_metadata = None
        quant_path = self.checkpoint_path / "_quantization_metadata.json"
        if quant_path.exists():
            with open(quant_path, 'r') as f:
                self.quantization_metadata = json.load(f)
                print(f"  Found quantization metadata: {len(self.quantization_metadata)} quantized parameters")

    def load_params(
        self,
        show_progress: bool = True,
        unpack_quantized: bool = False,
        validate_unpacking: bool = True
    ) -> Dict[str, Any]:
        """Load parameters from Orbax checkpoint.

        Args:
            show_progress: Show loading progress
            unpack_quantized: Automatically unpack MXFP4 quantized weights to float16
            validate_unpacking: Validate unpacking invariants (recommended)

        Returns:
            Parameter tree compatible with JAX/Flax models
        """
        import time

        # Check for state subdirectory (common Orbax structure)
        checkpoint_dir = self.checkpoint_path / "0"
        state_path = checkpoint_dir / "state"
        if state_path.exists() and (state_path / "_METADATA").exists():
            checkpoint_dir = state_path

        # Detect platform for optimized loading strategy
        platform = jax.default_backend()
        is_gpu = platform in ('gpu', 'cuda', 'rocm', 'tpu')
        target_device = jax.local_devices()[0]

        if show_progress:
            print(f"  Loading from: {checkpoint_dir}")
            print(f"  JAX platform: {platform}")
            print(f"  Target device: {target_device}")

        # Load checkpoint with device-agnostic sharding
        # This works on both Mac (CPU) and GPU by specifying target sharding
        checkpointer = ocp.PyTreeCheckpointer()

        # Get checkpoint metadata to understand structure
        try:
            t_load_start = time.time()

            # Read checkpoint metadata to get array shapes/dtypes
            ckpt_metadata = checkpointer.metadata(str(checkpoint_dir))

            if show_progress:
                print(f"  Building device-agnostic restore spec...")

            # Create a single-device sharding spec for all arrays
            # This tells Orbax to ignore saved CUDA sharding and use our local device
            from jax.sharding import SingleDeviceSharding

            def build_restore_args(tree):
                """Build restore args with single-device sharding for all arrays."""
                if isinstance(tree, dict):
                    return {k: build_restore_args(v) for k, v in tree.items()}
                # For leaf nodes (arrays), specify single-device sharding
                return ocp.ArrayRestoreArgs(sharding=SingleDeviceSharding(target_device))

            restore_args = build_restore_args(ckpt_metadata)

            if show_progress:
                print(f"  Restoring checkpoint from disk...", flush=True)

            # Restore with explicit sharding - this overrides the checkpoint's device info
            params = checkpointer.restore(str(checkpoint_dir), args=restore_args)

            t_load = time.time() - t_load_start
            if show_progress:
                print(f"  ✓ Orbax restore completed in {t_load:.2f}s", flush=True)

        except Exception as e:
            # Fallback for older Orbax versions or if metadata doesn't work
            if show_progress:
                print(f"  Note: Using direct restore (no sharding override)")
            t_load_start = time.time()
            params = checkpointer.restore(str(checkpoint_dir))
            t_load = time.time() - t_load_start
            if show_progress:
                print(f"  ✓ Direct restore completed in {t_load:.2f}s")

        # Platform-specific device placement
        if is_gpu:
            # GPU path: Use async device_put for fast host-to-device transfer
            # This leverages fast PCIe bandwidth on GPU systems
            if show_progress:
                print(f"  Transferring to GPU...", flush=True)

            t_transfer_start = time.time()

            # Use device_put with async semantics for faster host-to-device transfer
            params = jax.tree.map(
                lambda x: jax.device_put(x, device=target_device),
                params
            )

            # Block until transfer completes to get accurate timing
            jax.tree.map(
                lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
                params
            )

            t_transfer = time.time() - t_transfer_start
            if show_progress:
                print(f"  ✓ GPU transfer completed in {t_transfer:.2f}s", flush=True)
        else:
            # CPU path: Data is already on the host, no transfer needed
            # On macOS/CPU, arrays are already in the right place
            if show_progress:
                print(f"  ✓ Parameters loaded on CPU (no device transfer needed)")

        # Translate Orbax structure to JAX model structure
        if show_progress:
            print(f"  Translating parameter structure...")
        params = translate_orbax_to_model_structure(params)

        if show_progress:
            print(f"  ✓ Loaded {len(params)} top-level parameter groups")

            # Show size estimate
            def count_params(tree):
                if isinstance(tree, dict):
                    return sum(count_params(v) for v in tree.values())
                elif isinstance(tree, jnp.ndarray):
                    return tree.size
                return 0

            total_params = count_params(params)
            print(f"  ✓ Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")

        # Unpack quantized weights if requested
        if unpack_quantized and self.quantization_metadata:
            if show_progress:
                print(f"\n  Unpacking {len(self.quantization_metadata)} quantized parameters...")

            params, timing_info = unpack_quantized_param_tree(
                params,
                self.quantization_metadata,
                validate=validate_unpacking,
                show_progress=show_progress,
                parallel=True,
                backend='auto'
            )

            if show_progress:
                print(f"  ✓ Unpacked in {timing_info['total_time']:.2f}s (backend: {timing_info['backend']})")

        return params


def load_config_from_orbax(checkpoint_path: str) -> Dict[str, Any]:
    """Load model config from Orbax checkpoint directory.

    The Orbax conversion script (convert_checkpoint.py) saves config.json
    alongside the checkpoint data. This function reads that config to support
    both gpt-oss-20b (24 layers, 32 experts) and gpt-oss-120b (36 layers, 128 experts).

    Args:
        checkpoint_path: Path to Orbax checkpoint directory

    Returns:
        Dictionary with model configuration

    Raises:
        FileNotFoundError: If config.json is not found in checkpoint directory
    """
    config_path = Path(checkpoint_path) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found in Orbax checkpoint: {config_path}\n"
            f"If you converted this checkpoint with an older script, please re-convert using:\n"
            f"  python -m gpt_oss.jax.scripts.convert_checkpoint --input <safetensors_dir> --output {checkpoint_path}"
        )

    with open(config_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    """Test Orbax loader."""
    import time

    checkpoint_path = "../atsentia-orbax-to-jaxflaxmock/orbaxmodels/gpt-oss-20b"

    print("="*80)
    print("Testing Orbax Weight Loader")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}\n")

    print("Loading parameters...")
    t0 = time.time()
    loader = OrbaxWeightLoader(checkpoint_path)
    params = loader.load_params(show_progress=True)
    load_time = time.time() - t0

    print(f"\n✓ Loading completed in {load_time:.2f}s")
    print(f"  (Compare to SafeTensors: ~90s)")
    print("="*80)
