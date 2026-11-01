#!/usr/bin/env python3
"""Convert SafeTensors checkpoint to Orbax format for faster JAX loading.

This script converts gpt-oss weights from SafeTensors format (with MXFP4 quantization)
to Orbax format in BF16. This provides ~18x faster loading (5s vs 90s).

Usage:
    python -m gpt_oss.jax.safetensor2orbax \\
        --input gpt-oss-20b/original/ \\
        --output gpt-oss-20b-orbax/

The conversion only needs to be done once. After conversion, use the Orbax checkpoint
for faster inference startup times.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import jax.numpy as jnp
import orbax.checkpoint as ocp

from ..config import ModelConfig
from ..loader_safetensors import WeightLoader


def verify_params_structure(params: Dict[str, Any], config: ModelConfig) -> bool:
    """Verify parameter structure matches JAX model expectations.

    Args:
        params: Parameter tree from WeightLoader
        config: Model configuration

    Returns:
        True if structure is valid

    Raises:
        AssertionError if structure is invalid
    """
    # Check embedding
    assert 'embedding' in params, "Missing 'embedding' in params"
    assert 'embedding' in params['embedding'], "Missing 'embedding.embedding'"

    # Check blocks
    for i in range(config.num_hidden_layers):
        block_name = f'block_{i}'
        assert block_name in params, f"Missing '{block_name}'"

        # Check attention
        assert 'attn' in params[block_name], f"Missing '{block_name}.attn'"
        assert 'norm' in params[block_name]['attn'], f"Missing '{block_name}.attn.norm'"
        assert 'qkv' in params[block_name]['attn'], f"Missing '{block_name}.attn.qkv'"
        assert 'out' in params[block_name]['attn'], f"Missing '{block_name}.attn.out'"
        assert 'sinks' in params[block_name]['attn'], f"Missing '{block_name}.attn.sinks'"

        # Check MLP
        assert 'mlp' in params[block_name], f"Missing '{block_name}.mlp'"
        assert 'norm' in params[block_name]['mlp'], f"Missing '{block_name}.mlp.norm'"
        assert 'gate' in params[block_name]['mlp'], f"Missing '{block_name}.mlp.gate'"
        assert 'mlp1_weight' in params[block_name]['mlp'], f"Missing '{block_name}.mlp.mlp1_weight'"
        assert 'mlp2_weight' in params[block_name]['mlp'], f"Missing '{block_name}.mlp.mlp2_weight'"

    # Check final layers
    assert 'norm' in params, "Missing 'norm'"
    assert 'unembedding' in params, "Missing 'unembedding'"

    return True


def convert_checkpoint(
    safetensors_path: str,
    output_path: str,
    config: ModelConfig,
    show_progress: bool = True
):
    """Convert SafeTensors checkpoint to Orbax format.

    Args:
        safetensors_path: Path to SafeTensors checkpoint directory
        output_path: Path to output Orbax checkpoint directory
        config: Model configuration
        show_progress: Show conversion progress
    """
    safetensors_path = Path(safetensors_path).resolve()  # Absolute path
    output_path = Path(output_path).resolve()  # Absolute path for Orbax

    assert safetensors_path.exists(), \
        f"SafeTensors checkpoint not found: {safetensors_path}"

    if show_progress:
        print("="*80)
        print("SafeTensors → Orbax Checkpoint Converter")
        print("="*80)
        print(f"Input:  {safetensors_path}")
        print(f"Output: {output_path}")
        print()

    # Step 1: Load SafeTensors weights (with MXFP4 decompression)
    if show_progress:
        print("[1/3] Loading SafeTensors checkpoint...")

    t0 = time.time()
    loader = WeightLoader(str(safetensors_path))
    safetensors_weights = loader.load_params(config, show_progress=show_progress)
    load_time = time.time() - t0

    if show_progress:
        print(f"  ✓ Loaded in {load_time:.2f}s")
        print()

    # Step 2: Verify structure
    if show_progress:
        print("[2/3] Verifying parameter structure...")

    t0 = time.time()
    verify_params_structure(safetensors_weights, config)
    params = safetensors_weights  # Already in correct format!
    verify_time = time.time() - t0

    if show_progress:
        print(f"  ✓ Structure verified in {verify_time:.2f}s")
        print(f"  ✓ Parameter tree already matches JAX model (no mapping needed)")
        print()

    # Step 3: Save as Orbax checkpoint
    if show_progress:
        print("[3/3] Saving Orbax checkpoint...")

    t0 = time.time()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Orbax expects checkpoint in subdirectory "0/state"
    checkpoint_dir = output_path / "0" / "state"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint (force=True to overwrite if exists)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(checkpoint_dir), params, force=True)

    save_time = time.time() - t0

    if show_progress:
        print(f"  ✓ Saved in {save_time:.2f}s")
        print()

    # Save config for reference
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "head_dim": config.head_dim,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "sliding_window": config.sliding_window,
            "intermediate_size": config.intermediate_size,
            "num_experts": config.num_experts,
            "experts_per_token": config.experts_per_token,
            "vocab_size": config.vocab_size,
            "swiglu_limit": config.swiglu_limit,
            "rope_theta": config.rope_theta,
            "rope_scaling_factor": config.rope_scaling_factor,
            "rope_ntk_alpha": config.rope_ntk_alpha,
            "rope_ntk_beta": config.rope_ntk_beta,
            "initial_context_length": config.initial_context_length,
        }, f, indent=2)

    if show_progress:
        print("="*80)
        print("Conversion complete!")
        print("="*80)
        print(f"Total time: {load_time + verify_time + save_time:.2f}s")
        print(f"  - Loading SafeTensors: {load_time:.2f}s")
        print(f"  - Verifying structure: {verify_time:.2f}s")
        print(f"  - Saving Orbax: {save_time:.2f}s")
        print()
        print(f"Orbax checkpoint saved to: {output_path}")
        print()
        print("You can now use this checkpoint for faster inference:")
        print(f"  python -m gpt_oss.generate --backend jax {output_path}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SafeTensors checkpoint to Orbax format for JAX inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert gpt-oss-20b checkpoint
  python -m gpt_oss.jax.safetensor2orbax \\
      --input gpt-oss-20b/original/ \\
      --output gpt-oss-20b-orbax/

  # Quiet mode
  python -m gpt_oss.jax.safetensor2orbax \\
      --input gpt-oss-20b/original/ \\
      --output gpt-oss-20b-orbax/ \\
      --no-progress
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to SafeTensors checkpoint directory (e.g., gpt-oss-20b/original/)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output Orbax checkpoint directory (e.g., gpt-oss-20b-orbax/)"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output"
    )

    args = parser.parse_args()

    # Load config from SafeTensors checkpoint
    config_path = Path(args.input) / "config.json"
    assert config_path.exists(), \
        f"Config file not found: {config_path}"

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = ModelConfig(
        num_hidden_layers=config_dict["num_hidden_layers"],
        hidden_size=config_dict["hidden_size"],
        head_dim=config_dict.get("head_dim", 64),
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        sliding_window=config_dict.get("sliding_window", 128),
        intermediate_size=config_dict["intermediate_size"],
        num_experts=config_dict["num_experts"],
        experts_per_token=config_dict["experts_per_token"],
        vocab_size=config_dict["vocab_size"],
        swiglu_limit=config_dict.get("swiglu_limit", 7.0),
        rope_theta=config_dict["rope_theta"],
        rope_scaling_factor=config_dict.get("rope_scaling_factor", 1.0),
        rope_ntk_alpha=config_dict.get("rope_ntk_alpha", 1.0),
        rope_ntk_beta=config_dict.get("rope_ntk_beta", 32.0),
        initial_context_length=config_dict.get("initial_context_length", 4096),
    )

    # Convert checkpoint
    convert_checkpoint(
        safetensors_path=args.input,
        output_path=args.output,
        config=config,
        show_progress=not args.no_progress
    )


if __name__ == "__main__":
    main()
