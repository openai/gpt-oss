"""TokenGenerator wrapper for JAX backend to match gpt_oss.generate interface."""

import json
from pathlib import Path
from typing import List, Iterator, Tuple, Optional, Union

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .model import Transformer
from .loader_safetensors import WeightLoader
from .loader_orbax import OrbaxWeightLoader, load_config_from_orbax
from .kv_cache import KVCache


def detect_checkpoint_format(checkpoint_path: Path) -> str:
    """Detect whether checkpoint is Orbax or SafeTensors format.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        'orbax' or 'safetensors'
    """
    # Check for Orbax structure
    orbax_markers = [
        checkpoint_path / "0" / "state" / "_METADATA",
        checkpoint_path / "0" / "_METADATA",
    ]
    for marker in orbax_markers:
        if marker.exists():
            return 'orbax'

    # Check for SafeTensors files
    if list(checkpoint_path.glob('*.safetensors')):
        return 'safetensors'

    # Default to SafeTensors
    return 'safetensors'


def load_config_from_checkpoint(checkpoint_path: Path) -> ModelConfig:
    """Load model configuration from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        ModelConfig instance
    """
    config_path = checkpoint_path / "config.json"

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return ModelConfig(
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


class TokenGenerator:
    """JAX token generator matching gpt_oss.generate interface.

    This class wraps the JAX/Flax implementation to provide a generator-based
    interface compatible with the existing torch and triton backends.
    """

    def __init__(self, checkpoint: str, max_context_length: int = 4096, force_cpu: bool = False):
        """Initialize JAX token generator.

        Args:
            checkpoint: Path to checkpoint directory (SafeTensors or Orbax format)
            max_context_length: Maximum context length for KV cache
            force_cpu: If True, force CPU execution even if GPU is available.
                      On macOS, this is automatically detected and not needed.
        """
        # Optionally force CPU execution (useful for testing or debugging)
        if force_cpu:
            jax.config.update('jax_platform_name', 'cpu')

        checkpoint_path = Path(checkpoint)

        # Detect checkpoint format first (before trying to load config)
        checkpoint_format = detect_checkpoint_format(checkpoint_path)
        print(f"Loading JAX checkpoint ({checkpoint_format} format)...")

        # Load configuration based on checkpoint format
        if checkpoint_format == 'orbax':
            # Orbax checkpoints typically don't have config.json
            config_dict = load_config_from_orbax(str(checkpoint_path))
            self.config = ModelConfig(**config_dict)
        else:
            # SafeTensors checkpoints have config.json
            self.config = load_config_from_checkpoint(checkpoint_path)

        self.max_context_length = max_context_length

        # Load weights based on checkpoint format
        if checkpoint_format == 'orbax':
            loader = OrbaxWeightLoader(str(checkpoint_path))
            self.params = loader.load_params(
                show_progress=False,
                unpack_quantized=True,
                validate_unpacking=False
            )
        else:
            loader = WeightLoader(str(checkpoint_path))
            self.params = loader.load_params(self.config, show_progress=False)

        print(f"Loaded {self.config.num_hidden_layers}-layer model with {self.config.num_experts} experts/layer")

        # Create model
        self.model = Transformer(config=self.config)

        # Initialize KV caches
        self.kv_caches = [
            KVCache.create(
                batch_size=1,
                max_ctx=max_context_length,
                n_kv_heads=self.config.num_key_value_heads,
                d_head=self.config.head_dim
            )
            for _ in range(self.config.num_hidden_layers)
        ]

        # Warmup model
        print("Compiling model (JAX XLA)...")
        self._warmup()
        print("Ready to generate")

    def _warmup(self):
        """Pre-compile model with dummy inputs to avoid first-token compilation delay."""
        # Warmup with prompt processing
        dummy_prompt = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        _, kv_caches = self.model.apply({'params': self.params}, dummy_prompt, self.kv_caches)

        # Warmup with single token
        dummy_token = jnp.array([6], dtype=jnp.int32)
        _ = self.model.apply({'params': self.params}, dummy_token, kv_caches)

    def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        return_logprobs: bool = False
    ) -> Iterator[Union[int, Tuple[int, float]]]:
        """Generate tokens autoregressively.

        Args:
            prompt_tokens: Initial prompt as list of token IDs
            stop_tokens: List of token IDs that stop generation
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum number of tokens to generate (None = unlimited)
            return_logprobs: If True, yield (token, logprob) tuples

        Yields:
            Generated token IDs (or (token, logprob) if return_logprobs=True)
        """
        # Import tokenizer to decode prompt
        from .tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
        prompt_text = tokenizer.decode(prompt_tokens)
        print(f"Prompt: {prompt_text}")

        # Reset KV caches for new generation
        self.kv_caches = [
            KVCache.create(
                batch_size=1,
                max_ctx=self.max_context_length,
                n_kv_heads=self.config.num_key_value_heads,
                d_head=self.config.head_dim
            )
            for _ in range(self.config.num_hidden_layers)
        ]

        tokens = list(prompt_tokens)
        num_generated_tokens = 0

        # Initialize RNG key for temperature sampling
        rng_key = jax.random.PRNGKey(42) if temperature > 0.0 else None

        # Process prompt (or full context without KV cache)
        first_forward = True

        while max_tokens is None or num_generated_tokens < max_tokens:
            # Prepare input
            if first_forward:
                # First forward pass: process all prompt tokens
                tokens_array = jnp.array(tokens, dtype=jnp.int32)
                first_forward = False
            else:
                # Subsequent passes: only process last token (use KV cache)
                tokens_array = jnp.array([tokens[-1]], dtype=jnp.int32)

            # Forward pass with KV caching
            logits, self.kv_caches = self.model.apply(
                {'params': self.params},
                tokens_array,
                self.kv_caches
            )

            # Get logits for next token prediction
            next_token_logits = logits[-1]  # [vocab_size]

            # Sample next token
            if temperature == 0.0:
                # Greedy sampling
                predicted_token = int(jnp.argmax(next_token_logits))
            else:
                # Temperature sampling
                rng_key, sample_key = jax.random.split(rng_key)
                scaled_logits = next_token_logits / temperature
                predicted_token = int(jax.random.categorical(sample_key, scaled_logits))

            tokens.append(predicted_token)
            num_generated_tokens += 1

            # Yield result
            if return_logprobs:
                # Compute log probabilities
                logprobs = jax.nn.log_softmax(next_token_logits)
                selected_logprob = float(logprobs[predicted_token])
                yield predicted_token, selected_logprob
            else:
                yield predicted_token

            # Check stop tokens
            if predicted_token in stop_tokens:
                break
