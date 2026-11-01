"""End-to-end inference for gpt-oss-20b JAX implementation.

This module provides token generation utilities including:
- Greedy sampling (argmax)
- Temperature sampling
- Top-k sampling
- Incremental generation with context management
- KV caching for efficient autoregressive generation
"""

import time
import jax
import jax.numpy as jnp
from typing import List, Optional, Callable, Dict, Any
from tqdm import tqdm

# Handle both module import and direct execution
try:
    from .model import Transformer
    from .config import ModelConfig
    from .kv_cache import KVCache
except ImportError:
    from model import Transformer
    from config import ModelConfig
    from kv_cache import KVCache


@jax.jit
def _sample_token_jit(
    logits: jax.Array,
    temperature: float,
    top_k: int,
    rng_key: jax.Array
) -> jax.Array:
    """JIT-compiled token sampling (internal helper).

    Args:
        logits: Logits for next token prediction, shape [vocab_size]
        temperature: Sampling temperature (0.0 = greedy)
        top_k: Number of top tokens to consider (0 = all tokens)
        rng_key: JAX random key

    Returns:
        Sampled token ID as jax.Array (scalar)
    """
    # Greedy vs temperature sampling using jax.lax.cond
    def greedy_sample(logits):
        return jnp.argmax(logits)

    def temperature_sample(logits):
        # Apply temperature scaling
        scaled_logits = logits / jnp.maximum(temperature, 1e-8)  # Avoid div by zero

        # Top-k filtering using jax.lax.cond
        def apply_top_k(scaled_logits):
            # Get top-k indices
            top_k_indices = jnp.argsort(scaled_logits)[-top_k:]
            # Create mask: -inf for non-top-k tokens
            mask = jnp.full_like(scaled_logits, -jnp.inf)
            mask = mask.at[top_k_indices].set(0.0)
            return scaled_logits + mask

        def no_top_k(scaled_logits):
            return scaled_logits

        # Apply top-k only if top_k > 0
        scaled_logits = jax.lax.cond(
            top_k > 0,
            apply_top_k,
            no_top_k,
            scaled_logits
        )

        # Sample from categorical distribution
        return jax.random.categorical(rng_key, scaled_logits)

    # Use jax.lax.cond for greedy vs temperature sampling
    token = jax.lax.cond(
        temperature == 0.0,
        greedy_sample,
        temperature_sample,
        logits
    )

    return token


def sample_token(
    logits: jax.Array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng_key: Optional[jax.Array] = None
) -> int:
    """Sample next token from logits.

    JIT-compiled for optimal performance using jax.lax.cond for control flow.

    Args:
        logits: Logits for next token prediction, shape [vocab_size]
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_k: If set, only sample from top-k most likely tokens
        rng_key: JAX random key (required if temperature > 0)

    Returns:
        Sampled token ID (int)

    Example:
        >>> logits = jnp.array([0.1, 0.8, 0.1])  # Token 1 most likely
        >>> token = sample_token(logits, temperature=0.0)  # Greedy
        >>> assert token == 1
    """
    assert logits.ndim == 1, \
        f"sample_token: logits must be 1D, got shape {logits.shape}"
    assert temperature >= 0.0, \
        f"sample_token: temperature must be non-negative, got {temperature}"

    # For temperature sampling, rng_key is required
    if temperature > 0.0 and rng_key is None:
        raise ValueError("sample_token: rng_key required for temperature sampling (temperature > 0)")

    # Use dummy rng_key for greedy sampling (won't be used)
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Convert top_k to int (0 means no top_k filtering)
    top_k_value = top_k if top_k is not None else 0

    # Call JIT-compiled helper
    token = _sample_token_jit(logits, temperature, top_k_value, rng_key)

    return int(token)


def _create_jit_generate_step(model: Transformer, use_kv_cache: bool):
    """Create a JIT-compiled generation step function.

    This returns a JIT-compiled function that avoids re-compiling on each call.
    The model is captured in the closure, not passed as an argument.

    Args:
        model: The Transformer model (captured in closure)
        use_kv_cache: Whether KV caching is enabled

    Returns:
        JIT-compiled function: (params, tokens, kv_caches) -> (logits, updated_caches)
    """
    if use_kv_cache:
        @jax.jit
        def jitted_step(params: dict, tokens_array: jax.Array, kv_caches: Optional[List[Any]]):
            return model.apply({'params': params}, tokens_array, kv_caches)
    else:
        @jax.jit
        def jitted_step(params: dict, tokens_array: jax.Array, kv_caches: Optional[List[Any]]):
            logits = model.apply({'params': params}, tokens_array)
            return logits, None

    return jitted_step


def generate(
    model: Transformer,
    params: dict,
    prompt_tokens: List[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng_key: Optional[jax.Array] = None,
    show_progress: bool = True,
    token_callback: Optional[Callable[[int], None]] = None,
    return_stats: bool = False,
    use_kv_cache: bool = True,
    config: Optional[Any] = None,
    jit_generate_loop: bool = False
) -> List[int] | tuple[List[int], Dict[str, Any]]:
    """Generate tokens autoregressively from prompt.

    Args:
        model: Transformer model instance
        params: Model parameters (from WeightLoader)
        prompt_tokens: Initial prompt as list of token IDs
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_k: If set, only sample from top-k tokens
        rng_key: JAX random key (required if temperature > 0)
        show_progress: Show tqdm progress bar
        token_callback: Optional callback called with each generated token
        return_stats: If True, return (tokens, stats) with timing information
        use_kv_cache: If True, use KV caching for efficient generation (default: True)
        config: Model config (required if use_kv_cache=True)
        jit_generate_loop: If True, use JIT-compiled generation step (experimental)

    Returns:
        If return_stats=False: Full sequence (prompt + generated tokens)
        If return_stats=True: Tuple of (full sequence, stats dict with timing info)

    Example:
        >>> from gpt_oss_jax.model import Transformer
        >>> from gpt_oss_jax.loader_safetensors import WeightLoader
        >>>
        >>> config = ModelConfig(...)
        >>> model = Transformer(config=config)
        >>> loader = WeightLoader('checkpoint/')
        >>> params = loader.load_params(config)
        >>>
        >>> prompt = [1, 2, 3, 4]  # Token IDs
        >>> key = jax.random.PRNGKey(42)
        >>> tokens = generate(model, params, prompt, max_new_tokens=10,
        ...                   temperature=0.8, rng_key=key, config=config)
        >>> print(f"Generated {len(tokens) - len(prompt)} new tokens")
    """
    assert len(prompt_tokens) > 0, \
        "generate: prompt_tokens must not be empty"
    assert max_new_tokens > 0, \
        f"generate: max_new_tokens must be positive, got {max_new_tokens}"
    assert temperature >= 0.0, \
        f"generate: temperature must be non-negative, got {temperature}"

    if temperature > 0.0:
        assert rng_key is not None, \
            "generate: rng_key required for temperature sampling (temperature > 0)"

    if use_kv_cache:
        assert config is not None, \
            "generate: config required when use_kv_cache=True"

    # Initialize with prompt
    current_tokens = list(prompt_tokens)

    # Initialize KV caches if enabled
    kv_caches = None
    if use_kv_cache:
        # Create one cache per layer
        kv_caches = [
            KVCache.create(
                batch_size=1,
                max_ctx=4096,  # TODO: Make this configurable
                n_kv_heads=config.num_key_value_heads,
                d_head=config.head_dim
            )
            for _ in range(config.num_hidden_layers)
        ]
        if show_progress:
            print(f"[KV Cache] Initialized {len(kv_caches)} caches, shape: {kv_caches[0].k.shape}")

    # Create JIT-compiled step function if requested
    if jit_generate_loop:
        if show_progress:
            print(f"[JIT] Compiling generation step (may take a few seconds)...")
            print(f"[JIT] WARNING: JIT mode is experimental and may not provide significant speedup on CPU")
            print(f"[JIT] Note: Assertions are disabled during JIT compilation (not compatible with jax.jit)")

        # Create JIT-compiled step function (model captured in closure)
        jitted_step = _create_jit_generate_step(model, use_kv_cache)

        # Trigger compilation with dummy input to avoid first-token slowdown
        # Note: We skip warmup for now due to assertion compatibility issues
        # The first token will trigger JIT compilation
        # TODO: Use jax.experimental.checkify for JIT-compatible assertions

        if show_progress:
            print(f"[JIT] ✓ JIT wrapper created (will compile on first token)")
    else:
        jitted_step = None

    # Timing stats
    token_times = []
    first_token_time = None
    total_start = time.time()

    # Progress bar
    iterator = range(max_new_tokens)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating", unit="tok")

    # Generate tokens one at a time
    for i in iterator:
        token_start = time.time()

        # Convert to JAX array
        t_array_start = time.time()
        if use_kv_cache and i > 0:
            # After first token, only process the new token
            tokens_array = jnp.array([current_tokens[-1]], dtype=jnp.int32)
        else:
            # First iteration: process full prompt (or no cache)
            tokens_array = jnp.array(current_tokens, dtype=jnp.int32)
        t_array = time.time() - t_array_start

        # Forward pass (use JIT-compiled version if enabled)
        t_forward_start = time.time()
        if jitted_step is not None:
            # Use JIT-compiled step
            logits, kv_caches = jitted_step(params, tokens_array, kv_caches)
        elif use_kv_cache:
            # Standard path with KV cache
            logits, kv_caches = model.apply({'params': params}, tokens_array, kv_caches)
        else:
            # Standard path without KV cache
            logits = model.apply({'params': params}, tokens_array)

        # OPTIMIZED: Only block for timing measurements on first few tokens
        # Blocking on every token adds overhead - JAX will handle synchronization naturally
        if i < 3:
            logits[-1].block_until_ready()
        t_forward = time.time() - t_forward_start

        token_time = time.time() - token_start

        # Log detailed timing for first few tokens
        if i < 3:
            if use_kv_cache:
                cache_info = f", cache_offset={kv_caches[0].offset}"
            else:
                cache_info = ""
            print(f"\n[Token {i}] Detailed timing{cache_info}:")
            print(f"  Input shape: {tokens_array.shape}")
            print(f"  Array creation: {t_array*1000:.2f}ms")
            print(f"  Forward pass: {t_forward:.2f}s")
            print(f"  Total token time: {token_time:.2f}s")

        # Get logits for last token (next token prediction)
        next_token_logits = logits[-1]  # [vocab_size]

        # Sample next token
        if temperature > 0.0:
            # Split RNG key for this sample
            rng_key, sample_key = jax.random.split(rng_key)
            next_token = sample_token(next_token_logits, temperature, top_k, sample_key)
        else:
            next_token = sample_token(next_token_logits, temperature=0.0)

        # Append to sequence
        current_tokens.append(next_token)

        # Track timing
        token_times.append(token_time)
        if i == 0:
            first_token_time = token_time

        # Callback (for streaming output, etc.)
        if token_callback is not None:
            token_callback(next_token)

        # Update progress bar description with last token and timing
        if show_progress:
            if i == 0:
                iterator.set_postfix(last_token=next_token, ttft=f"{first_token_time:.2f}s")
            else:
                avg_tok_per_sec = (i + 1) / sum(token_times)
                iterator.set_postfix(last_token=next_token, tok_s=f"{avg_tok_per_sec:.2f}")

    total_time = time.time() - total_start

    if return_stats:
        stats = {
            'total_time': total_time,
            'first_token_time': first_token_time,
            'subsequent_tokens_time': sum(token_times[1:]) if len(token_times) > 1 else 0.0,
            'num_tokens': len(token_times),
            'tokens_per_second': len(token_times) / total_time if total_time > 0 else 0.0,
            'tokens_per_second_after_first': (len(token_times) - 1) / sum(token_times[1:]) if len(token_times) > 1 and sum(token_times[1:]) > 0 else 0.0,
            'token_times': token_times,
        }
        return current_tokens, stats

    return current_tokens


def generate_greedy(
    model: Transformer,
    params: dict,
    prompt_tokens: List[int],
    max_new_tokens: int = 100,
    show_progress: bool = True,
    token_callback: Optional[Callable[[int], None]] = None,
    use_kv_cache: bool = True,
    config: Optional[Any] = None
) -> List[int]:
    """Generate tokens using greedy sampling (argmax).

    Convenience wrapper around generate() with temperature=0.0.

    Args:
        model: Transformer model instance
        params: Model parameters
        prompt_tokens: Initial prompt as list of token IDs
        max_new_tokens: Maximum number of tokens to generate
        show_progress: Show tqdm progress bar
        token_callback: Optional callback for each generated token
        use_kv_cache: If True, use KV caching (default: True)
        config: Model config (required if use_kv_cache=True)

    Returns:
        Full sequence: prompt + generated tokens
    """
    return generate(
        model=model,
        params=params,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        show_progress=show_progress,
        token_callback=token_callback,
        use_kv_cache=use_kv_cache,
        config=config
    )


if __name__ == "__main__":
    """Test inference with mock model."""
    import jax

    print("Testing inference utilities...")
    print("=" * 80)

    # Test 1: sample_token with greedy sampling
    print("\nTest 1: Greedy sampling (argmax)")
    print("-" * 80)

    logits = jnp.array([0.1, 0.8, 0.1])
    token = sample_token(logits, temperature=0.0)
    print(f"Logits: {logits}")
    print(f"Sampled token (greedy): {token}")
    assert token == 1, f"Expected token 1, got {token}"
    print("✓ Test 1 passed")

    # Test 2: sample_token with temperature sampling
    print("\nTest 2: Temperature sampling")
    print("-" * 80)

    key = jax.random.PRNGKey(42)
    logits = jnp.array([1.0, 2.0, 1.0])
    token = sample_token(logits, temperature=1.0, rng_key=key)
    print(f"Logits: {logits}")
    print(f"Sampled token (temp=1.0): {token}")
    assert 0 <= token <= 2, f"Token {token} out of range"
    print("✓ Test 2 passed")

    # Test 3: sample_token with top-k
    print("\nTest 3: Top-k sampling")
    print("-" * 80)

    key = jax.random.PRNGKey(43)
    logits = jnp.array([0.1, 0.2, 0.3, 0.4])
    token = sample_token(logits, temperature=1.0, top_k=2, rng_key=key)
    print(f"Logits: {logits}")
    print(f"Sampled token (top_k=2): {token}")
    # Should be one of top-2: indices 2 or 3
    assert token in [2, 3], f"Token {token} not in top-2"
    print("✓ Test 3 passed")

    # Test 4: generate with small model
    print("\nTest 4: Generation with small model")
    print("-" * 80)

    config = ModelConfig(
        num_hidden_layers=2,
        hidden_size=128,
        head_dim=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        sliding_window=4,
        intermediate_size=256,
        num_experts=4,
        experts_per_token=2,
        vocab_size=100,
        swiglu_limit=7.0,
        rope_theta=150000.0,
        rope_scaling_factor=1.0,
        rope_ntk_alpha=1.0,
        rope_ntk_beta=32.0,
        initial_context_length=4096,
    )

    model = Transformer(config=config)
    key = jax.random.PRNGKey(44)
    prompt = [1, 2, 3, 4]

    # Initialize model
    init_key = jax.random.PRNGKey(45)
    params = model.init(init_key, jnp.array(prompt, dtype=jnp.int32))

    # Generate (greedy, no progress bar for test)
    tokens = generate_greedy(model, params['params'], prompt, max_new_tokens=5, show_progress=False)

    print(f"Prompt: {prompt}")
    print(f"Generated: {tokens}")
    print(f"Generated {len(tokens) - len(prompt)} new tokens")

    assert len(tokens) == len(prompt) + 5, \
        f"Expected {len(prompt) + 5} tokens, got {len(tokens)}"
    assert tokens[:len(prompt)] == prompt, \
        "Generated tokens should start with prompt"
    print("✓ Test 4 passed")

    print("\n" + "=" * 80)
    print("All inference tests passed!")
