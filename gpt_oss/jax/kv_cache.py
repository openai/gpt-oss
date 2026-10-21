"""KV Cache implementation for efficient autoregressive generation.

Port of the PyTorch Triton implementation to JAX.
Reference: gpt_oss/triton/model.py:121-154
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class


@jax.jit
def _extend_jit(
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    k_new: jnp.ndarray,
    v_new: jnp.ndarray,
    offset: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled KV cache extension (internal helper).

    Uses jax.lax.dynamic_update_slice for JIT-compatible dynamic indexing.

    Args:
        k_cache: Current key cache [batch, max_ctx, n_heads, d_head]
        v_cache: Current value cache [batch, max_ctx, n_heads, d_head]
        k_new: New keys to add [batch, n_new, n_heads, d_head]
        v_new: New values to add [batch, n_new, n_heads, d_head]
        offset: Current cache offset

    Returns:
        Tuple of (new_k_cache, new_v_cache)
    """
    # Update cache using dynamic_update_slice (JIT-compatible)
    # Start indices: [0, offset, 0, 0] for [batch, tokens, heads, head_dim]
    new_k = jax.lax.dynamic_update_slice(k_cache, k_new, (0, offset, 0, 0))
    new_v = jax.lax.dynamic_update_slice(v_cache, v_new, (0, offset, 0, 0))

    return new_k, new_v


@register_pytree_node_class
@dataclass
class KVCache:
    """Key-Value cache for efficient autoregressive generation.

    Stores previously computed key and value tensors to avoid recomputation
    during autoregressive generation. This provides:
    - O(n) complexity instead of O(n²)
    - Constant input shape (always 1 new token) → no recompilation

    Registered as a JAX PyTree to enable JIT compilation with KV caches.

    Attributes:
        k: Key cache of shape [batch_size, max_ctx, n_kv_heads, d_head]
        v: Value cache of shape [batch_size, max_ctx, n_kv_heads, d_head]
        offset: Current position in cache (number of tokens stored)

    Example:
        >>> cache = KVCache.create(batch_size=1, max_ctx=4096, n_kv_heads=8, d_head=64)
        >>> # First forward pass with prompt
        >>> k, v = cache.extend(k_prompt, v_prompt)  # k, v shape: [batch, 4, 8, 64]
        >>> # Subsequent single-token passes
        >>> k, v = cache.extend(k_new, v_new)  # k_new shape: [batch, 1, 8, 64]
    """
    k: jnp.ndarray  # [batch_size, max_ctx, n_kv_heads, d_head]
    v: jnp.ndarray  # [batch_size, max_ctx, n_kv_heads, d_head]
    offset: int

    def tree_flatten(self):
        """Flatten KVCache into children (arrays) and auxiliary data (offset).

        Required for JAX PyTree registration.
        """
        children = (self.k, self.v)
        aux_data = self.offset
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct KVCache from flattened representation.

        Required for JAX PyTree registration.
        """
        k, v = children
        offset = aux_data
        return cls(k=k, v=v, offset=offset)

    @staticmethod
    def create(batch_size: int, max_ctx: int, n_kv_heads: int, d_head: int = 64) -> 'KVCache':
        """Create a new KV cache with zeros.

        Args:
            batch_size: Batch size (typically 1 for inference)
            max_ctx: Maximum context length (e.g., 4096)
            n_kv_heads: Number of key-value heads (for GQA)
            d_head: Head dimension (default: 64)

        Returns:
            New KVCache instance initialized with zeros

        Example:
            >>> cache = KVCache.create(batch_size=1, max_ctx=4096, n_kv_heads=8)
            >>> print(cache.k.shape)  # (1, 4096, 8, 64)
        """
        assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
        assert max_ctx > 0, f"max_ctx must be positive, got {max_ctx}"
        assert n_kv_heads > 0, f"n_kv_heads must be positive, got {n_kv_heads}"
        assert d_head > 0, f"d_head must be positive, got {d_head}"

        k = jnp.zeros((batch_size, max_ctx, n_kv_heads, d_head), dtype=jnp.bfloat16)
        v = jnp.zeros((batch_size, max_ctx, n_kv_heads, d_head), dtype=jnp.bfloat16)

        return KVCache(k=k, v=v, offset=0)

    def reset(self) -> 'KVCache':
        """Reset cache to zeros and offset to 0.

        Returns:
            New KVCache with reset values

        Example:
            >>> cache = cache.extend(k1, v1).extend(k2, v2)
            >>> cache.offset  # 2
            >>> cache = cache.reset()
            >>> cache.offset  # 0
        """
        return KVCache(
            k=jnp.zeros_like(self.k),
            v=jnp.zeros_like(self.v),
            offset=0
        )

    def extend(self, k_new: jnp.ndarray, v_new: jnp.ndarray) -> Tuple['KVCache', jnp.ndarray, jnp.ndarray]:
        """Append new key-value pairs to cache and return full cache.

        JIT-compiled for optimal performance.

        Args:
            k_new: New keys of shape [batch_size, n_new_tokens, n_kv_heads, d_head]
            v_new: New values of shape [batch_size, n_new_tokens, n_kv_heads, d_head]

        Returns:
            Tuple of (updated_cache, full_k, full_v) where:
            - updated_cache: New KVCache with incremented offset
            - full_k: Full key cache [:, :offset+n_new, :, :]
            - full_v: Full value cache [:, :offset+n_new, :, :]

        Raises:
            AssertionError: If shapes are incompatible or cache overflow

        Example:
            >>> cache = KVCache.create(1, 4096, 8, 64)
            >>> # Add prompt tokens (4 tokens)
            >>> cache, k_full, v_full = cache.extend(k_prompt, v_prompt)
            >>> k_full.shape  # (1, 4, 8, 64) - only up to offset
            >>> cache.offset  # 4
            >>> # Add one new token
            >>> cache, k_full, v_full = cache.extend(k_new, v_new)
            >>> k_full.shape  # (1, 5, 8, 64)
            >>> cache.offset  # 5
        """
        # Input validation
        assert k_new.ndim == 4, \
            f"k_new must be 4D [batch, n_tokens, n_heads, d_head], got {k_new.ndim}D: {k_new.shape}"
        assert v_new.ndim == 4, \
            f"v_new must be 4D [batch, n_tokens, n_heads, d_head], got {v_new.ndim}D: {v_new.shape}"
        assert k_new.shape == v_new.shape, \
            f"k_new and v_new shapes must match: {k_new.shape} vs {v_new.shape}"

        batch_size, n_new_tokens, n_kv_heads, d_head = k_new.shape

        assert batch_size == self.k.shape[0], \
            f"Batch size mismatch: cache {self.k.shape[0]} vs new {batch_size}"
        assert n_kv_heads == self.k.shape[2], \
            f"n_kv_heads mismatch: cache {self.k.shape[2]} vs new {n_kv_heads}"
        assert d_head == self.k.shape[3], \
            f"d_head mismatch: cache {self.k.shape[3]} vs new {d_head}"
        assert self.offset + n_new_tokens <= self.k.shape[1], \
            f"Cache overflow: offset {self.offset} + {n_new_tokens} > max_ctx {self.k.shape[1]}"

        # Call JIT-compiled helper for cache update
        new_k, new_v = _extend_jit(
            self.k, self.v, k_new, v_new, self.offset
        )

        # Calculate new offset (outside JIT)
        new_offset = self.offset + n_new_tokens

        # Create new cache with updated values
        new_cache = KVCache(k=new_k, v=new_v, offset=new_offset)

        # Return full K/V up to current offset (slicing done outside JIT)
        k_full = new_k[:, :new_cache.offset, :, :]
        v_full = new_v[:, :new_cache.offset, :, :]

        return new_cache, k_full, v_full

    def truncate(self, n_ctx: int) -> 'KVCache':
        """Truncate cache to first n_ctx tokens.

        Args:
            n_ctx: Number of tokens to keep (must be <= current offset)

        Returns:
            New KVCache truncated to n_ctx tokens

        Raises:
            AssertionError: If n_ctx > max_ctx or n_ctx < 0

        Example:
            >>> cache = KVCache.create(1, 4096, 8, 64)
            >>> cache, k, v = cache.extend(k_10_tokens, v_10_tokens)
            >>> cache.offset  # 10
            >>> cache = cache.truncate(5)
            >>> cache.offset  # 5
        """
        assert 0 <= n_ctx <= self.k.shape[1], \
            f"n_ctx must be in [0, {self.k.shape[1]}], got {n_ctx}"

        # Zero out everything after n_ctx
        new_k = self.k.at[:, n_ctx:, :, :].set(0.0)
        new_v = self.v.at[:, n_ctx:, :, :].set(0.0)

        return KVCache(k=new_k, v=new_v, offset=n_ctx)


if __name__ == "__main__":
    """Test KVCache implementation."""
    import jax

    print("Testing KVCache...")
    print("=" * 80)

    # Test 1: Create cache
    print("\nTest 1: Create cache")
    print("-" * 80)
    cache = KVCache.create(batch_size=1, max_ctx=10, n_kv_heads=2, d_head=4)
    print(f"Cache k shape: {cache.k.shape}")
    print(f"Cache v shape: {cache.v.shape}")
    print(f"Initial offset: {cache.offset}")
    assert cache.k.shape == (1, 10, 2, 4)
    assert cache.v.shape == (1, 10, 2, 4)
    assert cache.offset == 0
    print("✓ Test 1 passed")

    # Test 2: Extend with prompt tokens
    print("\nTest 2: Extend with prompt tokens")
    print("-" * 80)
    key = jax.random.PRNGKey(42)
    k_prompt = jax.random.normal(key, (1, 3, 2, 4)).astype(jnp.bfloat16)
    v_prompt = jax.random.normal(key, (1, 3, 2, 4)).astype(jnp.bfloat16)

    cache, k_full, v_full = cache.extend(k_prompt, v_prompt)
    print(f"After extending with 3 tokens:")
    print(f"  cache.offset: {cache.offset}")
    print(f"  k_full.shape: {k_full.shape}")
    print(f"  v_full.shape: {v_full.shape}")
    assert cache.offset == 3
    assert k_full.shape == (1, 3, 2, 4)
    assert v_full.shape == (1, 3, 2, 4)
    print("✓ Test 2 passed")

    # Test 3: Extend with single token
    print("\nTest 3: Extend with single token")
    print("-" * 80)
    k_new = jax.random.normal(key, (1, 1, 2, 4)).astype(jnp.bfloat16)
    v_new = jax.random.normal(key, (1, 1, 2, 4)).astype(jnp.bfloat16)

    cache, k_full, v_full = cache.extend(k_new, v_new)
    print(f"After extending with 1 token:")
    print(f"  cache.offset: {cache.offset}")
    print(f"  k_full.shape: {k_full.shape}")
    assert cache.offset == 4
    assert k_full.shape == (1, 4, 2, 4)
    print("✓ Test 3 passed")

    # Test 4: Reset cache
    print("\nTest 4: Reset cache")
    print("-" * 80)
    cache = cache.reset()
    print(f"After reset:")
    print(f"  cache.offset: {cache.offset}")
    print(f"  k all zeros: {jnp.all(cache.k == 0)}")
    print(f"  v all zeros: {jnp.all(cache.v == 0)}")
    assert cache.offset == 0
    assert jnp.all(cache.k == 0)
    assert jnp.all(cache.v == 0)
    print("✓ Test 4 passed")

    # Test 5: Truncate cache
    print("\nTest 5: Truncate cache")
    print("-" * 80)
    cache, _, _ = cache.extend(k_prompt, v_prompt)  # Add 3 tokens
    cache, _, _ = cache.extend(k_new, v_new)  # Add 1 token (total: 4)
    print(f"Before truncate: offset = {cache.offset}")

    cache = cache.truncate(2)
    print(f"After truncate(2): offset = {cache.offset}")
    assert cache.offset == 2
    # Check that positions 2+ are zero
    assert jnp.all(cache.k[:, 2:, :, :] == 0)
    assert jnp.all(cache.v[:, 2:, :, :] == 0)
    print("✓ Test 5 passed")

    print("\n" + "=" * 80)
    print("All KVCache tests passed!")
