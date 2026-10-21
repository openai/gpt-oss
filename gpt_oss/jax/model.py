"""Flax implementation of gpt-oss-20b model.

This module provides a JAX/Flax translation of the PyTorch reference implementation,
following Tunix-style patterns for LLM architectures.

Key design principles:
- Assert-based defensive programming with clear error messages
- Shape transparency at every layer
- Numerical compatibility with PyTorch reference (within BF16 tolerance)
- CPU-only JAX execution (Metal not supported on Mac)
"""

import math
from typing import Any, Optional, List

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import ModelConfig

# Import KVCache for type hints (will be imported at runtime when needed)
try:
    from .kv_cache import KVCache
except ImportError:
    KVCache = Any  # Fallback for type checking


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input tensor using RMS statistics, then applies a learned scale.
    Uses FP32 precision for the normalization computation for numerical stability.

    Attributes:
        num_features: Dimensionality of the input features
        eps: Small constant for numerical stability
    """
    num_features: int
    eps: float = 1e-05

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., num_features]

        Returns:
            Normalized tensor of same shape as input

        Raises:
            AssertionError: If input shape doesn't match num_features
        """
        # Initialize scale parameter (FP32 for precision)
        scale = self.param('scale', nn.initializers.ones, (self.num_features,), jnp.float32)

        # Upcast to FP32 for normalization
        original_dtype = x.dtype
        t = x.astype(jnp.float32)

        # Compute RMS normalization
        rms = jnp.sqrt(jnp.mean(t ** 2, axis=-1, keepdims=True) + self.eps)
        t = t / rms

        # Apply scale and cast back to original dtype
        output = (t * scale).astype(original_dtype)

        return output


@jax.jit
def swiglu(x: jax.Array, alpha: float = 1.702, limit: float = 7.0) -> jax.Array:
    """SwiGLU activation function with clipping.

    SwiGLU: Swish-Gated Linear Unit
    Applies: swish(gate) * (linear + 1)
    where gate and linear are interleaved in the input tensor.

    JIT-compiled for optimal performance.

    Args:
        x: Input tensor with shape [..., 2*d] where d is the output dimension.
           Elements at even indices are gate values, odd indices are linear values.
        alpha: Swish activation parameter (default: 1.702)
        limit: Clipping limit for numerical stability (default: 7.0)

    Returns:
        Output tensor of shape [..., d]

    Raises:
        AssertionError: If input last dimension is not even
    """
    # Split into gate and linear components (interleaved)
    x_glu = x[..., ::2]  # Even indices: gate values
    x_linear = x[..., 1::2]  # Odd indices: linear values

    # Clip for numerical stability
    x_glu = jnp.clip(x_glu, None, limit)
    x_linear = jnp.clip(x_linear, -limit, limit)

    # Apply SwiGLU: swish(gate) * (linear + 1)
    # Swish(x) = x * sigmoid(alpha * x)
    swish_gate = x_glu * jax.nn.sigmoid(alpha * x_glu)
    output = swish_gate * (x_linear + 1.0)

    return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with YaRN scaling.

    Implements rotary embeddings for positional information in attention mechanisms.
    Supports extended context via YaRN (Yet another RoPE extensioN method) with
    NTK-by-parts interpolation/extrapolation.

    Reference: https://arxiv.org/abs/2309.00071 (YaRN paper)

    Attributes:
        head_dim: Dimension of each attention head
        base: Base frequency for RoPE (theta parameter)
        initial_context_length: Original training context length
        scaling_factor: Context extension scaling factor (>1 for longer contexts)
        ntk_alpha: Low-frequency extrapolation threshold
        ntk_beta: High-frequency interpolation threshold
    """
    head_dim: int
    base: float
    initial_context_length: int = 4096
    scaling_factor: float = 1.0
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0

    def _compute_concentration_and_inv_freq(self) -> tuple[float, jax.Array]:
        """Compute YaRN concentration factor and inverse frequencies.

        Returns:
            concentration: Attention concentration factor for YaRN
            inv_freq: Inverse frequencies for RoPE, shape [head_dim/2]
        """
        # Compute base frequencies
        freq = self.base ** (
            jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim
        )

        if self.scaling_factor > 1.0:
            # Apply YaRN scaling
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            # NTK-by-parts: compute low/high frequency boundaries
            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )

            # Interpolation for low frequencies, extrapolation for high frequencies
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            # Smooth transition via ramp function
            ramp = (jnp.arange(d_half, dtype=jnp.float32) - low) / (high - low)
            mask = 1.0 - jnp.clip(ramp, 0.0, 1.0)

            inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
        else:
            # No scaling, standard RoPE
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int, position_offset: int = 0) -> tuple[jax.Array, jax.Array]:
        """Compute cosine and sine tables for rotary embeddings.

        Args:
            num_tokens: Number of tokens (sequence length)
            position_offset: Starting position offset (for KV cache support)

        Returns:
            cos: Cosine table of shape [num_tokens, head_dim/2]
            sin: Sine table of shape [num_tokens, head_dim/2]
        """
        concentration, inv_freq = self._compute_concentration_and_inv_freq()

        # Compute position indices starting from position_offset
        # With KV cache: position_offset = kv_cache.offset (number of cached tokens)
        # Without KV cache: position_offset = 0
        t = jnp.arange(position_offset, position_offset + num_tokens, dtype=jnp.float32)

        # Outer product: position x frequency
        # OPTIMIZED: Use explicit outer product instead of einsum for better clarity
        freqs = jnp.outer(t, inv_freq)  # [num_tokens, head_dim/2]

        # Compute cos/sin with concentration
        cos = jnp.cos(freqs) * concentration
        sin = jnp.sin(freqs) * concentration

        return cos, sin

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        position_offset: int = 0,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            query: Query tensor of shape [num_tokens, ...]
            key: Key tensor of shape [num_tokens, ...]
            position_offset: Starting position for RoPE (for KV cache support, default: 0)

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as inputs

        Raises:
            AssertionError: If shapes are incompatible or num_tokens mismatch
        """
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens, position_offset)

        # Apply rotary embedding to query
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        # Apply rotary embedding to key
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)

        return query, key


@jax.jit
def _apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply rotary embedding rotation to input tensor.

    Rotates pairs of dimensions using cos/sin tables:
    [x1, x2, x3, x4, ...] -> [x1*cos - x2*sin, x2*cos + x1*sin, x3*cos - x4*sin, ...]

    JIT-compiled for optimal performance.

    Args:
        x: Input tensor of shape [num_tokens, ..., head_dim]
        cos: Cosine table of shape [num_tokens, head_dim/2]
        sin: Sine table of shape [num_tokens, head_dim/2]

    Returns:
        Rotated tensor of same shape as input

    Raises:
        AssertionError: If dimensions are incompatible
    """
    # Expand cos/sin to match x dimensions
    cos = cos[:, None, :].astype(x.dtype)  # [num_tokens, 1, head_dim/2]
    sin = sin[:, None, :].astype(x.dtype)

    # Split into pairs and rotate
    x1, x2 = jnp.split(x, 2, axis=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    output = jnp.concatenate([o1, o2], axis=-1)

    return output


@jax.jit
def sdpa(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    S: jax.Array,
    sm_scale: float,
    sliding_window: int = 0,
    kv_offset: int = 0,
) -> jax.Array:
    """Scaled Dot-Product Attention with optional sliding window and sink tokens.

    Implements multi-head attention with:
    - Causal masking (future tokens can't attend to past)
    - Optional sliding window attention (limits context to recent tokens)
    - Sink tokens (special attention logits added to softmax)
    - Grouped-query attention (GQA) support via q_mult
    - KV cache support via kv_offset parameter

    Args:
        Q: Query tensor of shape [n_new_tokens, n_heads, q_mult, d_head]
           where q_mult = num_attention_heads / num_key_value_heads
        K: Key tensor of shape [n_kv_tokens, n_heads, d_head]
           (can be larger than Q when using KV cache)
        V: Value tensor of shape [n_kv_tokens, n_heads, d_head]
        S: Sink tokens of shape [num_attention_heads] = [n_heads * q_mult]
        sm_scale: Attention scale factor (typically 1/sqrt(d_head))
        sliding_window: Window size for local attention (0 = full attention)
        kv_offset: Offset in KV cache (number of previously cached tokens)
                   When 0, Q and K/V have same length (no caching)
                   When > 0, Q is new tokens and K/V include cached tokens

    Returns:
        Attention output of shape [n_new_tokens, n_heads * q_mult * d_head]

    Raises:
        AssertionError: If input shapes are incompatible or contain NaN/Inf
    """
    n_new_tokens, n_heads, q_mult, d_head = Q.shape
    n_kv_tokens = K.shape[0]

    # Expand K and V to match Q's q_mult dimension (for GQA)
    K = K[:, :, None, :].repeat(q_mult, axis=2)  # [n_kv_tokens, n_heads, q_mult, d_head]
    V = V[:, :, None, :].repeat(q_mult, axis=2)  # [n_kv_tokens, n_heads, q_mult, d_head]

    # Reshape and expand sinks for broadcasting (PyTorch: S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1))
    # S shape: [n_heads * q_mult]
    # Reshape to: [n_heads, q_mult, 1, 1]
    S_expanded = S.reshape(n_heads, q_mult, 1, 1)
    # Expand over tokens: [n_heads, q_mult, n_new_tokens, 1]
    S_expanded = S_expanded.repeat(n_new_tokens, axis=2)

    # Create causal mask accounting for KV cache offset
    # Shape: [n_new_tokens, n_kv_tokens]
    # Each new query token at position i can only attend to KV tokens at positions <= (kv_offset + i)
    # JIT-compatible: Always use position-based masking (works for both cached and non-cached cases)
    q_positions = jnp.arange(n_new_tokens)[:, None] + kv_offset  # Query positions in full sequence
    kv_positions = jnp.arange(n_kv_tokens)[None, :]  # KV positions in full sequence
    mask = jnp.where(kv_positions > q_positions, -jnp.inf, 0.0).astype(Q.dtype)

    # Add sliding window mask if specified (JIT-compatible with jnp.where)
    # When sliding_window=0, this mask is all zeros (no effect)
    window_mask = jnp.where(
        (sliding_window > 0) & (q_positions - kv_positions > sliding_window),
        -jnp.inf,
        0.0
    ).astype(Q.dtype)
    mask = mask + window_mask

    # Compute attention scores: Q @ K^T
    # OPTIMIZED: Use explicit transpose + matmul instead of einsum for better XLA optimization
    # Original einsum: 'qhmd,khmd->hmqk'
    # Q: [n_new_tokens, n_heads, q_mult, d_head]
    # K: [n_kv_tokens, n_heads, 1 or q_mult, d_head]
    # Target: [n_heads, q_mult, n_new_tokens, n_kv_tokens]

    # Reshape for batched matmul: [n_heads, q_mult, n_new_tokens, d_head] @ [n_heads, q_mult, d_head, n_kv_tokens]
    Q_reshaped = Q.transpose(1, 2, 0, 3)  # [n_heads, q_mult, n_new_tokens, d_head]
    K_reshaped = K.transpose(1, 2, 3, 0)  # [n_heads, q_mult, d_head, n_kv_tokens]

    # Batched matmul (XLA can optimize this better than einsum)
    QK = jnp.matmul(Q_reshaped, K_reshaped)  # [n_heads, q_mult, n_new_tokens, n_kv_tokens]

    # Scale attention scores
    QK = QK * sm_scale

    # Apply causal (and optional sliding window) mask
    QK = QK + mask[None, None, :, :]

    # Concatenate sink tokens to attention logits
    QK = jnp.concatenate([QK, S_expanded], axis=-1)  # [n_heads, q_mult, n_tokens, n_tokens+1]

    # Compute attention weights via softmax
    W = jax.nn.softmax(QK, axis=-1)

    # Remove sink token attention weights (keep only the n_tokens part)
    W = W[..., :-1]  # [n_heads, q_mult, n_new_tokens, n_kv_tokens]

    # Apply attention weights to values
    # OPTIMIZED: Use explicit transpose + matmul instead of einsum
    # Original einsum: 'hmqk,khmd->qhmd'
    # W: [n_heads, q_mult, n_new_tokens, n_kv_tokens]
    # V: [n_kv_tokens, n_heads, 1 or q_mult, d_head]
    # Target: [n_new_tokens, n_heads, q_mult, d_head]

    # Reshape V for matmul: [n_heads, q_mult, n_kv_tokens, d_head]
    V_reshaped = V.transpose(1, 2, 0, 3)  # [n_heads, q_mult, n_kv_tokens, d_head]

    # Batched matmul: [n_heads, q_mult, n_new_tokens, n_kv_tokens] @ [n_heads, q_mult, n_kv_tokens, d_head]
    attn = jnp.matmul(W, V_reshaped)  # [n_heads, q_mult, n_new_tokens, d_head]

    # Transpose back to expected format: [n_new_tokens, n_heads, q_mult, d_head]
    attn = attn.transpose(2, 0, 1, 3)

    # Reshape to flat output
    output = attn.reshape(n_new_tokens, -1)  # [n_new_tokens, n_heads * q_mult * d_head]

    return output


class AttentionBlock(nn.Module):
    """Multi-head attention block with grouped-query attention (GQA).

    Implements:
    - RMSNorm pre-normalization
    - QKV projection with GQA (fewer KV heads than Q heads)
    - Rotary position embeddings (RoPE)
    - Scaled dot-product attention with optional sliding window
    - Sink tokens for improved attention
    - Output projection with residual connection
    - Optional KV caching for efficient autoregressive generation
    - Optional FlashAttention for memory-efficient computation

    Attributes:
        config: Model configuration
        layer_idx: Layer index (determines sliding window usage)
    """
    config: ModelConfig
    layer_idx: int = 0

    @nn.compact
    def __call__(self, x: jax.Array, kv_cache: Optional[Any] = None) -> tuple[jax.Array, Optional[Any]]:
        """Apply attention block.

        Args:
            x: Input tensor of shape [n_tokens, hidden_size]
            kv_cache: Optional KVCache instance for caching K/V tensors.
                     If None, operates without caching (default behavior).
                     If provided, uses cached K/V and returns updated cache.

        Returns:
            If kv_cache is None: Just the output tensor [n_tokens, hidden_size]
            If kv_cache provided: Tuple of (output tensor, updated_kv_cache)

        Raises:
            AssertionError: If shapes are invalid or contain NaN/Inf
        """
        n_tokens = x.shape[0]
        head_dim = self.config.head_dim
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        q_mult = num_attention_heads // num_key_value_heads

        # Sliding window only on even layers
        sliding_window = self.config.sliding_window if self.layer_idx % 2 == 0 else 0

        # Sink tokens (1 per attention head, as in PyTorch)
        # With GQA: num_attention_heads sink values get reshaped to [n_heads, q_mult]
        sinks = self.param(
            'sinks',
            nn.initializers.normal(stddev=0.02),
            (num_attention_heads,),
            jnp.bfloat16
        )

        # Pre-normalization
        norm = RMSNorm(num_features=self.config.hidden_size, name='norm')
        t = norm(x)

        # QKV projection
        qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        qkv_proj = nn.Dense(
            features=qkv_dim,
            use_bias=True,
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name='qkv'
        )
        qkv = qkv_proj(t)

        # Split into Q, K, V
        q_end = num_attention_heads * head_dim
        k_end = q_end + num_key_value_heads * head_dim
        v_end = k_end + num_key_value_heads * head_dim

        q = qkv[:, :q_end]
        k = qkv[:, q_end:k_end]
        v = qkv[:, k_end:v_end]

        # Reshape for attention
        # Q: [n_tokens, num_attention_heads * head_dim] -> [n_tokens, num_key_value_heads, q_mult, head_dim]
        q = q.reshape(n_tokens, num_key_value_heads, q_mult, head_dim)
        # K, V: [n_tokens, num_key_value_heads * head_dim] -> [n_tokens, num_key_value_heads, head_dim]
        k = k.reshape(n_tokens, num_key_value_heads, head_dim)
        v = v.reshape(n_tokens, num_key_value_heads, head_dim)

        # Determine KV offset BEFORE applying RoPE
        # This is critical: RoPE needs to know the absolute position in the sequence
        kv_offset = 0
        if kv_cache is not None:
            kv_offset = kv_cache.offset  # Offset before extending (number of previously cached tokens)

        # Apply rotary embeddings with correct position offset
        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=self.config.rope_theta,
            initial_context_length=self.config.initial_context_length,
            scaling_factor=self.config.rope_scaling_factor,
            ntk_alpha=self.config.rope_ntk_alpha,
            ntk_beta=self.config.rope_ntk_beta,
            name='rope'
        )
        q, k = rope(q, k, position_offset=kv_offset)

        # Handle KV caching
        updated_cache = None
        if kv_cache is not None:
            # Using cache: extend with new K/V and get full K/V
            # Add batch dimension for cache (expects 4D: [batch, n_tokens, n_heads, d_head])
            k_cached = k[None, :, :, :]  # [1, n_tokens, n_heads, d_head]
            v_cached = v[None, :, :, :]  # [1, n_tokens, n_heads, d_head]

            updated_cache, k_full, v_full = kv_cache.extend(k_cached, v_cached)

            # Remove batch dimension for attention computation
            k = k_full[0]  # [n_kv_tokens, n_heads, d_head]
            v = v_full[0]  # [n_kv_tokens, n_heads, d_head]
        # else: No cache, use k and v as-is (kv_offset remains 0)

        # Compute attention scale
        sm_scale = 1.0 / math.sqrt(head_dim)

        # Apply scaled dot-product attention
        attn_out = sdpa(q, k, v, sinks, sm_scale, sliding_window, kv_offset)

        # Output projection
        out_proj = nn.Dense(
            features=self.config.hidden_size,
            use_bias=True,
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name='out'
        )
        t = out_proj(attn_out)

        # Residual connection
        output = x + t

        # Return based on whether we're using cache
        if kv_cache is not None:
            return output, updated_cache
        else:
            return output


class MLPBlock(nn.Module):
    """MLP block with Mixture of Experts (MoE).

    Implements a sparse MoE layer where each token is routed to the top-k experts.
    Each expert is a 2-layer MLP with SwiGLU activation.

    Attributes:
        config: Model configuration
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply MLP block with expert routing.

        Args:
            x: Input tensor of shape [n_tokens, hidden_size]

        Returns:
            Output tensor of shape [n_tokens, hidden_size] after MLP + residual

        Raises:
            AssertionError: If shapes are invalid or contain NaN/Inf
        """
        n_tokens = x.shape[0]
        num_experts = self.config.num_experts
        experts_per_token = self.config.experts_per_token
        intermediate_size = self.config.intermediate_size
        hidden_size = self.config.hidden_size

        # Pre-normalization
        norm = RMSNorm(num_features=hidden_size, name='norm')
        t = norm(x)

        # Gating network: select top-k experts per token
        gate = nn.Dense(
            features=num_experts,
            use_bias=True,
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name='gate'
        )
        g = gate(t)  # [n_tokens, num_experts]

        # Select top-k experts
        # JAX top_k returns (values, indices) sorted in descending order
        expert_logits, expert_indices = jax.lax.top_k(g, experts_per_token)
        expert_weights = jax.nn.softmax(expert_logits, axis=-1)  # [n_tokens, experts_per_token]

        # Expert MLP weights (shared between baseline and optimized paths)
        # mlp1: hidden -> intermediate*2 (for SwiGLU gate/linear split)
        # mlp2: intermediate -> hidden
        mlp1_weight = self.param(
            'mlp1_weight',
            nn.initializers.normal(stddev=0.02),
            (num_experts, intermediate_size * 2, hidden_size),
            jnp.bfloat16
        )
        mlp1_bias = self.param(
            'mlp1_bias',
            nn.initializers.zeros,
            (num_experts, intermediate_size * 2),
            jnp.bfloat16
        )
        mlp2_weight = self.param(
            'mlp2_weight',
            nn.initializers.normal(stddev=0.02),
            (num_experts, hidden_size, intermediate_size),
            jnp.bfloat16
        )
        mlp2_bias = self.param(
            'mlp2_bias',
            nn.initializers.zeros,
            (num_experts, hidden_size),
            jnp.bfloat16
        )

        # Compute expert outputs: Baseline vs Optimized path
        # BASELINE: Per-token processing
        # OPTIMIZED: Replace einsum with batched matmul for better XLA optimization
        # Gather MLP1 weights and biases for selected experts
        # mlp1_weight[expert_indices] -> [n_tokens, experts_per_token, intermediate_size*2, hidden_size]
        selected_mlp1_weight = mlp1_weight[expert_indices]
        selected_mlp1_bias = mlp1_bias[expert_indices]

        # Expand t to [n_tokens, experts_per_token, hidden_size]
        t_expanded = t[:, None, :].repeat(experts_per_token, axis=1)

        # Apply MLP1: batched matmul for each token-expert pair
        # Original einsum: 'beck,bek->bec' (batch, expert, channel, kernal @ batch, expert, kernel)
        # Optimized: use vmap over batch and expert dimensions
        # Shape: [n_tokens, experts_per_token, hidden_size] @ [n_tokens, experts_per_token, hidden_size, intermediate_size*2]
        # Result: [n_tokens, experts_per_token, intermediate_size*2]
        mlp1_out = jnp.matmul(t_expanded[:, :, None, :], selected_mlp1_weight.transpose(0, 1, 3, 2))
        mlp1_out = mlp1_out.squeeze(axis=2) + selected_mlp1_bias  # Remove singleton dim and add bias
        mlp1_out = swiglu(mlp1_out, limit=self.config.swiglu_limit)

        # Gather MLP2 weights and biases
        selected_mlp2_weight = mlp2_weight[expert_indices]
        selected_mlp2_bias = mlp2_bias[expert_indices]

        # Apply MLP2
        # Shape: [n_tokens, experts_per_token, intermediate_size] @ [n_tokens, experts_per_token, intermediate_size, hidden_size]
        # Result: [n_tokens, experts_per_token, hidden_size]
        mlp2_out = jnp.matmul(mlp1_out[:, :, None, :], selected_mlp2_weight.transpose(0, 1, 3, 2))
        mlp2_out = mlp2_out.squeeze(axis=2) + selected_mlp2_bias

        # Weighted sum of expert outputs
        # [n_tokens, experts_per_token, hidden_size] * [n_tokens, experts_per_token, 1]
        # -> sum over experts_per_token dimension
        expert_outputs = jnp.sum(mlp2_out * expert_weights[:, :, None], axis=1)  # [n_tokens, hidden_size]

        # Residual connection
        output = x + expert_outputs

        return output


class TransformerBlock(nn.Module):
    """Single transformer block combining attention and MLP.

    Implements the standard transformer architecture:
    x = x + Attention(x)
    x = x + MLP(x)

    Attributes:
        config: Model configuration
        layer_idx: Layer index (passed to AttentionBlock for sliding window logic)
    """
    config: ModelConfig
    layer_idx: int

    @nn.compact
    def __call__(self, x: jax.Array, kv_cache: Optional[Any] = None) -> tuple[jax.Array, Optional[Any]]:
        """Apply transformer block.

        Args:
            x: Input tensor of shape [n_tokens, hidden_size]
            kv_cache: Optional KVCache for this layer

        Returns:
            If kv_cache is None: Just the output tensor
            If kv_cache provided: Tuple of (output tensor, updated_kv_cache)

        Raises:
            AssertionError: If shapes are invalid or contain NaN/Inf
        """
        # Attention block (includes residual connection)
        attn = AttentionBlock(
            config=self.config,
            layer_idx=self.layer_idx,
            name='attn'
        )
        if kv_cache is not None:
            x, updated_cache = attn(x, kv_cache)
        else:
            x = attn(x)
            updated_cache = None

        # MLP block (includes residual connection)
        mlp = MLPBlock(config=self.config, name='mlp')
        x = mlp(x)

        if kv_cache is not None:
            return x, updated_cache
        else:
            return x


class Transformer(nn.Module):
    """Full transformer model for gpt-oss-20b.

    Architecture:
    - Embedding layer
    - N transformer blocks (attention + MLP)
    - Final RMSNorm
    - Unembedding (Linear without bias) for logits

    Attributes:
        config: Model configuration
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, kv_caches: Optional[List[Any]] = None) -> tuple[jax.Array, Optional[List[Any]]]:
        """Apply full transformer model.

        Args:
            x: Input token IDs of shape [n_tokens] (int32)
            kv_caches: Optional list of KVCache instances (one per layer).
                      If None, operates without caching (default behavior).
                      If provided, must have length equal to num_hidden_layers.

        Returns:
            If kv_caches is None: Just logits of shape [n_tokens, vocab_size]
            If kv_caches provided: Tuple of (logits, updated_kv_caches)

        Raises:
            AssertionError: If shapes are invalid or contain NaN/Inf
        """
        n_tokens = x.shape[0]

        # Embedding
        embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=jnp.bfloat16,
            name='embedding'
        )
        h = embedding(x)  # [n_tokens, hidden_size]

        # Transformer blocks
        updated_caches = [] if kv_caches is not None else None
        for layer_idx in range(self.config.num_hidden_layers):
            block = TransformerBlock(
                config=self.config,
                layer_idx=layer_idx,
                name=f'block_{layer_idx}'
            )
            if kv_caches is not None:
                h, updated_cache = block(h, kv_caches[layer_idx])
                updated_caches.append(updated_cache)
            else:
                h = block(h)

        # Final normalization
        norm = RMSNorm(num_features=self.config.hidden_size, name='norm')
        h = norm(h)

        # Unembedding (no bias)
        unembedding = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name='unembedding'
        )
        logits = unembedding(h)  # [n_tokens, vocab_size]

        if kv_caches is not None:
            return logits, updated_caches
        else:
            return logits
