"""Model configuration for gpt-oss-20b.

This configuration is identical to the PyTorch reference implementation,
ensuring compatibility when loading weights and comparing outputs.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the gpt-oss-20b model architecture.

    Attributes:
        num_hidden_layers: Number of transformer layers
        num_experts: Total number of experts in MoE layers
        experts_per_token: Number of experts activated per token
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of hidden states
        intermediate_size: Dimension of MLP intermediate layer
        swiglu_limit: Clipping limit for SwiGLU activation
        head_dim: Dimension of each attention head
        num_attention_heads: Number of attention heads (query)
        num_key_value_heads: Number of key/value heads (GQA)
        sliding_window: Sliding window size for local attention
        initial_context_length: Initial context length for RoPE
        rope_theta: Base frequency for RoPE
        rope_scaling_factor: Scaling factor for extended context (YaRN)
        rope_ntk_alpha: NTK alpha parameter for frequency interpolation
        rope_ntk_beta: NTK beta parameter for frequency extrapolation
    """
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    def __post_init__(self):
        """Validate configuration parameters."""
        # Positive value checks
        assert self.num_hidden_layers > 0, \
            f"num_hidden_layers must be positive, got {self.num_hidden_layers}"
        assert self.num_experts > 0, \
            f"num_experts must be positive, got {self.num_experts}"
        assert self.experts_per_token > 0, \
            f"experts_per_token must be positive, got {self.experts_per_token}"
        assert self.vocab_size > 0, \
            f"vocab_size must be positive, got {self.vocab_size}"
        assert self.hidden_size > 0, \
            f"hidden_size must be positive, got {self.hidden_size}"
        assert self.intermediate_size > 0, \
            f"intermediate_size must be positive, got {self.intermediate_size}"
        assert self.head_dim > 0, \
            f"head_dim must be positive, got {self.head_dim}"
        assert self.num_attention_heads > 0, \
            f"num_attention_heads must be positive, got {self.num_attention_heads}"
        assert self.num_key_value_heads > 0, \
            f"num_key_value_heads must be positive, got {self.num_key_value_heads}"

        # Logical constraints
        assert self.experts_per_token <= self.num_experts, \
            f"experts_per_token ({self.experts_per_token}) cannot exceed num_experts ({self.num_experts})"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by " \
            f"num_key_value_heads ({self.num_key_value_heads})"
        assert self.intermediate_size % 2 == 0, \
            f"intermediate_size must be even for SwiGLU, got {self.intermediate_size}"

        # Sliding window check
        assert self.sliding_window >= 0, \
            f"sliding_window must be non-negative, got {self.sliding_window}"

        # RoPE parameter checks
        assert self.rope_theta > 0, \
            f"rope_theta must be positive, got {self.rope_theta}"
        assert self.rope_scaling_factor >= 1.0, \
            f"rope_scaling_factor must be >= 1.0, got {self.rope_scaling_factor}"
        assert self.rope_ntk_alpha > 0, \
            f"rope_ntk_alpha must be positive, got {self.rope_ntk_alpha}"
        assert self.rope_ntk_beta > 0, \
            f"rope_ntk_beta must be positive, got {self.rope_ntk_beta}"
        assert self.initial_context_length > 0, \
            f"initial_context_length must be positive, got {self.initial_context_length}"

    @property
    def q_mult(self) -> int:
        """Number of query heads per key/value head (GQA multiplier)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def total_attention_dim(self) -> int:
        """Total dimension of all attention heads."""
        return self.num_attention_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        """Total dimension of concatenated Q, K, V projections."""
        return self.head_dim * (self.num_attention_heads + 2 * self.num_key_value_heads)
