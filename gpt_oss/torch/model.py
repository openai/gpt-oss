import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

#from xformers.ops import RMSNorm

from gpt_oss.torch.weights import Checkpoint

try:
    profile # type: ignore
except NameError:
    profile = lambda f: f


@dataclass
class ModelConfig:
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

p ="""
import triton
import triton.language as tl


@triton.jit
def rms_norm_forward_kernel(
        x_ptr,  # Pointer to the input tensor
        output_ptr,  # Pointer to the output tensor
        scale_ptr,  # Pointer to the scaling tensor
        x_row_stride,  # Stride to move to the next row in the input tensor
        output_row_stride,  # Stride to move to the next row in the output tensor
        num_features,  # Number of features in each row
        eps,  # Epsilon value for numerical stability
        BLOCK_SIZE: tl.constexpr,  # Block size for Triton kernel
):

    # Get the row index of the current program instance
    row_idx = tl.program_id(axis=0)

    # --- Load the row of data from the input tensor ---
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < num_features
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # --- Compute the sum of squares for the row ---
    row_var = tl.sum(x * x, axis=0) / num_features
    rstd = tl.math.rsqrt(row_var + eps)

    # --- Normalize the input and apply the scaling factor ---
    scale = tl.load(scale_ptr + offsets, mask=mask)
    normalized_x = x * rstd
    output = normalized_x * scale

    # --- Write the result to the output tensor ---
    output_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_start_ptr + offsets
    tl.store(output_ptrs, output, mask=mask)


class RMSNorm_(torch.nn.Module):
    def __init__(
            self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )
    @profile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features

        output = torch.empty_like(x)

        # Get the number of rows
        n_rows = x.numel() // self.num_features

        # Define the block size for the Triton kernel
        BLOCK_SIZE = triton.next_power_of_2(self.num_features)

        # Launch the Triton kernel
        rms_norm_forward_kernel[(n_rows,)](
            x,
            output,
            self.scale,
            x.stride(0),
            output.stride(0),
            self.num_features,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
"""


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    @profile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb_optimized(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> torch.Tensor:
    half_dim = x.size(-1) // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat([o1, o2], dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
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
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int, start_pos: int = 0):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(start_pos, start_pos + num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens, start_pos=start_pos)
        cos = cos.unsqueeze(-2).to(query.dtype)
        sin = sin.unsqueeze(-2).to(query.dtype)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb_optimized(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb_optimized(key, cos, sin)
        key = key.reshape(key_shape)

        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self,
                x: torch.Tensor,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                start_pos: int = 0,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        t = self.norm(x)
        qkv = self.qkv(t)

        q_end = self.num_attention_heads * self.head_dim
        k_end = q_end + self.num_key_value_heads * self.head_dim
        q = qkv[:, :q_end]
        k = qkv[:, q_end:k_end]
        v = qkv[:, k_end:]

        q = q.view(-1, self.num_attention_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        q, k = self.rope(q, k, start_pos=start_pos)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=0)
            v = torch.cat([past_v, v], dim=0)

        new_kv = (k, v)
        num_groups = self.num_attention_heads // self.num_key_value_heads

        k_expanded = k.repeat_interleave(num_groups, dim=1)
        v_expanded = v.repeat_interleave(num_groups, dim=1)

        query_len, num_heads, head_dim = q.shape
        key_len = k_expanded.shape[0]

        q_permuted = q.permute(1, 0, 2)
        k_permuted = k_expanded.permute(1, 0, 2)
        QK = torch.bmm(q_permuted, k_permuted.transpose(1, 2))
        QK *= self.sm_scale

        all_indices = torch.arange(key_len, device=x.device)
        query_indices = torch.arange(start_pos, start_pos + query_len, device=x.device)

        causal_mask = query_indices[:, None] < all_indices[None, :]

        mask = causal_mask

        if self.sliding_window > 0:
            sliding_mask = query_indices[:, None] < (all_indices[None, :] + self.sliding_window)
            mask = mask | ~sliding_mask

        QK = QK.masked_fill(mask, -torch.inf)

        S = self.sinks.view(num_heads, 1, 1).expand(-1, query_len, -1)
        QK = torch.cat([QK, S], dim=-1)

        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]

        v_permuted = v_expanded.permute(1, 0, 2)
        attn = torch.bmm(W, v_permuted)
        attn = attn.permute(1, 0, 2)

        t = attn.reshape(-1, self.num_attention_heads * self.head_dim)
        t = self.out(t)

        return t, new_kv

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock_(torch.nn.Module): # memory unefficient
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        assert config.intermediate_size % self.world_size == 0
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return x + t

class MLPBlock(torch.nn.Module):
    def __init__(
            self,
            config: ModelConfig,
            device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        assert config.intermediate_size % self.world_size == 0
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    self.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, self.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)

        if t.dim() == 2:
            t = t.unsqueeze(1)
            g = g.unsqueeze(1)
            added_seq_dim = True
        else:
            added_seq_dim = False

        batch_size, seq_len, hidden_size = t.shape

        t_flat = t.reshape(-1, hidden_size)
        g_flat = g.reshape(-1, self.num_experts)

        experts = torch.topk(g_flat, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        output_flat = torch.zeros_like(t_flat)

        for k in range(self.experts_per_token):
            current_expert_indices = expert_indices[:, k]
            current_expert_weights = expert_weights[:, k]

            mlp1_w = self.mlp1_weight[
                current_expert_indices]
            mlp1_b = self.mlp1_bias[current_expert_indices]
            mlp2_w = self.mlp2_weight[
                current_expert_indices]
            mlp2_b = self.mlp2_bias[current_expert_indices]

            t_k = torch.bmm(t_flat.unsqueeze(1), mlp1_w.transpose(1, 2)).squeeze(1) + mlp1_b
            t_k = swiglu(t_k, limit=self.swiglu_limit)

            t_k = torch.bmm(t_k.unsqueeze(1), mlp2_w.transpose(1, 2)).squeeze(1)

            if self.world_size > 1:
                dist.all_reduce(t_k, op=dist.ReduceOp.SUM)

            t_k += mlp2_b

            output_flat += t_k * current_expert_weights.unsqueeze(1)

        output = output_flat.reshape(batch_size, seq_len, hidden_size)

        if added_seq_dim:
            output = output.squeeze(1)

        return x + output


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    @profile
    def forward(self,
                x: torch.Tensor,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                start_pos: int = 0,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_output, new_kv = self.attn(x, past_kv, start_pos)
        x = x + attn_output
        x = self.mlp(x)
        return x, new_kv

checkpoint = None

def get_free_gpu_memory_gb(device_id=0):
    """Returns free GPU memory in GB for specified device (default: 0)"""
    if not torch.cuda.is_available():
        return 0.0

    props = torch.cuda.get_device_properties(device_id)
    total_memory = props.total_memory
    reserved = torch.cuda.memory_reserved(device_id)

    free_memory = total_memory - reserved
    free_gb = free_memory / (1024 ** 3)

    return free_gb


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
        lazy_load: bool = True,
        extremly_low_memory: bool = False,
    ):
        super().__init__()
        free_mem = get_free_gpu_memory_gb()
        if free_mem < 6:
            lazy_load = True
            extremly_low_memory = True
        elif free_mem < 8:
            lazy_load = True
        self.lazy_load = lazy_load
        self.extremly_low_memory = extremly_low_memory
        self.config = config
        self.device = device
        self.loaded = {0: False, 1: False, 2: False}
        if self.lazy_load:
            self.block = torch.nn.ModuleList([
                TransformerBlock(config, 0, device=device),
            ])
        else:
            self.block = torch.nn.ModuleList(
                [
                    TransformerBlock(config, layer_idx, device)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
            self.norm = RMSNorm(config.hidden_size, device=device)
            self.embedding = torch.nn.Embedding(
                config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
            )
            self.unembedding = torch.nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                device=device,
                dtype=torch.bfloat16,
            )

    @profile
    def forward(self,
                x: torch.Tensor,
                kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
                start_pos: int = 0,
                ) -> tuple[
        torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        if self.lazy_load:
            if not self.loaded[1] or self.extremly_low_memory:
                self.embedding = torch.nn.Embedding(
                    self.config.vocab_size, self.config.hidden_size, device=self.device, dtype=torch.bfloat16
                )
                for name, param in self.embedding.named_parameters():
                    self.load_weights(param, f"embedding.{name}")
                self.loaded[1] = True
            x = self.embedding(x)
            if self.extremly_low_memory:
                torch.cuda.synchronize()
                self.embedding = None
                torch.cuda.empty_cache()
        else:
            x = self.embedding(x)

        if kv_cache is None:
            kv_cache = [None] * self.config.num_hidden_layers

        if self.lazy_load:
            for layer_idx in range(self.config.num_hidden_layers):
                # layers skipping experiment
                #if layer_idx % 2 == 0:
                #    continue
                for name, param in self.block[0].named_parameters():
                    self.load_weights(param, f"block.{layer_idx}.{name}")
                past_kv_for_layer = kv_cache[layer_idx]
                x, new_kv_for_layer = self.block[0](x, past_kv_for_layer, start_pos)
                kv_cache[layer_idx] = new_kv_for_layer
        else:
            for layer_idx in range(self.config.num_hidden_layers):
                past_kv_for_layer = kv_cache[layer_idx]
                x, new_kv_for_layer = self.block[layer_idx](x, past_kv_for_layer, start_pos)
                kv_cache[layer_idx] = new_kv_for_layer

        if self.lazy_load:
            norm = RMSNorm(self.config.hidden_size, device=self.device)
            for name, param in norm.named_parameters():
                self.load_weights(param, f"norm.{name}")
            x = norm(x)
            del norm
        else:
            x = self.norm(x)

        if self.lazy_load:
            if not self.loaded[0] or self.extremly_low_memory:
                self.unembedding = torch.nn.Linear(
                    self.config.hidden_size,
                    self.config.vocab_size,
                    bias=False,
                    device=self.device,
                    dtype=torch.bfloat16,
                 )
                for name, param in self.unembedding.named_parameters():
                    self.load_weights(param, f"unembedding.{name}")
                self.loaded[0] = True
            x = self.unembedding(x)
            if self.extremly_low_memory:
                torch.cuda.synchronize()
                self.unembedding = None
                torch.cuda.empty_cache()
        else:
            x = self.unembedding(x)
        return x, kv_cache

    def load_weights(self, param, name):
        global checkpoint
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = self.config.intermediate_size // world_size
        loaded_tensor = checkpoint.get(name)
        if "mlp1" in name:
            loaded_tensor = loaded_tensor[
                :,
                my_rank * 2
                * per_rank_intermediate_size: (my_rank + 1) * 2
                                              * per_rank_intermediate_size,
                ...,
                ]
        elif "mlp2_weight" in name:
            loaded_tensor = loaded_tensor[
                ...,
                my_rank
                * per_rank_intermediate_size: (my_rank + 1)
                                              * per_rank_intermediate_size,
                ]
        try:
            param.data.copy_(loaded_tensor)
        except:
            print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
            raise

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda", lazy_load: bool = True, pin_memory: bool = False
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        extremly_low_memory = False
        if not pin_memory:
            extremly_low_memory = True

        model = Transformer(
            config=config,
            device=device,
            extremly_low_memory=extremly_low_memory,
        )
        if not lazy_load:
            model.eval()

        global checkpoint
        checkpoint = Checkpoint(path, device, pin_memory)

        if not lazy_load:
            # Load weights
            my_rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            per_rank_intermediate_size = config.intermediate_size // world_size
            for name, param in model.named_parameters():
                loaded_tensor = checkpoint.get(name)
                # Note: it would be more efficient to do sharding before upcasting from MXFP4,
                # but for simplicity we do it after.
                if "mlp1" in name:  # both weight and bias
                    loaded_tensor = loaded_tensor[
                        :,
                        my_rank * 2
                        * per_rank_intermediate_size : (my_rank + 1) * 2
                        * per_rank_intermediate_size,
                        ...,
                    ]
                elif "mlp2_weight" in name:  # only weight
                    loaded_tensor = loaded_tensor[
                        ...,
                        my_rank
                        * per_rank_intermediate_size : (my_rank + 1)
                        * per_rank_intermediate_size,
                    ]
                try:
                    param.data.copy_(loaded_tensor)
                except:
                    print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                    raise

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device, pin_memory: bool = False):
        self.device = device
        self.model = Transformer.from_checkpoint(
            checkpoint, device=self.device, pin_memory=pin_memory)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        tokens = list(prompt_tokens)
        num_prompt_tokens = len(tokens)
        num_generated_tokens = 0

        kv_cache = None
        prompt_tensor = torch.as_tensor(tokens, dtype=torch.int32, device=self.device)
        logits, kv_cache = self.model(prompt_tensor, kv_cache=None, start_pos=0)
        logits = logits[-1]

        while max_tokens == 0 or num_generated_tokens < max_tokens:
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break

            tokens.append(predicted_token)
            num_generated_tokens += 1
            next_token_tensor = torch.as_tensor([predicted_token], dtype=torch.int32, device=self.device)
            start_pos = num_prompt_tokens + num_generated_tokens - 1
            logits, kv_cache = self.model(next_token_tensor, kv_cache=kv_cache, start_pos=start_pos)
            logits = logits[0]
