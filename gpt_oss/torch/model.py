import json
import math
from dataclasses import dataclass

from gpt_oss.torch.weights import Checkpoint

#from line_profiler import profile

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

    gpu_expert_cache_size: int = 5  # How many experts per block to keep in VRAM
    ram_expert_cache_size: int = 15  # How many experts per block to keep in Pinned RAM
    weights_path: str = ""  # Path to the folder containing converted mlp safetensors


from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open


class LazyMLPBlock(nn.Module):
    def __init__(
            self,
            config,
            layer_idx: int,
            device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.device = device or torch.device("cuda")

        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.hidden_size = config.hidden_size

        if dist.is_initialized():
            self.my_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.my_rank = 0
            self.world_size = 1

        self.per_rank_intermediate_size = config.intermediate_size // self.world_size

        # Permanent GPU layers
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )

        # Path Setup
        self.prefix = f"block.{layer_idx}.mlp"
        self.file_path = os.path.join(config.weights_path, "model.safetensors")

        # --- Preload Scales and Biases to CUDA ---
        self._preload_metadata_to_cuda()

        self.gpu_cache = OrderedDict()
        self.max_gpu_cache = getattr(config, "gpu_expert_cache_size", 6)
        self.ram_cache = OrderedDict()
        self.max_ram_cache = getattr(config, "ram_expert_cache_size", 18)


    def _preload_metadata_to_cuda(self):
        """Loads all scales and biases for all experts into GPU memory once."""
        with safe_open(self.file_path, framework="pt", device="cpu") as f:
            # Load full tensors for all experts
            s1 = f.get_tensor(f"{self.prefix}.mlp1_weight_scale")
            b1 = f.get_tensor(f"{self.prefix}.mlp1_bias")
            s2 = f.get_tensor(f"{self.prefix}.mlp2_weight_scale")
            b2 = f.get_tensor(f"{self.prefix}.mlp2_bias")

            if self.world_size > 1:
                p = self.per_rank_intermediate_size
                start = self.my_rank * 2 * p
                end = (self.my_rank + 1) * 2 * p
                b1 = b1[:, start:end]

            self.all_s1 = s1.to(self.device, non_blocking=True)
            self.all_b1 = b1.to(self.device, non_blocking=True)
            self.all_s2 = s2.to(self.device, non_blocking=True)
            self.all_b2 = b2.to(self.device, non_blocking=True)

    #@profile
    def _load_from_disk(self, expert_idx: int):
        """Loads only FP8 weights from disk and applies TP sharding."""
        with safe_open(self.file_path, framework="pt", device="cuda") as f:
            w1_fp8 = f.get_slice(f"{self.prefix}.mlp1_weight")[expert_idx:expert_idx + 1].squeeze(0)
            w2_fp8 = f.get_slice(f"{self.prefix}.mlp2_weight")[expert_idx:expert_idx + 1].squeeze(0)

            if self.world_size > 1:
                p = self.per_rank_intermediate_size
                w1_fp8 = w1_fp8[self.my_rank * 2 * p: (self.my_rank + 1) * 2 * p, :]
                w2_fp8 = w2_fp8[:, self.my_rank * p: (self.my_rank + 1) * p]

        return w1_fp8, w2_fp8

    #@profile
    def _get_expert_weights(self, idx: int):
        """Retrieves ONLY weights (FP8) from cache or disk."""
        if idx in self.gpu_cache:
            self.gpu_cache.move_to_end(idx)
            return self.gpu_cache[idx]

        if idx in self.ram_cache:
            tensors = self.ram_cache.pop(idx)
            gpu_tensors = tuple(t.to(self.device, non_blocking=True) for t in tensors)
            self._add_to_gpu_cache(idx, gpu_tensors)
            return gpu_tensors

        cpu_tensors = self._load_from_disk(idx)
        gpu_tensors = tuple(t.to(self.device, non_blocking=True) for t in cpu_tensors)
        self._add_to_gpu_cache(idx, gpu_tensors)
        return gpu_tensors

    #@profile
    def _add_to_gpu_cache(self, idx, weights):
        if len(self.gpu_cache) >= self.max_gpu_cache:
            evict_idx, evict_weights = self.gpu_cache.popitem(last=False)
            self._add_to_ram_cache(evict_idx, evict_weights)
        self.gpu_cache[idx] = weights

    #@profile
    def _add_to_ram_cache(self, idx, weights):
        if self.max_ram_cache > 0:
            if len(self.ram_cache) >= self.max_ram_cache:
                self.ram_cache.popitem(last=False)
            pinned = tuple(t.to("cpu", non_blocking=True).pin_memory() for t in weights)
            self.ram_cache[idx] = pinned

    #@profile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = x.shape[0] * x.shape[1] if len(x.shape) > 2 else x.shape[0]
        x_flat = x.view(-1, self.hidden_size)

        t = self.norm(x_flat)
        gate_logits = self.gate(t)
        experts = torch.topk(gate_logits, k=self.experts_per_token, dim=-1)

        expert_weights = F.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_weights.view(-1)

        sorted_experts, sorted_indices = torch.sort(flat_expert_indices)

        token_row_indices = torch.arange(batch_size, device=self.device).repeat_interleave(self.experts_per_token)
        sorted_row_indices = token_row_indices[sorted_indices]

        x_sorted = t[sorted_row_indices]

        active_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        active_experts_list = active_experts.tolist()
        counts_list = counts.tolist()

        final_output = torch.zeros_like(x_flat)

        start_idx = 0
        for i, exp_idx in enumerate(active_experts_list):
            count = counts_list[i]
            current_tokens = x_sorted[start_idx: start_idx + count]
            routing_weights = flat_expert_weights[sorted_indices[start_idx: start_idx + count]]

            w1_fp8, w2_fp8 = self._get_expert_weights(exp_idx)
            s1, b1 = self.all_s1[exp_idx], self.all_b1[exp_idx]
            s2, b2 = self.all_s2[exp_idx], self.all_b2[exp_idx]

            # MLP1
            w1_bf16 = w1_fp8.to(torch.bfloat16) * s1
            h = F.linear(current_tokens, w1_bf16, bias=b1)
            h = swiglu(h, limit=self.swiglu_limit)

            # MLP2
            w2_bf16 = w2_fp8.to(torch.bfloat16) * s2
            h = F.linear(h, w2_bf16, bias=None)

            if self.world_size > 1:
                dist.all_reduce(h, op=dist.ReduceOp.SUM)

            h = h + b2
            h = h * routing_weights.unsqueeze(-1)

            active_rows = sorted_row_indices[start_idx: start_idx + count]
            final_output.index_add_(0, active_rows, h)

            start_idx += count

        return x + final_output.view(original_shape)


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
        self.mlp = LazyMLPBlock(config, layer_idx, device)

    #@profile
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
    ):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    #@profile
    def forward(self, x: torch.Tensor,  kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
                start_pos: int = 0,) -> torch.Tensor:
        x = self.embedding(x)
        if kv_cache is None:
            kv_cache = [None] * self.config.num_hidden_layers
        layer_idx = 0
        for block in self.block:
            past_kv_for_layer = kv_cache[layer_idx]
            x, new_kv_for_layer = block(x, past_kv_for_layer, start_pos)
            kv_cache[layer_idx] = new_kv_for_layer
            layer_idx += 1
        x = self.norm(x)
        x = self.unembedding(x)
        return x, kv_cache

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda", mlp_safetensors: str = ""
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config,
                weights_path = mlp_safetensors,
                gpu_expert_cache_size = 6,
                ram_expert_cache_size = 18
            )

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        checkpoint = Checkpoint(path, device)
        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)
            try:
                param.data.copy_(loaded_tensor)
            except:
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device, mlp_safetensors: str = ""):
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device, mlp_safetensors=mlp_safetensors)

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
