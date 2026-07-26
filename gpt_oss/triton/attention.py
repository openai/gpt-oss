"""FlashAttention w/support for learned sinks and banded attention.

This is an expanded version of the Flash Attention v2 implementation (see https://tridao.me/publications/flash2/flash2.pdf)
which can be found at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.

This version has been extended to support banded attention and learned attention sinks.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor



@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    M,
    Out,  #
    Start_q,
    Z,
    H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    if BANDWIDTH:
        lo, hi = tl.maximum(0, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, start_q + (start_m + 1) * BLOCK_M
    hi = tl.minimum(hi, N_KV_CTX)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T
        qk = tl.dot(q, k, allow_tf32=False)

        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc, allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    sink = tl.math.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]
    m_i += tl.math.log(l_i)
    m_ptrs = M + off_hz * N_Q_CTX + offs_m
    tl.store(m_ptrs, m_i)
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        assert len(start_q) == 1
        bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM_Q = q.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K = k.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_V = v.shape
        n_heads = n_kv_heads * repeat_kv
        q = q.view(bs, n_ctx, n_heads, HEAD_DIM_Q)
        k = k.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        q = q.transpose(1, 2).contiguous()
        k = k.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v = v.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        BLOCK_M = 64
        BLOCK_N = 64
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        # pad q to multiple of its block size in the n_ctx dimension (-2)
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))
        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        # pad k and v to multiple of their block size in the n_kv_ctx dimension
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        o = torch.empty_like(q)
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)
        _attn_fwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, HEAD_DIM_K]),
            sinks,
            sm_scale,
            M,
            TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM_K]),
            start_q,
            q.shape[0],
            q.shape[1],
            N_Q_CTX=n_ctx + m_pad_size,
            N_KV_CTX=n_kv_ctx,
            HEAD_DIM=HEAD_DIM_K,
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
        ctx.sm_scale = sm_scale
        ctx.bandwidth = bandwidth

        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()
        o = o.view(bs, n_ctx, n_heads * HEAD_DIM_V)
        return o


attention = _attention.apply


def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
):
    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups * head_dim).bfloat16()
    return output


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_queries", [1, 128])
@pytest.mark.parametrize("num_keys", [128, 32])
@pytest.mark.parametrize("num_key_value_heads", [8])
@pytest.mark.parametrize("num_key_value_groups", [8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("sm_scale", [0.125])
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("start_q", [0, 5])
def test_eq(batch_size, num_queries, num_keys, num_key_value_heads, num_key_value_groups, head_dim, sm_scale, sliding_window, start_q):
    if num_queries > num_keys:
        pytest.skip("too many queries")

    q = torch.randn(batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim).bfloat16().cuda()
    k = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    v = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    sinks = torch.randn(num_key_value_heads * num_key_value_groups).bfloat16().cuda()

    start_q = torch.tensor([start_q], dtype=torch.int32).cuda()

    o1 = attention(q, k, v, sinks, sm_scale, sliding_window, start_q)
    o2 = attention_ref(q, k, v, sinks, sm_scale, sliding_window, start_q)

    torch.testing.assert_close(o1, o2)