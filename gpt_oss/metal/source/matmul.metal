#include <metal_atomic>
#include <metal_compute>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


// Each simdgroup reduces all channels of the input and computes a single channel of the output
// + Efficient synchronization
// + Sequential memory access within a warp
// Each threadgroup computes (simdgroups_per_threadgroup) consecutive output channels
// + Reuse input vector from threadgroup memory
// + Avoid synchronization across warps when doing reduction

kernel void gptoss_f32_bf16w_matmul(
    constant gptoss_matmul_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device bfloat4* weight [[ buffer(2) ]],
    const device bfloat* bias [[ buffer(3) ]],
    device float* output [[ buffer(4) ]],
    const device gptoss_control* control [[ buffer(5) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;
    if (control->abort != 0) {
        return;
    }

    const uint num_column_vecs = args.num_column_vecs;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;

    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row + simdgroup_tid;
    bias += row;
    output += gid.y * args.num_rows + row;

    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    float4 sum4 = 0.0f;
    do {
        const bfloat4 w = *weight;
        const float4 i = *input;
        sum4 = metal::fma(static_cast<float4>(w), i, sum4);

        weight += simdgroup_size;
        input += simdgroup_size;
    } while (--num_iter != 0);
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    if (metal::simd_is_first()) {
        sum += static_cast<float>(*bias);
        if (args.add) {
            *output += sum;
        } else {
            *output = sum;
        }
    }
}

kernel void gptoss_f32_bf16w_matmul_qkv(
    constant gptoss_qkv_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device bfloat4* weight [[ buffer(2) ]],
    const device bfloat* bias [[ buffer(3) ]],
    device float* q [[ buffer(4) ]],
    device float* kv [[ buffer(5) ]],
    const device gptoss_control* control [[ buffer(6) ]],
    threadgroup void* scratch [[ threadgroup(0) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;
    const uint head_dim = 64;
    const uint num_q_heads = 64;
    const uint num_kv_heads = 8;
    if (control->abort != 0) {
        return;
    }

    const uint num_column_vecs = args.num_column_vecs;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;

    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row + simdgroup_tid;
    bias += row;
    q += gid.y * args.num_rows;

    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    float4 sum4 = 0.0f;
    do {
        const bfloat4 w = *weight;
        const float4 i = *input;
        sum4 = metal::fma(static_cast<float4>(w), i, sum4);

        weight += simdgroup_size;
        input += simdgroup_size;
    } while (--num_iter != 0);
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    if (metal::simd_is_first()) {
        sum += static_cast<float>(*bias);
        static_cast<threadgroup float*>(scratch)[simdgroup_idx] = sum;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simdgroup_idx == 0) {
        const uint num_half_simdgroups = num_simdgroups / 2;
        if (simdgroup_tid < num_half_simdgroups) {
            float2 vals = static_cast<const threadgroup float2*>(scratch)[simdgroup_tid];
            const uint idx = gid.x * num_half_simdgroups + simdgroup_tid;
            const uint head_idx = idx / (head_dim / 2);
            const uint token_idx = args.token_offset + gid.y;
            const uint dim_idx = idx % (head_dim / 2);
            if (head_idx < num_q_heads + num_kv_heads) {
                const float dim_idx_val = static_cast<float>(dim_idx);
                const float inv_extrapolation_freq = metal::precise::exp(dim_idx_val * args.freq_scale);
                const float inv_interpolation_freq = inv_extrapolation_freq * args.interpolation_scale;
                const float alpha = metal::saturate(metal::fma(dim_idx_val, args.yarn_scale, args.yarn_offset));
                const float inv_freq = metal::mix(inv_extrapolation_freq, inv_interpolation_freq, alpha);

                const float phi = static_cast<float>(token_idx) * inv_freq;
                const float yarn_multiplier = args.yarn_multiplier;
                float cosphi;
                const float sinphi = metal::precise::sincos(phi, cosphi) * yarn_multiplier;
                cosphi *= yarn_multiplier;

                vals = (float2) {
                    vals.x * cosphi - vals.y * sinphi,
                    vals.x * sinphi + vals.y * cosphi,
                };
            }
            if (head_idx < num_q_heads) {
                reinterpret_cast<device float2*>(q)[idx] = vals;
            } else if (head_idx < num_q_heads + num_kv_heads) {
                const uint h = head_idx - num_q_heads;
                reinterpret_cast<device float2*>(kv + (h * args.max_tokens + token_idx) * 2 * head_dim)[dim_idx] = vals;
            } else {
                const uint h = head_idx - num_q_heads - num_kv_heads;
                reinterpret_cast<device float2*>(kv + (h * args.max_tokens + token_idx) * 2 * head_dim + head_dim)[dim_idx] = vals;
            }
        }
    }
}

kernel void gptoss_f32_bf16w_unembedding(
    constant gptoss_unembedding_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device bfloat4* weight [[ buffer(2) ]],
    device float* output [[ buffer(3) ]],
    device metal::atomic_ulong* argmax [[ buffer(4) ]],
    const device gptoss_control* control [[ buffer(5) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;
    threadgroup uint2 threadgroup_buffer[32];
    if (control->abort != 0) {
        return;
    }

    const uint num_column_vecs = args.num_column_vecs;
    const uint row_start = gid.x * args.num_rows_per_threadgroup + simdgroup_idx;
    const uint row_end = metal::min(gid.x * args.num_rows_per_threadgroup + args.num_rows_per_threadgroup, args.num_rows);
    const uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row_start + simdgroup_tid;
    output += gid.y * args.num_rows + row_start;

    uint2 row_sum{0xFFFFFFFFul, 0xFFFFFFFFul};
    for (uint row = row_start; row < row_end; row += num_simdgroups) {
        uint n = num_iter;

        float4 sum4 = 0.0f;
        do {
            const bfloat4 w = *weight;
            const float4 i = *input;

            sum4 = metal::fma(static_cast<float4>(w), i, sum4);

            weight += simdgroup_size;
            input += simdgroup_size;
        } while (--n != 0);
        input -= num_iter * simdgroup_size;
        weight -= num_iter * simdgroup_size;

        const float2 sum2 = sum4.xy + sum4.zw;
        float sum = sum2.x + sum2.y;
        sum = metal::simd_sum(sum);
        uint sum_bits = as_type<uint>(sum);
        if (static_cast<int>(sum_bits) >= 0) {
            sum_bits ^= 0x7FFFFFFFu;
        }
        row_sum = as_type<uint2>(metal::min(as_type<ulong>(row_sum), as_type<ulong>(uint2{row, sum_bits})));
        if (metal::simd_is_first()) {
            *output = sum;
        }

        weight += num_column_vecs * num_simdgroups;
        output += num_simdgroups;
    }
    if (metal::simd_is_first()) {
        threadgroup_buffer[simdgroup_idx] = row_sum;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simdgroup_idx == 0) {
        // Min-Reduce threadgroup_buffer
        if (simdgroup_tid < num_simdgroups) {
            row_sum = threadgroup_buffer[simdgroup_tid];
        }
        const uint sum_bits = row_sum.y;
        const uint sum_bits_min = metal::simd_min(sum_bits);
        const uint row_min = metal::simd_min(sum_bits == sum_bits_min ? row_sum.x : 0xFFFFFFFFu);
        if (metal::simd_is_first()) {
            const uint2 threadgroup_output{row_min, sum_bits_min};
            atomic_min_explicit(&argmax[gid.y], as_type<ulong>(threadgroup_output), metal::memory_order_relaxed);
        }
    }
}

// Current constraints for the dense matmul kernel:
//  1- All B* and Sg_* are a multiple of 8.
//  2- Bm is divisible by Sg_n and Bn is divisible by Sg_n.
//  3- M, N and K are all divisible by 8..
template <uint Bm, uint Bn, uint Bk, uint Sg_Bm, uint Sg_Bn, uint add = 0>
inline void _gptoss_f32_bf16w_dense_matmul_impl(
    constant gptoss_dense_matmul_args& args, const device float* lhs,
    const device bfloat* rhs, const device bfloat* __restrict__ bias,
    device float* out, const device gptoss_control* control, threadgroup float* scratch, threadgroup float* bias_tile,
    uint sg_id, uint sg_count_per_tg, uint3 gid, uint3 tg_id, uint3 local_tid,
    uint3 threadgroup_size) {

    if (control->abort != 0) {
        return;
    }

    // The kernel assumes that M, K, and N are divisible by 8.
    const uint M = args.m;
    const uint K = args.k;
    const uint N = args.n;
    static_assert((Bm % 8u) == 0u, "Bm must be a multiple of 8");
    static_assert((Bn % 8u) == 0u, "Bn must be a multiple of 8");
    static_assert((Bk % 8u) == 0u, "Bk must be a multiple of 8");
    static_assert((Sg_Bm % 8u) == 0u, "Bk must be a multiple of 8");
    static_assert((Sg_Bn % 8u) == 0u, "Bk must be a multiple of 8");
    static_assert((Bn % Sg_Bn) == 0u, "Bn must be a multiple of Sg_Bn");
    static_assert((Bm % Sg_Bm) == 0u, "Bm must be a multiple of Sg_Bm");

    // Get row and col tg.
    const uint row_tg = tg_id.y;
    const uint col_tg = tg_id.x;
    // Get row and col local tid.
    const uint row_tg_offset = row_tg * Bm;
    const uint col_tg_offset = col_tg * Bn;

    const uint sg_col_count = Bn / Sg_Bn;
    const uint row_sg = sg_id / sg_col_count;
    const uint col_sg = sg_id % sg_col_count;

    const uint row_sg_offset = row_sg * Sg_Bm;
    const uint col_sg_offset = col_sg * Sg_Bn;
    constexpr uint temp_result_size = (Sg_Bm / 8) * (Sg_Bn / 8);
    // Create an array of simdgroup_float8x8 to hold temp results.
    metal::simdgroup_float8x8 OutTiles[temp_result_size];
#pragma clang loop unroll(full)
    for (uint i = 0; i < temp_result_size; i++) {
        OutTiles[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(
            static_cast<float>(0.0));
    }

    for (uint k_offset = 0; k_offset < K; k_offset += Bk) {
#pragma clang loop unroll(full)
        for (uint k = 0; k < Bk; k += 8) {
#pragma clang loop unroll(full)
            for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
                // const uint m_subtile = row_sg_offset + m_subtile_;
                // const uint row_index_in_out_tile = (m_subtile - row_sg_offset) / 8;
                const uint row_index_in_out_tile = m_subtile_ / 8;
                metal::simdgroup_float8x8 LHStile;
                const uint k_id = k + k_offset;
                const uint row_offset = row_tg_offset + row_sg_offset + m_subtile_;
                metal::simdgroup_load(LHStile, lhs, K, ulong2(k_id, row_offset));
                metal::simdgroup_bfloat8x8 RHStile;
#pragma clang loop unroll(full)
                for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
                    const uint col_index_in_out_tile = n_subtile_ / 8;
                    const uint current_index_out_tile =
                        row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;
                    const uint col_offset = col_tg_offset + col_sg_offset + n_subtile_;
                    simdgroup_load(RHStile, rhs, K, ulong2(k_id, col_offset), /*transpose=*/true);
                    // If rhs was not transposed, use the following instead:
                    // simdgroup_load(RHStile, rhs, N, ulong2(col_offset, k_id));
                    simdgroup_multiply_accumulate(OutTiles[current_index_out_tile],
                                                  LHStile, RHStile,
                                                  OutTiles[current_index_out_tile]);
                }
            }
        }
    }
    // Epilogue.
#pragma clang loop unroll(full)
    for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
        const uint col_index_in_out_tile = n_subtile_ / 8;
        const uint local_col_offset = col_sg_offset + n_subtile_;
#pragma clang loop unroll(full)
        for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
            const uint row_index_in_out_tile = m_subtile_ / 8;
            const uint local_row_offset = row_sg_offset + m_subtile_;
            const uint current_index_out_tile =
                row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;
            simdgroup_store(OutTiles[current_index_out_tile], scratch, Bn,
                            ulong2(local_col_offset, local_row_offset));
        }
    }
    // TODO(ibahmed): vectorize these loads an maybe unroll the loop.
    const uint thread_count_per_tg =
        threadgroup_size.x * threadgroup_size.y * threadgroup_size.z;
    for (uint c_local = local_tid.x; c_local < Bn;
         c_local += thread_count_per_tg) {
        const uint c_global = col_tg_offset + c_local;
        bias_tile[c_local] =
            (c_global < N) ? static_cast<float>(bias[c_global]) : 0.0f;
    }

    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // TODO(ibahmed): vectorize these stores and maybe unroll the loop.
    for (uint idx = local_tid.x; idx < Bm * Bn; idx += thread_count_per_tg) {
        const uint r = idx / Bn;
        const uint c = idx % Bn;

        const uint out_row = row_tg_offset + r;
        const uint out_col = col_tg_offset + c;

        if (out_row < M && out_col < N) {
            float acc = scratch[idx] + bias_tile[c];
            if (add) {
                acc += out[out_row * N + out_col];
            }
            out[out_row * N + out_col] = acc;
        }
    }
}

kernel void gptoss_f32_bf16w_dense_matmul_qkv(
    constant gptoss_dense_matmul_qkv_args& args [[buffer(0)]],
    const device float* lhs [[buffer(1)]],
    const device bfloat* rhs [[buffer(2)]],
    const device bfloat* __restrict__ bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    device float* kv [[buffer(5)]],
    const device gptoss_control* control [[buffer(6)]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_count_per_tg [[dispatch_simdgroups_per_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 local_tid [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]]) {
    threadgroup float scratch[QKV_Bm * QKV_Bn];
    threadgroup float bias_tile[QKV_Bn];
    if (control->abort != 0) {
        return;
    }

    // The kernel assumes that QKV_Bm, QKV_Bn, QKV_Bk, QKV_Sg_Bm, QKV_Sg_Bn are divisible by 8.
    const uint M = args.m;
    const uint K = args.k;
    const uint N = args.n;
    const uint Bm = QKV_Bm;
    const uint Bn = QKV_Bn;
    const uint Bk = QKV_Bk;
    const uint Sg_Bm = QKV_Sg_Bm;
    const uint Sg_Bn = QKV_Sg_Bn;
    static_assert((Bm % 8u) == 0u, "Bm must be a multiple of 8");
    static_assert((Bn % 8u) == 0u, "Bn must be a multiple of 8");
    static_assert((Bk % 8u) == 0u, "Bk must be a multiple of 8");
    static_assert((Sg_Bm % 8u) == 0u, "Bk must be a multiple of 8");
    static_assert((Sg_Bn % 8u) == 0u, "Bk must be a multiple of 8");
    static_assert((Bn % Sg_Bn) == 0u, "Bn must be a multiple of Sg_Bn");
    static_assert((Bm % Sg_Bm) == 0u, "Bm must be a multiple of Sg_Bm");

    // Get row and col tg.
    const uint row_tg = tg_id.y;
    const uint col_tg = tg_id.x;
    // Get row and col local tid.
    const uint row_tg_offset = row_tg * Bm;
    const uint col_tg_offset = col_tg * Bn;

    const uint sg_col_count = Bn / Sg_Bn;
    const uint row_sg = sg_id / sg_col_count;
    const uint col_sg = sg_id % sg_col_count;

    const uint row_sg_offset = row_sg * Sg_Bm;
    const uint col_sg_offset = col_sg * Sg_Bn;
    constexpr uint temp_result_size = (Sg_Bm / 8) * (Sg_Bn / 8);
    // Create an array of simdgroup_float8x8 to hold temp results.
    metal::simdgroup_float8x8 OutTiles[temp_result_size];
#pragma clang loop unroll(full)
    for (uint i = 0; i < temp_result_size; i++) {
        OutTiles[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(
            static_cast<float>(0.0));
    }

    for (uint k_offset = 0; k_offset < K; k_offset += Bk) {
#pragma clang loop unroll(full)
        for (uint k = 0; k < Bk; k += 8) {
#pragma clang loop unroll(full)
            for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
                const uint row_index_in_out_tile = m_subtile_ / 8;
                metal::simdgroup_float8x8 LHStile;
                const uint k_id = k + k_offset;
                const uint row_offset = row_tg_offset + row_sg_offset + m_subtile_;
                metal::simdgroup_load(LHStile, lhs, K, ulong2(k_id, row_offset));
                metal::simdgroup_bfloat8x8 RHStile;
#pragma clang loop unroll(full)
                for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
                    const uint col_index_in_out_tile = n_subtile_ / 8;
                    const uint current_index_out_tile =
                        row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;
                    const uint col_offset = col_tg_offset + col_sg_offset + n_subtile_;
                    simdgroup_load(RHStile, rhs, K, ulong2(k_id, col_offset), /*transpose=*/true);
                    simdgroup_multiply_accumulate(OutTiles[current_index_out_tile],
                                                  LHStile, RHStile,
                                                  OutTiles[current_index_out_tile]);
                }
            }
        }
    }
    // Epilogue.
#pragma clang loop unroll(full)
    for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
        const uint col_index_in_out_tile = n_subtile_ / 8;
        const uint local_col_offset = col_sg_offset + n_subtile_;
#pragma clang loop unroll(full)
        for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
            const uint row_index_in_out_tile = m_subtile_ / 8;
            const uint local_row_offset = row_sg_offset + m_subtile_;
            const uint current_index_out_tile =
                row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;
            simdgroup_store(OutTiles[current_index_out_tile], scratch, Bn,
                            ulong2(local_col_offset, local_row_offset));
        }
    }
    // TODO(ibahmed): vectorize these loads an maybe unroll the loop.
    const uint thread_count_per_tg =
        threadgroup_size.x * threadgroup_size.y * threadgroup_size.z;
    for (uint c_local = local_tid.x; c_local < Bn;
         c_local += thread_count_per_tg) {
        const uint c_global = col_tg_offset + c_local;
        bias_tile[c_local] =
            (c_global < N) ? static_cast<float>(bias[c_global]) : 0.0f;
    }

    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const uint q_heads = 64;
    const uint kv_heads = 8;
    const uint head_dim = 64;
    const uint q_cols = q_heads * head_dim;
    const uint k_cols = kv_heads * head_dim;

    // TODO(ibahmed): vectorize these stores and maybe unroll the loop.
    for (uint idx = local_tid.x; idx < Bm * Bn; idx += thread_count_per_tg) {
        const uint r = idx / Bn;
        const uint c = idx % Bn;

        const uint out_row = row_tg_offset + r;
        const uint out_col = col_tg_offset + c;

        if (out_row < M && out_col < N) {
            float acc = scratch[idx] + bias_tile[c];
            if ((out_col < q_cols + k_cols)) {
                out[out_row * N + out_col] = acc;
            } else {
                // Write v into kv cache.
                const uint v_col = out_col - q_cols - k_cols;
                const uint v_head = v_col / head_dim;
                const uint dim_idx = v_col % head_dim;
                const uint token_idx = args.token_offset + out_row;
                kv[(v_head * args.max_tokens + token_idx) * 2 * head_dim + head_dim + dim_idx] = acc;
            }
        }
    }
}

kernel void gptoss_f32_bf16w_dense_matmul_attn_output(
    constant gptoss_dense_matmul_args& args [[buffer(0)]],
    const device float* lhs [[buffer(1)]],
    const device bfloat* rhs [[buffer(2)]],
    const device bfloat* __restrict__ bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    const device gptoss_control* control [[buffer(5)]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_count_per_tg [[dispatch_simdgroups_per_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 local_tid [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]]) {
    threadgroup float scratch[ATTN_OUTPUT_Bm * ATTN_OUTPUT_Bn];
    threadgroup float bias_tile[ATTN_OUTPUT_Bn];
    _gptoss_f32_bf16w_dense_matmul_impl<ATTN_OUTPUT_Bm, ATTN_OUTPUT_Bn,
                                        ATTN_OUTPUT_Bk, ATTN_OUTPUT_Sg_Bm,
                                        ATTN_OUTPUT_Sg_Bn, /*add=*/1>(
        args, lhs, rhs, bias, out, control, scratch, bias_tile, sg_id, sg_count_per_tg,
        gid, tg_id, local_tid, threadgroup_size);
}

kernel void gptoss_f32_bf16w_dense_matmul_mlp_gate(
    constant gptoss_dense_matmul_args& args [[buffer(0)]],
    const device float* lhs [[buffer(1)]],
    const device bfloat* rhs [[buffer(2)]],
    const device bfloat* __restrict__ bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    const device gptoss_control* control [[buffer(5)]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_count_per_tg [[dispatch_simdgroups_per_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 local_tid [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]]) {
    threadgroup float scratch[MLP_GATE_Bm * MLP_GATE_Bn];
    threadgroup float bias_tile[MLP_GATE_Bn];
    _gptoss_f32_bf16w_dense_matmul_impl<MLP_GATE_Bm, MLP_GATE_Bn, MLP_GATE_Bk,
                                        MLP_GATE_Sg_Bm, MLP_GATE_Sg_Bn>(
        args, lhs, rhs, bias, out, control, scratch, bias_tile, sg_id, sg_count_per_tg,
        gid, tg_id, local_tid, threadgroup_size);
}
