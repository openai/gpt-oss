#include <internal/kernel-args.h>
#include <metal_common>
#include <metal_compute>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#define ceil_div(a, b) (((a) + (b) - 1) / (b))

// Each simdgroup reduces all channels of the input and computes a single channel of the output
// + Efficient synchronization
// + Sequential memory access within a warp
// Each threadgroup computes (simdgroups_per_threadgroup) consecutive output channels
// + Reuse input vector from threadgroup memory
// + Avoid synchronization across warps when doing reduction

kernel void gptoss_f32_mf4w_moe_matmul_swiglu(
    constant gptoss_moe_matmul_swiglu_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device gptoss_expert_prediction* expert [[ buffer(2) ]],
    const device uint4* weight_blocks [[ buffer(3) ]],
    const device uchar* weight_scales [[ buffer(4) ]],
    const device bfloat* bias [[ buffer(5) ]],
    device float* output [[ buffer(6) ]],
    const device gptoss_control* control [[ buffer(7) ]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;
    threadgroup float threadgroup_buffer[32];
    if (control->abort != 0) {
        return;
    }

    const uint num_column_vecs = args.num_column_vecs;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;
    const uint expert_id = expert[gid.y * args.num_active_experts + gid.z].expert_id;

    input += 8 * (gid.y * num_column_vecs + simdgroup_tid);
    weight_blocks = (const device uint4*) ((uintptr_t) (weight_blocks + num_column_vecs * row + simdgroup_tid) + expert_id * args.weight_expert_stride);
    weight_scales = (const device uchar*) ((uintptr_t) (weight_scales + num_column_vecs * row + simdgroup_tid) + expert_id * args.weight_expert_stride);
    bias = (const device bfloat*) ((uintptr_t) (bias + row) + expert_id * args.weight_expert_stride);
    output += gid.y * args.num_rows + gid.x * (num_simdgroups / 2) + gid.z * args.output_expert_stride;

    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    float4 sum4 = 0.0f;
    do {
        const uint4 wblock = *weight_blocks;
        const float wscale = as_type<float>(static_cast<uint>(*weight_scales) << 23);
        uint4 wblock02468ACEGIKMOQSU = wblock + wblock;
        uint4 wblock13579BDFHJLNPRTV = wblock >> 3;
        wblock02468ACEGIKMOQSU &= 0x1E1E1E1Eu;
        wblock13579BDFHJLNPRTV &= 0x1E1E1E1Eu;
        wblock02468ACEGIKMOQSU += 0x70707070u;
        wblock13579BDFHJLNPRTV += 0x70707070u;
        wblock02468ACEGIKMOQSU &= 0x8E8E8E8Eu;
        wblock13579BDFHJLNPRTV &= 0x8E8E8E8Eu;
        const uint4 wblock26AEIMQU = wblock02468ACEGIKMOQSU & 0xFF00FF00u;
        const uint4 wblock048CGKOS = (wblock02468ACEGIKMOQSU << 8) & 0xFF00FF00u;
        const uint4 wblock37BFJNRV = wblock13579BDFHJLNPRTV & 0xFF00FF00u;
        const uint4 wblock159DHLPT = (wblock13579BDFHJLNPRTV << 8) & 0xFF00FF00u;
        const float4 w048C = static_cast<float4>(as_type<half4>(wblock048CGKOS.xy));
        const float4 wGKOS = static_cast<float4>(as_type<half4>(wblock048CGKOS.zw));
        const float4 w26AE = static_cast<float4>(as_type<half4>(wblock26AEIMQU.xy));
        const float4 wIMQU = static_cast<float4>(as_type<half4>(wblock26AEIMQU.zw));
        const float4 w159D = static_cast<float4>(as_type<half4>(wblock159DHLPT.xy));
        const float4 wHLPT = static_cast<float4>(as_type<half4>(wblock159DHLPT.zw));
        const float4 w37BF = static_cast<float4>(as_type<half4>(wblock37BFJNRV.xy));
        const float4 wJNRV = static_cast<float4>(as_type<half4>(wblock37BFJNRV.zw));

        const float4 w0123 = (float4) { w048C.x, w159D.x, w26AE.x, w37BF.x };
        const float4 w4567 = (float4) { w048C.y, w159D.y, w26AE.y, w37BF.y };
        const float4 w89AB = (float4) { w048C.z, w159D.z, w26AE.z, w37BF.z };
        const float4 wCDEF = (float4) { w048C.w, w159D.w, w26AE.w, w37BF.w };
        const float4 wGHIJ = (float4) { wGKOS.x, wHLPT.x, wIMQU.x, wJNRV.x };
        const float4 wKLMN = (float4) { wGKOS.y, wHLPT.y, wIMQU.y, wJNRV.y };
        const float4 wOPQR = (float4) { wGKOS.z, wHLPT.z, wIMQU.z, wJNRV.z };
        const float4 wSTUV = (float4) { wGKOS.w, wHLPT.w, wIMQU.w, wJNRV.w };

        const float4 i0123 = input[0];
        const float4 i4567 = input[1];
        const float4 i89AB = input[2];
        const float4 iCDEF = input[3];
        const float4 iGHIJ = input[4];
        const float4 iKLMN = input[5];
        const float4 iOPQR = input[6];
        const float4 iSTUV = input[7];

        float4 psum0 = i0123 * w0123;
        float4 psum1 = i4567 * w4567;
        psum0 = metal::fma(i89AB, w89AB, psum0);
        psum1 = metal::fma(iCDEF, wCDEF, psum1);
        psum0 = metal::fma(iGHIJ, wGHIJ, psum0);
        psum1 = metal::fma(iKLMN, wKLMN, psum1);
        psum0 = metal::fma(iOPQR, wOPQR, psum0);
        psum1 = metal::fma(iSTUV, wSTUV, psum1);
        sum4 = metal::fma(psum0, wscale, sum4);
        sum4 = metal::fma(psum1, wscale, sum4);

        weight_blocks += simdgroup_size;
        weight_scales += simdgroup_size;
        input += 8 * simdgroup_size;
    } while (--num_iter != 0);
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    if (metal::simd_is_first()) {
        sum += static_cast<float>(*bias);
        threadgroup_buffer[simdgroup_idx] = sum;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (tid * 2 < num_simdgroups) {
        const float2 x = reinterpret_cast<const threadgroup float2*>(threadgroup_buffer)[tid];
        const float swish_x = metal::min(x.x, args.swiglu_max);
        const float linear_x = metal::clamp(x.y, args.swiglu_min, args.swiglu_max);
        const float alpha = 1.702f;
        const float swish_y = swish_x / (1.0f + metal::precise::exp(-alpha * swish_x));
        const float swiglu_y = metal::fma(swish_y, linear_x, swish_y);
        output[tid] = swiglu_y;
    }
}

kernel void gptoss_f32_mf4w_moe_matmul(
    constant gptoss_moe_matmul_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device gptoss_expert_prediction* expert [[ buffer(2) ]],
    const device uint4* weight_blocks [[ buffer(3) ]],
    const device uchar* weight_scales [[ buffer(4) ]],
    const device bfloat* bias [[ buffer(5) ]],
    device float* output [[ buffer(6) ]],
    const device gptoss_control* control [[ buffer(7) ]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
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
    const uint expert_id = expert[gid.y * args.num_active_experts + gid.z].expert_id;

    input += 8 * (gid.y * num_column_vecs + simdgroup_tid + gid.z * args.input_expert_stride);
    weight_blocks = (const device uint4*) ((uintptr_t) (weight_blocks + num_column_vecs * row + simdgroup_tid) + expert_id * args.weight_expert_stride);
    weight_scales = (const device uchar*) ((uintptr_t) (weight_scales + num_column_vecs * row + simdgroup_tid) + expert_id * args.weight_expert_stride);
    bias = (const device bfloat*) ((uintptr_t) (bias + row) + expert_id * args.weight_expert_stride);
    output += gid.y * args.num_rows + row + gid.z * args.output_expert_stride;

    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    float4 sum4 = 0.0f;
    do {
        const uint4 wblock = *weight_blocks;
        const float wscale = as_type<float>(static_cast<uint>(*weight_scales) << 23);
        uint4 wblock02468ACEGIKMOQSU = wblock + wblock;
        uint4 wblock13579BDFHJLNPRTV = wblock >> 3;
        wblock02468ACEGIKMOQSU &= 0x1E1E1E1Eu;
        wblock13579BDFHJLNPRTV &= 0x1E1E1E1Eu;
        wblock02468ACEGIKMOQSU += 0x70707070u;
        wblock13579BDFHJLNPRTV += 0x70707070u;
        wblock02468ACEGIKMOQSU &= 0x8E8E8E8Eu;
        wblock13579BDFHJLNPRTV &= 0x8E8E8E8Eu;
        const uint4 wblock26AEIMQU = wblock02468ACEGIKMOQSU & 0xFF00FF00u;
        const uint4 wblock048CGKOS = (wblock02468ACEGIKMOQSU << 8) & 0xFF00FF00u;
        const uint4 wblock37BFJNRV = wblock13579BDFHJLNPRTV & 0xFF00FF00u;
        const uint4 wblock159DHLPT = (wblock13579BDFHJLNPRTV << 8) & 0xFF00FF00u;
        const float4 w048C = static_cast<float4>(as_type<half4>(wblock048CGKOS.xy));
        const float4 wGKOS = static_cast<float4>(as_type<half4>(wblock048CGKOS.zw));
        const float4 w26AE = static_cast<float4>(as_type<half4>(wblock26AEIMQU.xy));
        const float4 wIMQU = static_cast<float4>(as_type<half4>(wblock26AEIMQU.zw));
        const float4 w159D = static_cast<float4>(as_type<half4>(wblock159DHLPT.xy));
        const float4 wHLPT = static_cast<float4>(as_type<half4>(wblock159DHLPT.zw));
        const float4 w37BF = static_cast<float4>(as_type<half4>(wblock37BFJNRV.xy));
        const float4 wJNRV = static_cast<float4>(as_type<half4>(wblock37BFJNRV.zw));

        const float4 w0123 = (float4) { w048C.x, w159D.x, w26AE.x, w37BF.x };
        const float4 w4567 = (float4) { w048C.y, w159D.y, w26AE.y, w37BF.y };
        const float4 w89AB = (float4) { w048C.z, w159D.z, w26AE.z, w37BF.z };
        const float4 wCDEF = (float4) { w048C.w, w159D.w, w26AE.w, w37BF.w };
        const float4 wGHIJ = (float4) { wGKOS.x, wHLPT.x, wIMQU.x, wJNRV.x };
        const float4 wKLMN = (float4) { wGKOS.y, wHLPT.y, wIMQU.y, wJNRV.y };
        const float4 wOPQR = (float4) { wGKOS.z, wHLPT.z, wIMQU.z, wJNRV.z };
        const float4 wSTUV = (float4) { wGKOS.w, wHLPT.w, wIMQU.w, wJNRV.w };

        const float4 i0123 = input[0];
        const float4 i4567 = input[1];
        const float4 i89AB = input[2];
        const float4 iCDEF = input[3];
        const float4 iGHIJ = input[4];
        const float4 iKLMN = input[5];
        const float4 iOPQR = input[6];
        const float4 iSTUV = input[7];

        float4 psum0 = i0123 * w0123;
        float4 psum1 = i4567 * w4567;
        psum0 = metal::fma(i89AB, w89AB, psum0);
        psum1 = metal::fma(iCDEF, wCDEF, psum1);
        psum0 = metal::fma(iGHIJ, wGHIJ, psum0);
        psum1 = metal::fma(iKLMN, wKLMN, psum1);
        psum0 = metal::fma(iOPQR, wOPQR, psum0);
        psum1 = metal::fma(iSTUV, wSTUV, psum1);
        sum4 = metal::fma(psum0, wscale, sum4);
        sum4 = metal::fma(psum1, wscale, sum4);

        weight_blocks += simdgroup_size;
        weight_scales += simdgroup_size;
        input += 8 * simdgroup_size;
    } while (--num_iter != 0);
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    if (metal::simd_is_first()) {
        sum += static_cast<float>(*bias);
        *output = sum;
    }
}

kernel void gptoss_f32_mf4w_moe_dense_matmul_swiglu(
    constant gptoss_moe_dense_matmul_swiglu_args& params [[ buffer(0) ]],
    const device uint* __restrict__ expert_offsets [[ buffer(1) ]],
    const device float* lhs [[ buffer(2) ]],
    const device uint* weight_blocks [[ buffer(3) ]],
    const device uchar* weight_scales [[ buffer(4) ]],
    const device bfloat* __restrict__ bias [[ buffer(5) ]],
    device float* out [[ buffer(6) ]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint3 threads_per_tg [[threads_per_threadgroup]],
    uint sg_count_per_tg [[dispatch_simdgroups_per_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 local_tid [[thread_position_in_threadgroup]]) 
{
    constexpr uint Bm = MOE_DENSE_MATMUL_SWIGLU_Bm;
    constexpr uint Bn = MOE_DENSE_MATMUL_SWIGLU_Bn;
    constexpr uint Bk = MOE_DENSE_MATMUL_SWIGLU_Bk;
    constexpr uint Sg_Bm = MOE_DENSE_MATMUL_SWIGLU_Sg_Bm;
    constexpr uint Sg_Bn = MOE_DENSE_MATMUL_SWIGLU_Sg_Bn;

    // Assumptions about shapes.
    assert(Bm % 8 == 0);
    assert(Bn % 8 == 0);
    assert(Bk % 8 == 0);
    assert(Sg_Bm % 8 == 0);
    assert(Sg_Bn % 8 == 0);
    assert(Bm % Sg_Bm == 0);
    assert(Bn % Sg_Bn == 0);

    const uint K = params.k;
    const uint N = params.n;
    const uint M = expert_offsets[tg_id.z + 1] - expert_offsets[tg_id.z];
    assert((K % 32) == 0);
    assert((K % 8) == 0);
    assert(N % Bn == 0);
    assert(K % Bk == 0);
    // Get row and col tg.
    const uint row_tg = tg_id.y;
    const uint col_tg = tg_id.x;
    // Get row and col local tid.
    const uint row_tg_offset = row_tg * Bm;
    const uint col_tg_offset = col_tg * Bn;
    if (row_tg_offset >= M || col_tg_offset >= N) {
        return;
    }
    // Move lhs and output according to the passed offset.
    const uint expert_offset = expert_offsets[tg_id.z];
    lhs += expert_offset * K;
    const uint N_output = N / 2;
    out += expert_offset * N_output;

    const uint S = params.weight_blocks_expert_stride_bytes;
    const uint S_scales = params.weight_scales_expert_stride_bytes;
    const uint S_bias = params.bias_expert_stride_bytes;

    const device char* wb0 = reinterpret_cast<const device char*>(weight_blocks);
    const device char* sc0 = reinterpret_cast<const device char*>(weight_scales);
    const device char* bi0 = reinterpret_cast<const device char*>(bias);

    weight_blocks = reinterpret_cast<const device uint*>(wb0 + tg_id.z * S);
    weight_scales = reinterpret_cast<const device uchar*>(sc0 + tg_id.z * S_scales);
    bias = reinterpret_cast<const device bfloat*>(bi0 + tg_id.z * S_bias);

    const uint sg_col_count = Bn / Sg_Bn;
    const uint row_sg = sg_id / sg_col_count;
    const uint col_sg = sg_id % sg_col_count;

    const uint row_sg_offset = row_sg * Sg_Bm;
    const uint col_sg_offset = col_sg * Sg_Bn;
    // Declare threadgroup blocks.
    threadgroup float lhs_block[Bm * Bk];
    // rhs_block will hold the scaled fp32 weights.
    threadgroup float rhs_block[Bn * Bk];

    constexpr uint temp_result_size = (Sg_Bm / 8) * (Sg_Bn / 8);
    // Create an array of simdgroup_float8x8 to hold temp results.
    metal::simdgroup_float8x8 OutTiles[temp_result_size];
    for (uint i = 0; i < temp_result_size; i++) {
        OutTiles[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0);
    }
    // Linear thread id within TG (we launch 1-D TGs)
    const uint lin_tid = local_tid.x;
    const uint thread_count_per_tg = threads_per_tg.x * threads_per_tg.y * threads_per_tg.z;

    // Iterate over all Bk blocks.
    for (uint k_offset = 0; k_offset < K; k_offset += Bk) {
        constexpr uint lhs_row_stride = Bk;
        constexpr uint lhs_vec_cols = Bk / 4;
        constexpr uint lhs_vec_total = Bm * lhs_vec_cols;

        const uint LHS_ITERS = ceil_div(lhs_vec_total, thread_count_per_tg);

        // #pragma clang loop unroll(full)
        for (uint t = 0; t < LHS_ITERS; ++t) {
            const uint i = t * thread_count_per_tg + lin_tid;
            if (i < lhs_vec_total) {
                const uint r = i / lhs_vec_cols;
                const uint c4 = i % lhs_vec_cols;

                const uint gr = row_tg_offset + r;
                const uint gc4 = (k_offset / 4) + c4;

                threadgroup float4* dst4 =
                    reinterpret_cast<threadgroup float4*>(lhs_block + r * lhs_row_stride + (c4 << 2));
                if (gr < M) {
                    const device float4* src4 =
                        reinterpret_cast<const device float4*>(lhs + gr * K + (gc4 << 2));

                    *dst4 = *src4;
                } else {
                    *dst4 = float4(0.0);
                }
            }
        }

        // Load weights with vector loads.
        constexpr uint rhs_row_stride = Bk;
        constexpr uint weights_per_elem = 8;
        constexpr uint rhs_loads_per_col = Bk / weights_per_elem;
        constexpr uint rhs_loads_total = Bn * rhs_loads_per_col;
        const uint RHS_ITERS = ceil_div(rhs_loads_total, thread_count_per_tg);
        // #pragma clang loop unroll(full)
        for (uint t = 0; t < RHS_ITERS; ++t) {
            const uint i = t * thread_count_per_tg + lin_tid;
            if (i < rhs_loads_total) {
                const uint r = i / rhs_loads_per_col;
                const uint c = i % rhs_loads_per_col;

                const uint gr = col_tg_offset + r;
                const uint gc = (k_offset / weights_per_elem) + c;
                const uint gc_scale = (k_offset / 32) + (c >> 2);

                const uint wblock = weight_blocks[gr * (K / weights_per_elem) + gc];
                const float scale =
                    as_type<float>(static_cast<uint>(weight_scales[gr * (K / 32) + gc_scale]) << 23);
                uint wblock0246 = (wblock + wblock);
                uint wblock1357 = (wblock >> 3);
                wblock0246 &= 0x1E1E1E1Eu;
                wblock1357 &= 0x1E1E1E1Eu;

                wblock0246 += 0x70707070u;
                wblock1357 += 0x70707070u;
                wblock0246 &= 0x8E8E8E8Eu;
                wblock1357 &= 0x8E8E8E8Eu;

                uint wblock26 = (wblock0246) & 0xFF00FF00u;
                uint wblock04 = ((wblock0246 << 8)) & 0xFF00FF00u;
                uint wblock37 = (wblock1357) & 0xFF00FF00u;
                uint wblock15 = ((wblock1357 << 8)) & 0xFF00FF00u;

                half4 wblock0426 = as_type<half4>(uint2(wblock04, wblock26));
                half4 wblock1537 = as_type<half4>(uint2(wblock15, wblock37));

                // Convert to float scalars and apply scale
                const float w0 = float(wblock0426.x) * scale;
                const float w1 = float(wblock1537.x) * scale;
                const float w2 = float(wblock0426.z) * scale;
                const float w3 = float(wblock1537.z) * scale;
                const float w4 = float(wblock0426.y) * scale;
                const float w5 = float(wblock1537.y) * scale;
                const float w6 = float(wblock0426.w) * scale;
                const float w7 = float(wblock1537.w) * scale;
                const uint rhs_offset = r * rhs_row_stride + c * 8;
                rhs_block[rhs_offset] = w0;
                rhs_block[rhs_offset + 1] = w1;
                rhs_block[rhs_offset + 2] = w2;
                rhs_block[rhs_offset + 3] = w3;
                rhs_block[rhs_offset + 4] = w4;
                rhs_block[rhs_offset + 5] = w5;
                rhs_block[rhs_offset + 6] = w6;
                rhs_block[rhs_offset + 7] = w7;
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint k = 0; k < Bk; k += 8) {
#pragma clang loop unroll(full)
            for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
                const uint row_index_in_out_tile = m_subtile_ / 8;
                metal::simdgroup_float8x8 lhs_frag;

                simdgroup_load(lhs_frag, lhs_block, Bk, ulong2(k, m_subtile_ + row_sg_offset));
#pragma clang loop unroll(full)
                for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
                    const uint col_index_in_out_tile = n_subtile_ / 8;
                    const uint current_index_out_tile =
                        row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;
                    metal::simdgroup_float8x8 rhs_frag;
                    simdgroup_load(rhs_frag, rhs_block, Bk, ulong2(k, n_subtile_ + col_sg_offset), true);

                    simdgroup_multiply_accumulate(OutTiles[current_index_out_tile], lhs_frag, rhs_frag,
                        OutTiles[current_index_out_tile]);
                }
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Epilogue.
    threadgroup float scratch[Bm * Bn];
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
    threadgroup float bias_tile[Bn];
    // TODO(ibahmed): vectorize these loads an maybe unroll the loop.
    for (uint c_local = local_tid.x; c_local < Bn; c_local += thread_count_per_tg) {
        const uint c_global = col_tg_offset + c_local;
        bias_tile[c_local] = (c_global < N) ? static_cast<float>(bias[c_global]) : 0.0f;
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const float alpha = 1.702f;
    // TODO(ibahmed): vectorize these stores and maybe unroll the loop.
    for (uint idx = local_tid.x; idx < Bm * Bn / 2; idx += thread_count_per_tg) {
        const uint idx_swish = idx * 2;
        const uint r = idx_swish / Bn;
        const uint c_swish = idx_swish % Bn;

        const uint out_row = row_tg_offset + r;
        const uint out_col = (col_tg_offset / 2) + (c_swish / 2);

        if (out_row < M && out_col < N_output) {
            float acc_swish = scratch[idx_swish] + bias_tile[c_swish];
            float acc_linear = scratch[idx_swish + 1] + bias_tile[c_swish + 1];
            const float swish = metal::min(acc_swish, params.swiglu_max);
            const float linear = metal::clamp(acc_linear, params.swiglu_min, params.swiglu_max);
            const float swish_y = swish / (1.0f + metal::precise::exp(-alpha * swish));
            const float swiglu_y = metal::fma(swish_y, linear, swish_y);
            out[out_row * N_output + out_col] = swiglu_y;
        }
    }
}

kernel void gptoss_f32_mf4w_moe_dense_matmul(
    constant gptoss_moe_dense_matmul_args& params [[ buffer(0) ]],
    const device uint* __restrict__ expert_offsets [[ buffer(1) ]],
    const device float* lhs [[ buffer(2) ]],
    const device uint* weight_blocks [[ buffer(3) ]],
    const device uchar* weight_scales [[ buffer(4) ]],
    const device bfloat* __restrict__ bias [[ buffer(5) ]],
    device float* out [[ buffer(6) ]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint3 threads_per_tg [[threads_per_threadgroup]],
    uint sg_count_per_tg [[dispatch_simdgroups_per_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 local_tid [[thread_position_in_threadgroup]]) 
{
    const uint Bm = MOE_DENSE_MATMUL_Bm;
    const uint Bn = MOE_DENSE_MATMUL_Bn;
    const uint Bk = MOE_DENSE_MATMUL_Bk;
    const uint Sg_Bm = MOE_DENSE_MATMUL_Sg_Bm;
    const uint Sg_Bn = MOE_DENSE_MATMUL_Sg_Bn;
    assert(Bm % 8 == 0);
    assert(Bn % 8 == 0);
    assert(Bk % 8 == 0);
    assert(Sg_Bm % 8 == 0);
    assert(Sg_Bn % 8 == 0);
    assert(Bm % Sg_Bm == 0);
    assert(Bn % Sg_Bn == 0);

    const uint K = params.k;
    const uint N = params.n;
    const uint M = expert_offsets[tg_id.z + 1] - expert_offsets[tg_id.z];
    assert((K % 32) == 0);
    assert((K % 8) == 0);
    assert(N % Bn == 0);
    assert(K % Bk == 0);
    // Get row and col tg.
    const uint row_tg = tg_id.y;
    const uint col_tg = tg_id.x;
    // Get row and col local tid.
    const uint row_tg_offset = row_tg * Bm;
    const uint col_tg_offset = col_tg * Bn;
    if (row_tg_offset >= M || col_tg_offset >= N) {
        return;
    }
    // Move lhs and output according to the passed offset.
    const uint expert_offset = expert_offsets[tg_id.z];
    lhs += expert_offset * K;
    out += expert_offset * N;

    const uint S = params.weight_blocks_expert_stride_bytes;
    const uint S_scales = params.weight_scales_expert_stride_bytes;
    const uint S_bias = params.bias_expert_stride_bytes;

    const device char* wb0 = reinterpret_cast<const device char*>(weight_blocks);
    const device char* sc0 = reinterpret_cast<const device char*>(weight_scales);
    const device char* bi0 = reinterpret_cast<const device char*>(bias);

    weight_blocks = reinterpret_cast<const device uint*>(wb0 + tg_id.z * S);
    weight_scales = reinterpret_cast<const device uchar*>(sc0 + tg_id.z * S_scales);
    bias = reinterpret_cast<const device bfloat*>(bi0 + tg_id.z * S_bias);

    const uint sg_col_count = Bn / Sg_Bn;
    const uint row_sg = sg_id / sg_col_count;
    const uint col_sg = sg_id % sg_col_count;

    const uint row_sg_offset = row_sg * Sg_Bm;
    const uint col_sg_offset = col_sg * Sg_Bn;
    // Declare threadgroup blocks.
    threadgroup float lhs_block[Bm * Bk];
    // rhs_block will hold the scaled fp32 weights.
    threadgroup float rhs_block[Bn * Bk];

    constexpr uint temp_result_size = (Sg_Bm / 8) * (Sg_Bn / 8);
    // Create an array of simdgroup_float8x8 to hold temp results.
    metal::simdgroup_float8x8 OutTiles[temp_result_size];
    for (uint i = 0; i < temp_result_size; i++) {
        OutTiles[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0);
    }
    // Linear thread id within TG (we launch 1-D TGs)
    const uint lin_tid = local_tid.x;

    const uint thread_count_per_tg = threads_per_tg.x * threads_per_tg.y * threads_per_tg.z;
    // Iterate over all Bk blocks.
    for (uint k_offset = 0; k_offset < K; k_offset += Bk) {
        constexpr uint lhs_row_stride = Bk;
        constexpr uint lhs_vec_cols = Bk / 4;
        constexpr uint lhs_vec_total = Bm * lhs_vec_cols;

        const uint LHS_ITERS = ceil_div(lhs_vec_total, thread_count_per_tg);

        for (uint t = 0; t < LHS_ITERS; ++t) {
            const uint i = t * thread_count_per_tg + lin_tid;
            if (i < lhs_vec_total) {
                const uint r = i / lhs_vec_cols;
                const uint c4 = i % lhs_vec_cols;

                const uint gr = row_tg_offset + r;
                const uint gc4 = (k_offset / 4) + c4;

                threadgroup float4* dst4 =
                    reinterpret_cast<threadgroup float4*>(lhs_block + r * lhs_row_stride + (c4 << 2));
                if (gr < M) {
                    const device float4* src4 =
                        reinterpret_cast<const device float4*>(lhs + gr * K + (gc4 << 2));

                    *dst4 = *src4;
                } else {
                    *dst4 = float4(0.0);
                }
            }
        }

        // Load weights with vector loads.
        constexpr uint rhs_row_stride = Bk;
        constexpr uint weights_per_elem = 8;
        constexpr uint rhs_loads_per_col = Bk / weights_per_elem;
        constexpr uint rhs_loads_total = Bn * rhs_loads_per_col;
        const uint RHS_ITERS = ceil_div(rhs_loads_total, thread_count_per_tg);
        // #pragma clang loop unroll(full)
        for (uint t = 0; t < RHS_ITERS; ++t) {
            const uint i = t * thread_count_per_tg + lin_tid;
            if (i < rhs_loads_total) {
                const uint r = i / rhs_loads_per_col;
                const uint c = i % rhs_loads_per_col;

                const uint gr = col_tg_offset + r;
                const uint gc = (k_offset / weights_per_elem) + c;
                const uint gc_scale = (k_offset / 32) + (c >> 2);

                const uint wblock = weight_blocks[gr * (K / weights_per_elem) + gc];
                const float scale =
                    as_type<float>(static_cast<uint>(weight_scales[gr * (K / 32) + gc_scale]) << 23);

                uint wblock0246 = (wblock + wblock);
                uint wblock1357 = (wblock >> 3);
                wblock0246 &= 0x1E1E1E1Eu;
                wblock1357 &= 0x1E1E1E1Eu;

                wblock0246 += 0x70707070u;
                wblock1357 += 0x70707070u;
                wblock0246 &= 0x8E8E8E8Eu;
                wblock1357 &= 0x8E8E8E8Eu;

                uint wblock26 = (wblock0246) & 0xFF00FF00u;
                uint wblock04 = ((wblock0246 << 8)) & 0xFF00FF00u;
                uint wblock37 = (wblock1357) & 0xFF00FF00u;
                uint wblock15 = ((wblock1357 << 8)) & 0xFF00FF00u;

                half4 wblock0426 = as_type<half4>(uint2(wblock04, wblock26));
                half4 wblock1537 = as_type<half4>(uint2(wblock15, wblock37));

                const float w0 = float(wblock0426.x) * scale;
                const float w1 = float(wblock1537.x) * scale;
                const float w2 = float(wblock0426.z) * scale;
                const float w3 = float(wblock1537.z) * scale;
                const float w4 = float(wblock0426.y) * scale;
                const float w5 = float(wblock1537.y) * scale;
                const float w6 = float(wblock0426.w) * scale;
                const float w7 = float(wblock1537.w) * scale;
                const uint rhs_offset = r * rhs_row_stride + c * 8;
                rhs_block[rhs_offset] = w0;
                rhs_block[rhs_offset + 1] = w1;
                rhs_block[rhs_offset + 2] = w2;
                rhs_block[rhs_offset + 3] = w3;
                rhs_block[rhs_offset + 4] = w4;
                rhs_block[rhs_offset + 5] = w5;
                rhs_block[rhs_offset + 6] = w6;
                rhs_block[rhs_offset + 7] = w7;
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint k = 0; k < Bk; k += 8) {
#pragma clang loop unroll(full)
            for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
                const uint row_index_in_out_tile = m_subtile_ / 8;
                metal::simdgroup_float8x8 lhs_frag;

                simdgroup_load(lhs_frag, lhs_block, Bk, ulong2(k, m_subtile_ + row_sg_offset));
#pragma clang loop unroll(full)
                for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
                    const uint col_index_in_out_tile = n_subtile_ / 8;
                    const uint current_index_out_tile =
                        row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;
                    metal::simdgroup_float8x8 rhs_frag;
                    simdgroup_load(rhs_frag, rhs_block, Bk, ulong2(k, n_subtile_ + col_sg_offset), true);
                    simdgroup_multiply_accumulate(OutTiles[current_index_out_tile], lhs_frag, rhs_frag,
                        OutTiles[current_index_out_tile]);
                }
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Epilogue.
    threadgroup float scratch[Bm * Bn];
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
    threadgroup float bias_tile[Bn];
    for (uint c_local = local_tid.x; c_local < Bn; c_local += thread_count_per_tg) {
        const uint c_global = col_tg_offset + c_local;
        bias_tile[c_local] = (c_global < N) ? static_cast<float>(bias[c_global]) : 0.0f;
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint idx = local_tid.x; idx < Bm * Bn; idx += thread_count_per_tg) {
        const uint r = idx / Bn;
        const uint c = idx % Bn;

        const uint out_row = row_tg_offset + r;
        const uint out_col = col_tg_offset + c;

        if (out_row < M && out_col < N) {
            float acc = scratch[idx] + bias_tile[c];
            out[out_row * N + out_col] = acc;
        }
    }
}
