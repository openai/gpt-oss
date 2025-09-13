#include <metal_common>
#include <metal_compute>
#include <metal_math>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


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
