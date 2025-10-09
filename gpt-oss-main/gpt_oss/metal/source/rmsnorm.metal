#include <metal_compute>
#include <metal_math>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


[[max_total_threads_per_threadgroup(1024)]]
kernel void gptoss_f32_bf16w_rmsnorm(
    constant gptoss_rmsnorm_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device bfloat4* weights [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    const device gptoss_control* control [[ buffer(4) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const uint simdgroup_size = 32;
    threadgroup float threadgroup_buffer[32];
    if (control->abort != 0) {
        return;
    }

    input += gid * args.num_vecs;
    output += gid * args.num_vecs;

    float4 sumsq4 = 0.0f;
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        const float4 val = input[i];
        sumsq4 = metal::fma(val, val, sumsq4);
    }

    // Tree-reduce sumsq within thread, then all-reduce within threadgroup.
    const float2 sumsq2 = sumsq4.xy + sumsq4.zw;
    float sumsq = sumsq2.x + sumsq2.y;
    // Warning: this all-reduce works only for simdgroup of 32 threads and threadgroup of 32*32=1024 threads.
    sumsq = metal::simd_sum(sumsq);
    if (metal::simd_is_first()) {
        const uint simdgroup_idx = tid / simdgroup_size;
        threadgroup_buffer[simdgroup_idx] = sumsq;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const uint simdgroup_tid = tid % simdgroup_size;
    sumsq = threadgroup_buffer[simdgroup_tid];
    sumsq = metal::simd_sum(sumsq);

    const float avgsq = sumsq / args.num_channels;
    const float scale = metal::precise::rsqrt(avgsq + args.epsilon);
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        const float4 val = input[i] * scale;
        const float4 weight_val = static_cast<float4>(weights[i]);
        output[i] = val * weight_val;
    }
}
