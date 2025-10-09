#include <metal_integer>
#include <metal_math>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


kernel void gptoss_f32_accumulate_e4(
    constant gptoss_accumulate_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device gptoss_expert_prediction* expert [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    const device gptoss_control* control [[ buffer(4) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 threadgroup_size [[ threads_per_threadgroup ]])
{
    const uint num_active_experts = 4;
    if (control->abort != 0) {
        return;
    }

    const uint num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const uint threadgroup_start = gid.x * num_vecs_per_threadgroup;
    const uint num_vecs = args.num_vecs;
    const uint threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, num_vecs);
    const uint thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size.x - 1)) / threadgroup_size.x);

    const uint num_vecs_per_expert = args.num_vecs_per_expert;
    const float scale0 = expert[gid.y * num_active_experts + 0].score;
    const device float4* input0 = input + gid.y * num_vecs + thread_start;
    const float scale1 = expert[gid.y * num_active_experts + 1].score;
    const device float4* input1 = input0 + num_vecs_per_expert;
    const float scale2 = expert[gid.y * num_active_experts + 2].score;
    const device float4* input2 = input1 + num_vecs_per_expert;
    const float scale3 = expert[gid.y * num_active_experts + 3].score;
    const device float4* input3 = input2 + num_vecs_per_expert;
    output += gid.y * num_vecs + thread_start;
    for (; num_iter != 0; num_iter--) {
        float4 acc = *output;
        const float4 val0 = *input0;
        const float4 val1 = *input1;
        const float4 val2 = *input2;
        const float4 val3 = *input3;
        input0 += threadgroup_size.x;
        acc = metal::fma(val0, scale0, acc);
        input1 += threadgroup_size.x;
        acc = metal::fma(val1, scale1, acc);
        input2 += threadgroup_size.x;
        acc = metal::fma(val2, scale2, acc);
        input3 += threadgroup_size.x;
        acc = metal::fma(val3, scale3, acc);
        *output = acc;
        output += threadgroup_size.x;
    }
}
