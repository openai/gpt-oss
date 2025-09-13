#include <metal_compute>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


[[max_total_threads_per_threadgroup(32)]]
kernel void gptoss_f32_topk_softmax_e128_k4(
    constant gptoss_topk_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    device gptoss_expert_prediction* output [[ buffer(2) ]],
    const device gptoss_control* control [[ buffer(3) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint num_experts = 128;
    const uint num_active_experts = 4;
    if (control->abort != 0) {
        return;
    }

    input += gid * (num_experts / 4);
    output += gid * num_active_experts;

    uint4 idx = tid * 4 + (uint4) {0, 1, 2, 3};
    float4 val = input[tid];

    const float topval0 = metal::simd_max(metal::max3(metal::max(val.x, val.y), val.z, val.w));
    uint idx0 = 0xFFFFFFFFu;
    if (val.w == topval0) {
        idx0 = idx.w;
    }
    if (val.z == topval0) {
        idx0 = idx.z;
    }
    if (val.y == topval0) {
        idx0 = idx.y;
    }
    if (val.x == topval0) {
        idx0 = idx.x;
    }
    const uint topidx0 = metal::simd_min(idx0);
    const bool4 is_topidx0 = idx == topidx0;
    val = metal::select(val, -INFINITY, is_topidx0);
    idx = metal::select(idx, 0xFFFFFFFFu, is_topidx0);

    const float topval1 = metal::simd_max(metal::max3(metal::max(val.x, val.y), val.z, val.w));
    uint idx1 = 0xFFFFFFFFu;
    if (val.w == topval1) {
        idx1 = idx.w;
    }
    if (val.z == topval1) {
        idx1 = idx.z;
    }
    if (val.y == topval1) {
        idx1 = idx.y;
    }
    if (val.x == topval1) {
        idx1 = idx.x;
    }
    const uint topidx1 = metal::simd_min(idx1);
    const bool4 is_topidx1 = idx == topidx1;
    val = metal::select(val, -INFINITY, is_topidx1);
    idx = metal::select(idx, 0xFFFFFFFFu, is_topidx1);

    const float topval2 = metal::simd_max(metal::max3(metal::max(val.x, val.y), val.z, val.w));
    uint idx2 = 0xFFFFFFFFu;
    if (val.w == topval2) {
        idx2 = idx.w;
    }
    if (val.z == topval2) {
        idx2 = idx.z;
    }
    if (val.y == topval2) {
        idx2 = idx.y;
    }
    if (val.x == topval2) {
        idx2 = idx.x;
    }
    const uint topidx2 = metal::simd_min(idx2);
    const bool4 is_topidx2 = idx == topidx2;
    val = metal::select(val, -INFINITY, is_topidx2);
    idx = metal::select(idx, 0xFFFFFFFFu, is_topidx2);

    const float topval3 = metal::simd_max(metal::max3(metal::max(val.x, val.y), val.z, val.w));
    uint idx3 = 0xFFFFFFFFu;
    if (val.w == topval3) {
        idx3 = idx.w;
    }
    if (val.z == topval3) {
        idx3 = idx.z;
    }
    if (val.y == topval3) {
        idx3 = idx.y;
    }
    if (val.x == topval3) {
        idx3 = idx.x;
    }
    const uint topidx3 = metal::simd_min(idx3);

    if (metal::simd_is_first()) {
        const float topexp0 = 1.0f;
        const float topexp1 = metal::precise::exp(topval1 - topval0);
        const float topexp2 = metal::precise::exp(topval2 - topval0);
        const float topexp3 = metal::precise::exp(topval3 - topval0);

        const float sum = (topexp0 + topexp1) + (topexp2 + topexp3);
        const float scale = 1.0 / sum;

        output[0] = (gptoss_expert_prediction) {
            .expert_id = topidx0,
            .score = topexp0 * scale,
        };
        output[1] = (gptoss_expert_prediction) {
            .expert_id = topidx1,
            .score = topexp1 * scale,
        };
        output[2] = (gptoss_expert_prediction) {
            .expert_id = topidx2,
            .score = topexp2 * scale,
        };
        output[3] = (gptoss_expert_prediction) {
            .expert_id = topidx3,
            .score = topexp3 * scale,
        };
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void gptoss_f32_topk_softmax_e32_k4(
    constant gptoss_topk_args& args [[ buffer(0) ]],
    const device float* input [[ buffer(1) ]],
    device gptoss_expert_prediction* output [[ buffer(2) ]],
    const device gptoss_control* control [[ buffer(3) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint num_experts = 32;
    const uint num_active_experts = 4;
    if (control->abort != 0) {
        return;
    }

    input += gid * num_experts;
    output += gid * num_active_experts;

    float val = input[tid];
    uint idx = tid;

    const float topval0 = metal::simd_max(val);
    const uint topidx0 = metal::simd_min(val == topval0 ? idx : 0xFFFFFFFFu);
    if (idx == topidx0) {
        val = -INFINITY;
        idx = 0xFFFFFFFFu;
    }

    const float topval1 = metal::simd_max(val);
    const uint topidx1 = metal::simd_min(val == topval1 ? idx : 0xFFFFFFFFu);
    if (idx == topidx1) {
        val = -INFINITY;
        idx = 0xFFFFFFFFu;
    }

    const float topval2 = metal::simd_max(val);
    const uint topidx2 = metal::simd_min(val == topval2 ? idx : 0xFFFFFFFFu);
    if (idx == topidx2) {
        val = -INFINITY;
        idx = 0xFFFFFFFFu;
    }

    const float topval3 = metal::simd_max(val);
    const uint topidx3 = metal::simd_min(val == topval3 ? idx : 0xFFFFFFFFu);

    if (metal::simd_is_first()) {
        const float topexp0 = 1.0f;
        const float topexp1 = metal::precise::exp(topval1 - topval0);
        const float topexp2 = metal::precise::exp(topval2 - topval0);
        const float topexp3 = metal::precise::exp(topval3 - topval0);

        const float sum = (topexp0 + topexp1) + (topexp2 + topexp3);
        const float scale = 1.0 / sum;

        output[0] = (gptoss_expert_prediction) {
            .expert_id = topidx0,
            .score = topexp0 * scale,
        };
        output[1] = (gptoss_expert_prediction) {
            .expert_id = topidx1,
            .score = topexp1 * scale,
        };
        output[2] = (gptoss_expert_prediction) {
            .expert_id = topidx2,
            .score = topexp2 * scale,
        };
        output[3] = (gptoss_expert_prediction) {
            .expert_id = topidx3,
            .score = topexp3 * scale,
        };
    }
}
