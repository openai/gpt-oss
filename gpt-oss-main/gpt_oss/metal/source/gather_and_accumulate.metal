#include <internal/kernel-args.h>
#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>

// TODO(ibrahim): This is not optimal as each thread only gathers and accumulates a single float4. To amortize the
// cost of reading the expert, offset and scales for a token, we should let each thread gather and accumulate several
// float4s.
kernel void gptoss_f32_gather_and_accumulate_e4(
    constant gptoss_gather_args& args [[ buffer(0) ]],
    const device float* in [[ buffer(1) ]],
    const device gptoss_expert_prediction* __restrict__ expert_predictions [[ buffer(2) ]],
    const device uint* expert_offsets [[ buffer(3) ]],
    const device uint* intra_expert_offsets [[ buffer(4) ]],
    device float* out [[ buffer(5) ]],
    uint3 gid [[thread_position_in_grid]]) 
{
    const uint T = args.tokens;
    const uint k = args.active_experts_per_token;
    const uint D = args.token_stride;

    assert((D & 3u) == 0);
    assert(k == 4);

    const uint row = gid.y;
    if (row >= T) {
        return;
    }

    const uint col_vec4 = gid.x;
    const uint col = col_vec4 * 4u;
    if (col >= D) {
        return;
    }

    device float4* dst4 = reinterpret_cast<device float4*>(out + row * D + col);

    const uint base = row * k;
    const gptoss_expert_prediction expert0 = expert_predictions[base];
    const gptoss_expert_prediction expert1 = expert_predictions[base + 1];
    const gptoss_expert_prediction expert2 = expert_predictions[base + 2];
    const gptoss_expert_prediction expert3 = expert_predictions[base + 3];
    const uint expert0_id = expert0.expert_id;
    const uint expert1_id = expert1.expert_id;
    const uint expert2_id = expert2.expert_id;
    const uint expert3_id = expert3.expert_id;
    const float scale0 = expert0.score;
    const float scale1 = expert1.score;
    const float scale2 = expert2.score;
    const float scale3 = expert3.score;
    const uint4 current_intra_expert_offsets =
        *reinterpret_cast<const device uint4*>(&intra_expert_offsets[base]);
    // Get the row indices for the current expert ids
    const uint r0 = expert_offsets[expert0_id] + current_intra_expert_offsets.x;
    const uint r1 = expert_offsets[expert1_id] + current_intra_expert_offsets.y;
    const uint r2 = expert_offsets[expert2_id] + current_intra_expert_offsets.z;
    const uint r3 = expert_offsets[expert3_id] + current_intra_expert_offsets.w;

    const device float4* src0 =
        reinterpret_cast<const device float4*>(in + r0 * D + col);
    const device float4* src1 =
        reinterpret_cast<const device float4*>(in + r1 * D + col);
    const device float4* src2 =
        reinterpret_cast<const device float4*>(in + r2 * D + col);
    const device float4* src3 =
        reinterpret_cast<const device float4*>(in + r3 * D + col);

    float4 acc = *dst4;
    acc = metal::fma(*src0, scale0, acc);
    acc = metal::fma(*src1, scale1, acc);
    acc = metal::fma(*src2, scale2, acc);
    acc = metal::fma(*src3, scale3, acc);
    *dst4 = acc;
}