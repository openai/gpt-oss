#include <internal/kernel-args.h>
#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>

// TODO(ibrahim): This is not optimal as each thread only scatters a single float4. To amortize the
// cost of reading the expert id and offset for a token, we should let each thread scatter several
// float4s.
kernel void gptoss_f32_scatter_e4(
    constant gptoss_scatter_args& args [[ buffer(0) ]],
    const device float* in [[ buffer(1) ]],
    const device gptoss_expert_prediction* __restrict__ expert_predictions [[ buffer(2) ]],
    const device uint* __restrict__ expert_offsets [[ buffer(3) ]],
    const device uint* __restrict__ intra_expert_offsets [[ buffer(4) ]],
    device float* out [[ buffer(5) ]],
    uint3 gid [[thread_position_in_grid]]) 
{
    const uint total_tokens = args.tokens;
    const uint active_experts_per_token = args.active_experts_per_token;
    const uint embedding_dim = args.token_stride;
    assert(embedding_dim % 4 == 0);
    // Hard coded to top4 for now.
    assert(active_experts_per_token == 4);
    const uint row_in = gid.y;
    if (row_in >= total_tokens) {
        return;
    }
    // Consecutive threads in a tg read consecutive columns of the input.
    const uint col_in_vec4 = gid.x;
    const uint col_in = col_in_vec4 * 4;
    if (col_in >= embedding_dim) {
        return;
    }
    // Pointer to the piece of the input that we will copy to the top4 experts.
    const device float4* src4 =
        reinterpret_cast<const device float4*>(in + row_in * embedding_dim + col_in);

    // Get the 4 destinations -- 4 experts.
    const uint base = row_in * active_experts_per_token;
    const uint expert0_id = expert_predictions[base].expert_id;
    const uint expert1_id = expert_predictions[base + 1].expert_id;
    const uint expert2_id = expert_predictions[base + 2].expert_id;
    const uint expert3_id = expert_predictions[base + 3].expert_id;
    const uint expert0_offset = expert_offsets[expert0_id];
    const uint expert1_offset = expert_offsets[expert1_id];
    const uint expert2_offset = expert_offsets[expert2_id];
    const uint expert3_offset = expert_offsets[expert3_id];
    const uint expert0_intra_expert_offset = intra_expert_offsets[base];
    const uint expert1_intra_expert_offset = intra_expert_offsets[base + 1];
    const uint expert2_intra_expert_offset = intra_expert_offsets[base + 2];
    const uint expert3_intra_expert_offset = intra_expert_offsets[base + 3];
    device float4* dst4_0 = reinterpret_cast<device float4*>(
        out + (expert0_offset + expert0_intra_expert_offset) * embedding_dim + col_in);
    device float4* dst4_1 = reinterpret_cast<device float4*>(
        out + (expert1_offset + expert1_intra_expert_offset) * embedding_dim + col_in);
    device float4* dst4_2 = reinterpret_cast<device float4*>(
        out + (expert2_offset + expert2_intra_expert_offset) * embedding_dim + col_in);
    device float4* dst4_3 = reinterpret_cast<device float4*>(
        out + (expert3_offset + expert3_intra_expert_offset) * embedding_dim + col_in);
    const float4 data = *src4;
    *dst4_0 = data;
    *dst4_1 = data;
    *dst4_2 = data;
    *dst4_3 = data;
}
