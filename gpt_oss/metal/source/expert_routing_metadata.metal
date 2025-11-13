#include <internal/kernel-args.h>
#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>

constant uint kMaxExperts = 128;

kernel void gptoss_f32_expert_routing_metadata(
    constant gptoss_expert_routing_metadata_args& args [[ buffer(0) ]],
    const device gptoss_expert_prediction* __restrict__ expert_predictions [[ buffer(1) ]],
    device uint* __restrict__ expert_offsets [[ buffer(2) ]],
    device uint* __restrict__ intra_expert_offsets [[ buffer(3) ]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid [[thread_position_in_threadgroup]]) 
{
    assert(args.num_experts <= kMaxExperts);
    // Create threadgroup mem and initialize it to 0.
    threadgroup metal::atomic_uint tg_counts[kMaxExperts];
    for (uint e = tid; e < args.num_experts; e += tg_size) {
        metal::atomic_store_explicit(&tg_counts[e], 0u, metal::memory_order_relaxed);
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint i = tid; i < args.tokens; i += tg_size) {
        const uint e = expert_predictions[i].expert_id;
        const uint r = metal::atomic_fetch_add_explicit(&tg_counts[e], 1u, metal::memory_order_relaxed);
        intra_expert_offsets[i] = r;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint total = 0;
        for (uint e = 0; e < args.num_experts; ++e) {
            const uint bin = metal::atomic_load_explicit(&tg_counts[e], metal::memory_order_relaxed);
            expert_offsets[e] = total;
            total += bin;
        }
        expert_offsets[args.num_experts] = total;
    }
}