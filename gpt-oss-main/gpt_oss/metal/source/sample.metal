#include <metal_compute>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


inline static uint rng_squares32(ulong offset, ulong seed) {
    const ulong y = offset * seed;
    const ulong z = y + seed;

    /* Round 1 */
    ulong x = y * y + y;
    x = metal::rotate(x, 32ul);

    /* Round 2 */
    x = x * x + z;
    x = metal::rotate(x, 32ul);

    /* Round 3 */
    x = x * x + y;
    x = metal::rotate(x, 32ul);

    /* Round 4 */
    x = x * x + z;
    return as_type<uint2>(x).y;
}

kernel void gptoss_f32_softmax(
    constant gptoss_softmax_args& args [[ buffer(0) ]],
    const device float* score [[ buffer(1) ]],
    const device uint2* argmax [[ buffer(2) ]],
    device float* prob [[ buffer(3) ]],
    device float* sum [[ buffer(4) ]],
    const device gptoss_control* control [[ buffer(5) ]],
    uint tidx [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 threadgroup_size [[threads_per_threadgroup]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    threadgroup float threadgroup_sumexp[32];
    if (control->abort != 0) {
        return;
    }

    score += gid.y * args.num_vecs + gid.x * args.num_vecs_per_threadgroup;
    prob += gid.y * args.num_vecs + gid.x * args.num_vecs_per_threadgroup;
    sum += gid.y * args.max_threadgroups;

    uint max_bits = argmax[gid.y].y;
    if (static_cast<int>(max_bits) >= 0) {
        max_bits ^= 0x7FFFFFFFu;
    }
    const float max_val = as_type<float>(max_bits);
    float sum_exp = 0.0f;
    const uint num_vecs_per_threadgroup = metal::min(args.num_vecs - gid.x * args.num_vecs_per_threadgroup, args.num_vecs_per_threadgroup);
    for (uint i = tidx; i < num_vecs_per_threadgroup; i += threadgroup_size.x) {
        const float score_val = score[i];
        const float prob_val = metal::precise::exp((score_val - max_val) * args.temperature);
        prob[i] = prob_val;
        sum_exp += prob_val;
    }
    sum_exp = metal::simd_sum(sum_exp);
    if (metal::simd_is_first()) {
        threadgroup_sumexp[simdgroup_idx] = sum_exp;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simdgroup_idx == 0) {
        // Sum-Reduce threadgroup_sumexp
        sum_exp = 0.0f;
        if (simdgroup_tid < num_simdgroups) {
            sum_exp = threadgroup_sumexp[simdgroup_tid];
        }
        sum_exp = metal::simd_sum(sum_exp);
        if (metal::simd_is_first()) {
            sum[gid.x] = sum_exp;
        }
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void gptoss_f32_sample(
    constant gptoss_sample_args& args [[ buffer(0) ]],
    device const float* prob [[ buffer(1) ]],
    device const float* sum [[ buffer(2) ]],
    device uint* prediction [[ buffer(3) ]],
    device gptoss_control* control [[ buffer(4) ]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    threadgroup float threadgroup_sum_buffer[32];
    threadgroup uint threadgroup_idx_buffer[32];
    threadgroup float threadgroup_cumsum_buffer[32];
    if (control->abort != 0) {
        return;
    }

    const uint sample_word = rng_squares32(args.rng_offset, args.rng_seed);
    float sample_cdf = static_cast<float>(sample_word & 0x00FFFFFFu) * 0x1.0p-24f;

    float cumsum = 0.0f;
    if (tid < args.num_blocks) {
        cumsum = sum[tid];
    }
    cumsum = metal::simd_prefix_inclusive_sum(cumsum);
    if (simdgroup_tid == 31) {
        threadgroup_sum_buffer[simdgroup_idx] = cumsum;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float threadgroup_cumsum = 0.0f, threadgroup_sum = 0.0f;
    if (simdgroup_tid < num_simdgroups) {
        threadgroup_sum = threadgroup_sum_buffer[simdgroup_tid];
        if (simdgroup_tid < simdgroup_idx) {
            threadgroup_cumsum = threadgroup_sum;
        }
    }
    threadgroup_sum = metal::simd_sum(threadgroup_sum);
    cumsum += metal::simd_sum(threadgroup_cumsum);

    sample_cdf *= threadgroup_sum;
    sample_cdf = metal::max(sample_cdf, 0x1.0p-149f);

    // Find the block: the smallest tid where sample_cdf >= s
    uint block_idx = args.num_blocks;
    float block_sum = cumsum;
    if (tid >= args.num_blocks - 1) {
        block_idx = args.num_blocks - 1;
        block_sum = 0.0f;
    } else if (cumsum >= sample_cdf) {
        block_idx = tid;
        block_sum = 0.0f;
    }
    block_idx = metal::simd_min(block_idx);
    block_sum = metal::simd_max(block_sum);
    if (simdgroup_tid == 0) {
        threadgroup_idx_buffer[simdgroup_idx] = block_idx;
        threadgroup_cumsum_buffer[simdgroup_idx] = block_sum;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simdgroup_tid < num_simdgroups) {
        block_idx = threadgroup_idx_buffer[simdgroup_tid];
        block_sum = threadgroup_cumsum_buffer[simdgroup_tid];
    }
    block_idx = metal::simd_min(block_idx);
    block_sum = metal::simd_max(block_sum);

    const uint block_start = args.num_dims_per_block * block_idx;
    const uint block_end = metal::min(block_start + args.num_dims_per_block, args.num_dims);
    uint offset = block_start + tid;
    float accumulated_sum = block_sum;
    uint sample_idx;

    // This loop must be threadgroup-uniform.
    do {
        // Find the token: the smallest tid where sample_cdf >= s
        float cumsum = 0.0f;
        if (offset < block_end) {
            cumsum = prob[offset];
        }
        cumsum = metal::simd_prefix_inclusive_sum(cumsum);
        if (simdgroup_tid == 31) {
            threadgroup_sum_buffer[simdgroup_idx] = cumsum;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        float threadgroup_cumsum = 0.0f, threadgroup_sum = 0.0f;
        if (simdgroup_tid < num_simdgroups) {
            threadgroup_sum = threadgroup_sum_buffer[simdgroup_tid];
            if (simdgroup_tid < simdgroup_idx) {
                threadgroup_cumsum = threadgroup_sum;
            }
        }
        threadgroup_sum = metal::simd_sum(threadgroup_sum);
        cumsum += metal::simd_sum(threadgroup_cumsum);
        cumsum += accumulated_sum;

        sample_idx = block_end;
        if (offset >= block_end) {
            // Trigger loop exit, with the last token in the block being sampled if no other candidate was found.
            sample_idx = block_end - 1;
        } else if (cumsum >= sample_cdf) {
            sample_idx = offset;
        }
        sample_idx = metal::simd_min(sample_idx);
        if (simdgroup_tid == 0) {
            threadgroup_idx_buffer[simdgroup_idx] = sample_idx;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (simdgroup_tid < num_simdgroups) {
            sample_idx = threadgroup_idx_buffer[simdgroup_tid];
        }
        sample_idx = metal::simd_min(sample_idx);

        offset += threadgroup_size;
        accumulated_sum += threadgroup_sum;
    } while (sample_idx == block_end);

    if (tid == 0) {
        *prediction = sample_idx;
    }
}
