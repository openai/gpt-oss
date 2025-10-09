#include <metal_integer>
#include <metal_math>

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

kernel void gptoss_u32_fill_random(
    constant gptoss_u32_fill_random_args& args [[ buffer(0) ]],
    device uint* output [[ buffer(1) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);
    const ulong thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    output += thread_start;
    ulong offset = args.offset + thread_start;
    for (; num_iter != 0; num_iter--) {
        *output = rng_squares32(offset, args.seed);
        output += threadgroup_size;
        offset += threadgroup_size;
    }
}

kernel void gptoss_f32_fill_random(
    constant gptoss_f32_fill_random_args& args [[ buffer(0) ]],
    device float* output [[ buffer(1) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);
    const ulong thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    output += thread_start;
    ulong offset = args.offset + thread_start;
    for (; num_iter != 0; num_iter--) {
        const uint word = rng_squares32(offset, args.seed);
        *output = metal::fma(static_cast<float>(as_type<int>(word)), args.scale, args.bias);
        output += threadgroup_size;
        offset += threadgroup_size;
    }
}

kernel void gptoss_bf16_fill_random(
    constant gptoss_f32_fill_random_args& args [[ buffer(0) ]],
    device bfloat* output [[ buffer(1) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);
    const ulong thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    output += thread_start;
    ulong offset = args.offset + thread_start;
    for (; num_iter != 0; num_iter--) {
        const uint word = rng_squares32(offset, args.seed);
        *output = static_cast<bfloat>(metal::fma(static_cast<float>(as_type<int>(word)), args.scale, args.bias));
        output += threadgroup_size;
        offset += threadgroup_size;
    }
}
