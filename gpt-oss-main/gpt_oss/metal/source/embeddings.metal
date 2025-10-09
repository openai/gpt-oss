#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


kernel void gptoss_bf16_f32_embeddings(
    constant gptoss_embeddings_args& args [[ buffer(0) ]],
    const device uint* tokens [[ buffer(1) ]],
    const device bfloat4* weights [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    const device gptoss_control* control [[ buffer(4) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    if (control->abort != 0) {
        return;
    }

    const uint t = tokens[gid];

    weights += t * args.num_vecs;
    output += gid * args.num_vecs;
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        const bfloat4 w = weights[i];
        output[i] = static_cast<float4>(w);
    }
}
