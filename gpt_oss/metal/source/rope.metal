#include <metal_common>
#include <metal_math>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


// Each thread handles 2 head elements.
// Each simdgroup handles one head (64 head elements).

kernel void gptoss_f32_rope(
    constant gptoss_rope_args& args [[ buffer(0) ]],
    device float2* activations [[ buffer(1) ]],
    device float2* kv [[ buffer(2) ]],
    const device gptoss_control* control [[ buffer(3) ]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint num_head_dims = 64;
    if (control->abort != 0) {
        return;
    }

    const float dim_idx = static_cast<float>(gid.x % (num_head_dims / 2));
    const uint token_idx = args.token_offset + gid.y;
    activations += gid.y * args.token_stride + gid.x;

    const float2 input_vals = *activations;
    const float inv_extrapolation_freq = metal::precise::exp(dim_idx * args.freq_scale);
    const float inv_interpolation_freq = inv_extrapolation_freq * args.interpolation_scale;
    const float alpha = metal::saturate(metal::fma(dim_idx, args.yarn_scale, args.yarn_offset));
    const float inv_freq = metal::mix(inv_extrapolation_freq, inv_interpolation_freq, alpha);

    const float phi = static_cast<float>(token_idx) * inv_freq;
    const float yarn_multiplier = args.yarn_multiplier;
    float cosphi;
    const float sinphi = metal::precise::sincos(phi, cosphi) * yarn_multiplier;
    cosphi *= yarn_multiplier;

    const float output_re = input_vals.x * cosphi - input_vals.y * sinphi;
    const float output_im = input_vals.x * sinphi + input_vals.y * cosphi;
    *activations = (float2) { output_re, output_im };

    const uint head_dim = 64;
    const uint num_q_heads = 64;
    const uint num_kv_heads = 8;
    const uint head_idx = gid.x / (head_dim / 2);
    float2 vals = (float2) { output_re, output_im };
    if ((head_idx < num_q_heads)) {
        *activations = vals;
    } else if (head_idx < num_q_heads + num_kv_heads) {
        // Write k and v directly to the kv cache.
        const uint kv_head_idx = head_idx - num_q_heads;
        const uint dim_pair_idx = gid.x % (head_dim / 2);
        kv[(kv_head_idx * args.max_tokens + token_idx) * head_dim + dim_pair_idx] = vals;
    }
}