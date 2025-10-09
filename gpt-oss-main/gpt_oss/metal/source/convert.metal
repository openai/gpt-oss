#include <metal_integer>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


kernel void gptoss_mf4_f32_convert(
    constant gptoss_convert_args& args [[ buffer(0) ]],
    const device uint4* blocks [[ buffer(1) ]],
    const device uchar* scales [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);
    const ulong thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    blocks += thread_start;
    scales += thread_start;
    output += 8 * thread_start;
    for (; num_iter != 0; num_iter--) {
        const uint4 block = *blocks;
        const float scale = as_type<float>((static_cast<uint>(*scales) + 14) << 23);
        uint4 block02468ACEGIKMOQSU = block + block;
        uint4 block13579BDFHJLNPRTV = block >> 3;
        block02468ACEGIKMOQSU &= 0x1E1E1E1Eu;
        block13579BDFHJLNPRTV &= 0x1E1E1E1Eu;
        block02468ACEGIKMOQSU += 0x70707070u;
        block13579BDFHJLNPRTV += 0x70707070u;
        block02468ACEGIKMOQSU &= 0x8E8E8E8Eu;
        block13579BDFHJLNPRTV &= 0x8E8E8E8Eu;
        const uint4 block26AEIMQU = block02468ACEGIKMOQSU & 0xFF00FF00u;
        const uint4 block048CGKOS = (block02468ACEGIKMOQSU << 8) & 0xFF00FF00u;
        const uint4 block37BFJNRV = block13579BDFHJLNPRTV & 0xFF00FF00u;
        const uint4 block159DHLPT = (block13579BDFHJLNPRTV << 8) & 0xFF00FF00u;
        const float4 block048C = static_cast<float4>(as_type<half4>(block048CGKOS.xy)) * scale;
        const float4 blockGKOS = static_cast<float4>(as_type<half4>(block048CGKOS.zw)) * scale;
        const float4 block26AE = static_cast<float4>(as_type<half4>(block26AEIMQU.xy)) * scale;
        const float4 blockIMQU = static_cast<float4>(as_type<half4>(block26AEIMQU.zw)) * scale;
        const float4 block159D = static_cast<float4>(as_type<half4>(block159DHLPT.xy)) * scale;
        const float4 blockHLPT = static_cast<float4>(as_type<half4>(block159DHLPT.zw)) * scale;
        const float4 block37BF = static_cast<float4>(as_type<half4>(block37BFJNRV.xy)) * scale;
        const float4 blockJNRV = static_cast<float4>(as_type<half4>(block37BFJNRV.zw)) * scale;

        output[0] = (float4) { block048C.x, block159D.x, block26AE.x, block37BF.x };
        output[1] = (float4) { block048C.y, block159D.y, block26AE.y, block37BF.y };
        output[2] = (float4) { block048C.z, block159D.z, block26AE.z, block37BF.z };
        output[3] = (float4) { block048C.w, block159D.w, block26AE.w, block37BF.w };
        output[4] = (float4) { blockGKOS.x, blockHLPT.x, blockIMQU.x, blockJNRV.x };
        output[5] = (float4) { blockGKOS.y, blockHLPT.y, blockIMQU.y, blockJNRV.y };
        output[6] = (float4) { blockGKOS.z, blockHLPT.z, blockIMQU.z, blockJNRV.z };
        output[7] = (float4) { blockGKOS.w, blockHLPT.w, blockIMQU.w, blockJNRV.w };

        blocks += threadgroup_size;
        scales += threadgroup_size;
        output += 8 * threadgroup_size;
    }
}
