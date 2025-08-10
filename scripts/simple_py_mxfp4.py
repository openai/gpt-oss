import torch


# the functions are adapted from https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0bea1c31d75761002aad4290e572cf7c512d8b3a/modelopt/torch/quantization/qtensor/mxfp4_tensor.py#L25

E2M1_max = 6.0

E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

# TODO (yiakwy) : create from E2M1_values
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

def pack_uint4x2_to_uint8(x):
    # If the last dimension is odd, pad with zeros
    # If this behavior is not desired, please modify the code accordingly
    left_side = x[..., 0::2]  # Even indices (0, 2, 4...)
    right_side = x[..., 1::2]  # Odd indices (1, 3, 5...)
    new_data = right_side.clone() << 4  # Put odd indices (higher addresses) in high bits
    new_data[..., : left_side.shape[-1]] += left_side  # Put even indices in low bits
    return new_data

def cast_fp4(x):
    sign = torch.sign(x)
    sign_bit = (2 - sign) // 2
    ord_ = torch.sum(
        (x.abs().unsqueeze(-1) - E2M1_bounds.to(x.device)) > 0, dim=-1
    )
    fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
    return fp4_val

# convert bf16 tensor to uint8
def quantize_bf16_mxfp4(input : torch.Tensor, block_size : int | None):
    block_size = block_size or 32

    input = input.view(-1, block_size)
    
    input_amax = input.abs().max(dim=-1, keepdim=True).values
    descale = input_amax / E2M1_max

    min_value = torch.tensor(-127.0, device=descale.device)
    e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

    original_shape = input.shape
    input = (input / torch.exp2(e8m0_scale)).view(original_shape)
    input_q = cast_fp4(input)
    input_q = pack_uint4x2_to_uint8(input_q)

    e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
    return input_q, e8m0_scale


# the function is adapted from GPT_OSS repo
def convert_fp4_bf16(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    import math

    # Check if blocks and scales are on CPU, and move to GPU if so
    if not blocks.is_cuda and torch.cuda.is_available():
        blocks = blocks.cuda()
        scales = scales.cuda()

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    # TODO: Delete after making sure this is not necessary! since we go back to cpu in the end in create_quantized_param using .to(target_device)
    # Move back to CPU if needed
    # if need_to_move_back:
    #     out = out.cpu()
    del blocks, scales, lut
    return out