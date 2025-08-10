import torch
import triton

import triton_kernels
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

def quantize_bf16_mxfp4(w, block_size=None):
    w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=-1)
    return w, w_scale