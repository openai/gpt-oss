import math
import os
import functools
import collections
import threading
import torch
from safetensors import safe_open

try:
    profile # type: ignore
except NameError:
    profile = lambda f: f


# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

# Map the names assumed in this implementation to the checkpoint names.
PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales") for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales") for n in range(36)
}


class TensorLRUCache:
    """
    A thread-safe, memory-aware LRU cache decorator with an individual tensor size limit.

    Args:
        max_memory_gb (float): The maximum total memory in GB to be used by the cache.
        max_individual_size_gb (float, optional): The maximum size in GB for any single
            tensor to be considered for caching. If a tensor is larger than this, it will
            be bypassed and not cached. Defaults to None (no individual limit).
    """

    def __init__(self, max_memory_gb: float, max_individual_size_gb: float | None = None, min_individual_size_gb: float | None = None):
        self.max_memory_bytes = int(max_memory_gb * (1024 ** 3))
        if max_individual_size_gb is not None:
            self.max_individual_size_bytes = int(max_individual_size_gb * (1024 ** 3))
        else:
            self.max_individual_size_bytes = None
        if min_individual_size_gb is not None:
            self.min_individual_size_bytes = int(min_individual_size_gb * (1024 ** 3))
        else:
            self.min_individual_size_bytes = None

        self.cache = collections.OrderedDict()
        self.current_size_bytes = 0
        self.lock = threading.Lock()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            try:
                key = (args, frozenset(kwargs.items()))
            except TypeError:
                return func(*args, **kwargs)

            # First, check if the item is already in the cache (fast path)
            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]

            # Item not in cache, so we must call the expensive function
            result = func(*args, **kwargs)

            if not isinstance(result, torch.Tensor):
                return result

            tensor_size = result.nbytes

            # --- Bypass cache for oversized/small tensors ---
            if self.max_individual_size_bytes is not None and tensor_size > self.max_individual_size_bytes or self.min_individual_size_bytes > tensor_size:
                #print(tensor_size)
                #print(*args)
                return result

            # If the tensor itself is larger than the *total* cache capacity, don't cache
            if tensor_size > self.max_memory_bytes:
                return result

            # Evict items until there's space for the new tensor
            with self.lock:
                while self.current_size_bytes + tensor_size > self.max_memory_bytes:
                    evicted_key, evicted_tensor = self.cache.popitem(last=False)
                    self.current_size_bytes -= evicted_tensor.nbytes

                # Add the new item
                self.cache[key] = result
                self.current_size_bytes += tensor_size

            return result

        return wrapped_func


class Checkpoint:
    def __init__(self, path: str, device: torch.device, pin_memory_for_faster_cpu_gpu_transfer: bool = False):
        device_str = (
            device.type
            if device.index is None
            else device.type + ":" + str(device.index)
        )
        self.device_str = device_str
        self.pin_memory = pin_memory_for_faster_cpu_gpu_transfer # use gpu shared memory 2.5Gb for gpt-oss-20b 5% faster
        self.lut_buffer = False # use additional gpu shared memory 1Gb for gpt-oss-20b, 1-2% faster
        self.file_handles = {}

        # Read from all files ending with .safetensors in the checkpoint directory
        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        # Build a mapping from tensor name to (file, key)
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            handle = safe_open(safetensor_file, framework="pt", device='cpu')
            self.file_handles[safetensor_file] = handle
            for key in handle.keys():
                tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file
        if self.lut_buffer:
            self._lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16, device=device)

    #@TensorLRUCache(max_memory_gb=3.6, max_individual_size_gb=0.2, min_individual_size_gb=0.03) # (3.6, 0.2, 0.03) cache for mlp1 use additional memory but 5% faster, use it only for gpt_oss.generate (Max vram used: 7.6Gb)
    def _get_tensor(self, name: str) -> torch.Tensor:
        assert name in self.tensor_name_to_file, f"Tensor {name} not found."
        file_key = self.tensor_name_to_file[name]
        handle = self.file_handles[file_key]

        # Get tensor and pin memory for faster GPU transfers
        tensor = handle.get_tensor(name)
        if self.pin_memory and self.device_str.startswith('cuda'):
            tensor = tensor.pin_memory()

        return tensor.to(self.device_str, non_blocking=True)

    def __del__(self):
        for handle in self.file_handles.values():
            pass

    def get(self, name: str) -> torch.Tensor:
        match PARAM_NAME_MAP.get(name, name):
            case (blocks_name, scales_name):
                # MoE weights: are in block-based MXFP4 format
                return self._get_mxfp4_tensor(blocks_name, scales_name, dtype=torch.bfloat16)
            case tensor_name:
                # MoE biases and other weights
                return self._get_tensor(tensor_name)

    def _get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 16384 * 64,
    ) -> torch.Tensor:
        assert blocks_name in self.tensor_name_to_file, (
            f"Blocks tensor {blocks_name} not found in checkpoint."
        )
        assert scales_name in self.tensor_name_to_file, (
            f"Scales tensor {scales_name} not found in checkpoint."
        )

        blocks = self._get_tensor(blocks_name)
        scales = self._get_tensor(scales_name).to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )

        if self.lut_buffer:
            lut = self._lut
        else:
            lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total   = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)
            sub = out[r0:r1]
            sub[:, 0::2] = lut[torch.bitwise_and(blocks[r0:r1], 0x0F).long()]
            sub[:, 1::2] = lut[torch.bitwise_right_shift(blocks[r0:r1], 4).long()]
            torch.ldexp_(sub, scales[r0:r1])

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    def _get_mxfp4_tensor_copy(self, blocks_name: str, scales_name: str, dtype: torch.dtype = torch.bfloat16):
        "short version that uses a lot of memory"

        loaded_blocks = self._get_tensor(blocks_name)
        # Split it into low and high nibbles, upcast to bytes, and interleave (for swiglu)
        loaded_blocks_lo = loaded_blocks & 0x0F
        loaded_blocks_hi = loaded_blocks >> 4
        loaded_blocks = torch.stack((loaded_blocks_lo, loaded_blocks_hi), dim=-1)
        loaded_blocks = loaded_blocks.view(*loaded_blocks.shape[:-2], loaded_blocks.shape[-2] * 2)

        loaded_scales = self._get_tensor(scales_name)
        # Upcast to int32 and subtract bias
        loaded_scales = loaded_scales.int() - 127

        # Convert MXFP4 numbers into target dtype
        fp4_values = torch.tensor(FP4_VALUES, dtype=dtype, device=self.device_str)
        loaded_tensor = torch.ldexp(fp4_values[loaded_blocks.int()], loaded_scales.unsqueeze(-1))
        loaded_tensor = loaded_tensor.view(*loaded_tensor.shape[:-2], -1)
        return loaded_tensor
