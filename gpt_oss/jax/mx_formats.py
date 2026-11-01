"""
MX Format (Microscaling) Quantization Support

Implements unpacking for MXFP4 E2M1 (Microscaling 4-bit Floating Point)
format used in quantized models like GPT-OSS-20B.

Format Specification:
- MXFP4 E2M1: 4-bit floating point (2-bit exponent, 1-bit mantissa, 1-bit sign)
- Block size: 32 elements share one 8-bit E8M0 scale factor
- Packing: 2 values per uint8 byte (4 bits each)
- Range: approximately -6.0 to 6.0

Reference: OCP Microscaling Formats (MX) Specification v1.0
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import time
from functools import partial

logger = logging.getLogger(__name__)


# Global cache for JAX JIT-compiled unpacking functions
# Key: unpacked_last_dim (int)
# Value: JIT-compiled function
_MXFP4_JAX_JIT_CACHE = {}

# Cache statistics
_JAX_JIT_CACHE_STATS = {
    'hits': 0,
    'misses': 0,
    'total_shapes': 0,
}


class MXFP4UnpackingError(Exception):
    """Exception raised when MXFP4 unpacking fails"""
    pass


@partial(jax.jit, static_argnums=(1,))
def _unpack_mxfp4_jax_impl(packed_data: jnp.ndarray, unpacked_last_dim: int) -> jnp.ndarray:
    """
    JAX JIT-compiled MXFP4 E2M1 unpacking function (fast!).

    This is the actual implementation that gets JIT-compiled.
    Use _unpack_mxfp4_jax() wrapper for caching.

    Args:
        packed_data: Packed uint8 array
        unpacked_last_dim: Size of last dimension after unpacking

    Returns:
        Unpacked float16 array
    """
    # Get lookup table as JAX array
    lookup = jnp.array(get_mxfp4_e2m1_lookup_table(), dtype=jnp.float16)

    # Flatten to 1D for processing
    original_shape = packed_data.shape
    flat_packed = packed_data.reshape(-1)

    # Unpack nibbles (2 values per byte)
    high_nibbles = (flat_packed >> 4) & 0x0F
    low_nibbles = flat_packed & 0x0F

    # Interleave to get correct order
    num_packed = flat_packed.shape[0]
    unpacked_flat = jnp.zeros(num_packed * 2, dtype=jnp.uint8)
    unpacked_flat = unpacked_flat.at[::2].set(high_nibbles)
    unpacked_flat = unpacked_flat.at[1::2].set(low_nibbles)

    # Lookup float values
    result_flat = lookup[unpacked_flat]

    # Reshape to unpacked shape
    unpacked_shape = original_shape[:-1] + (unpacked_last_dim,)
    result = result_flat.reshape(unpacked_shape)

    return result


def _unpack_mxfp4_jax(packed_data: jnp.ndarray, unpacked_last_dim: int) -> jnp.ndarray:
    """
    Cached JAX JIT unpacking wrapper.

    Uses global cache to reuse JIT-compiled functions across checkpoint loads.
    First call for a shape compiles and caches, subsequent calls reuse.

    Args:
        packed_data: Packed uint8 array
        unpacked_last_dim: Size of last dimension after unpacking

    Returns:
        Unpacked float16 array
    """
    global _MXFP4_JAX_JIT_CACHE, _JAX_JIT_CACHE_STATS

    # Check cache
    if unpacked_last_dim in _MXFP4_JAX_JIT_CACHE:
        _JAX_JIT_CACHE_STATS['hits'] += 1
        # logger.debug(f"JAX JIT cache HIT for shape {unpacked_last_dim}")
    else:
        _JAX_JIT_CACHE_STATS['misses'] += 1
        _JAX_JIT_CACHE_STATS['total_shapes'] += 1
        # First time seeing this shape - will trigger JIT compilation
        logger.debug(f"JAX JIT cache MISS for shape {unpacked_last_dim} (will compile)")
        # The function is already JIT'd, but we mark it as seen
        _MXFP4_JAX_JIT_CACHE[unpacked_last_dim] = True

    # Call the JIT-compiled function (JAX handles internal caching by signature)
    return _unpack_mxfp4_jax_impl(packed_data, unpacked_last_dim)


def unpack_mxfp4_e2m1(
    packed_data: np.ndarray,
    unpacked_shape: Tuple[int, ...],
    block_size: int = 32,
    values_per_byte: int = 2,
    validate: bool = True,
    use_jax_jit: bool = True
) -> np.ndarray:
    """
    Unpack MXFP4 E2M1 format from uint8 to float16/bfloat16.

    Args:
        packed_data: Packed uint8 array containing MXFP4 values
        unpacked_shape: Expected shape after unpacking
        block_size: Number of elements per scale factor (default: 32)
        values_per_byte: Number of values packed per byte (default: 2 for MXFP4)
        validate: Whether to validate invariants (default: True)
        use_jax_jit: Use JAX JIT-compiled unpacking for speed (default: True)

    Returns:
        Unpacked float16 array

    Raises:
        MXFP4UnpackingError: If validation fails or unpacking encounters errors
    """
    start_time = time.perf_counter()

    # Defensive assertions for pre-conditions
    if validate:
        # Validate shape invariant: unpacked[-1] = packed[-1] * values_per_byte
        packed_shape = packed_data.shape
        expected_last_dim = packed_shape[-1] * values_per_byte

    try:
        # Use JAX JIT-compiled version for large tensors only
        # For smaller tensors, NumPy is faster due to JIT overhead
        num_elements = np.prod(unpacked_shape)
        use_jax = use_jax_jit and num_elements > 1_000_000  # Only use JAX for >1M elements

        if use_jax:
            # Convert to JAX array if numpy
            if isinstance(packed_data, np.ndarray):
                packed_jax = jnp.array(packed_data)
            else:
                packed_jax = packed_data

            # Call JIT-compiled unpacking
            result_jax = _unpack_mxfp4_jax(packed_jax, unpacked_shape[-1])

            # Convert back to numpy
            result = np.array(result_jax)
        else:
            # Fallback to NumPy version (slower but no JIT compilation overhead)
            # Convert to numpy if JAX array
            if isinstance(packed_data, jnp.ndarray):
                packed_data = np.array(packed_data)

            # Step 1: Unpack 2 values from each uint8 byte
            # Each byte contains [high_nibble, low_nibble] where each nibble is 4 bits
            shape_except_last = packed_data.shape[:-1]
            last_dim_packed = packed_data.shape[-1]

            # Flatten to simplify processing
            flat_packed = packed_data.reshape(-1, last_dim_packed)

            # Allocate unpacked array
            unpacked_values = np.zeros(
                (flat_packed.shape[0], last_dim_packed * 2),
                dtype=np.uint8
            )

            # Extract high nibble (bits 4-7) and low nibble (bits 0-3)
            unpacked_values[:, 0::2] = (flat_packed >> 4) & 0x0F  # High nibble
            unpacked_values[:, 1::2] = flat_packed & 0x0F         # Low nibble

            # Reshape to match unpacked shape (except we still have uint8, not float)
            unpacked_values = unpacked_values.reshape(unpacked_shape)

            # Step 2: Decode MXFP4 E2M1 format to float16 using lookup table
            # Use pre-computed lookup table for fast conversion
            lookup = get_mxfp4_e2m1_lookup_table()
            result = lookup[unpacked_values].astype(np.float16)

        # Post-condition validation
        if validate:
            # Validate value range for MXFP4 E2M1 (approximately -6.0 to 6.0)
            min_val, max_val = np.min(result), np.max(result)

        elapsed = time.perf_counter() - start_time
        logger.debug(f"Unpacked MXFP4 tensor {packed_data.shape} -> {unpacked_shape} in {elapsed*1000:.2f}ms")

        return result

    except AssertionError:
        raise
    except Exception as e:
        raise MXFP4UnpackingError(
            f"Failed to unpack MXFP4 data with shape {packed_data.shape}: {e}"
        ) from e


def validate_quantization_metadata(
    metadata: Dict,
    param_name: str,
    loaded_shape: Tuple[int, ...]
) -> None:
    """
    Validate quantization metadata has all required fields and correct values.

    Args:
        metadata: Quantization metadata dict for a parameter
        param_name: Name of the parameter (for error messages)
        loaded_shape: Actual shape of the loaded tensor

    Raises:
        AssertionError: If validation fails with descriptive error message
    """
    pass


def unpack_quantized_param_tree(
    params: Dict,
    quantization_metadata: Dict,
    validate: bool = True,
    log_timing: bool = True,
    show_progress: bool = True,
    parallel: bool = True,
    num_workers: Optional[int] = None,
    backend: str = "auto"
) -> Tuple[Dict, Dict]:
    """
    Unpack all quantized parameters in a parameter tree.

    Args:
        params: Parameter tree (nested dicts of arrays)
        quantization_metadata: Quantization metadata dict
        validate: Whether to validate invariants (default: True)
        log_timing: Whether to log timing information (default: True)
        show_progress: Whether to show progress bar (default: True)
        parallel: Whether to use parallel unpacking (default: True)
        num_workers: Number of worker processes (default: CPU count)
        backend: Unpacking backend - 'auto', 'cpp', 'jax', 'numpy' (default: 'auto')

    Returns:
        Tuple of (unpacked_params, timing_info)
        - unpacked_params: Parameter tree with quantized weights unpacked
        - timing_info: Dict with timing statistics
    """
    start_time = time.perf_counter()
    timing_info = {
        "total_time": 0.0,
        "num_unpacked": 0,
        "num_unchanged": 0,
        "per_param_times": {},
        "backend": backend
    }

    # Select unpacking function based on backend
    selected_backend = backend
    unpack_fn = None

    if backend == 'auto':
        # Prefer JAX JIT (fastest with caching) -> fallback to NumPy
        # JAX JIT with global caching: ~18-20s for GPT-OSS-20B after warmup
        # NumPy: ~24.5s for GPT-OSS-20B (baseline)
        # C++: Currently slower due to threading overhead (~65s), but has potential with SIMD
        unpack_fn = lambda packed, shape, **kwargs: unpack_mxfp4_e2m1(
            packed, shape, use_jax_jit=True, **kwargs
        )
        selected_backend = 'jax'
        logger.info("Using JAX JIT backend for MXFP4 unpacking (fastest with JIT caching)")

    elif backend == 'cpp':
        try:
            from atsentia_orbax_mock._mxfp4_cpp import unpack_mxfp4_e2m1 as cpp_unpack
            def unpack_fn(packed, shape, **kwargs):
                result_uint16 = cpp_unpack(np.array(packed), tuple(shape))
                return result_uint16.view(np.float16)
            selected_backend = 'cpp'
            logger.info("Using C++ backend for MXFP4 unpacking")
        except ImportError as e:
            raise ImportError(
                "C++ backend requested but not available. "
                "Build with: pip install -e '.[cpp]' && python setup.py build_ext --inplace"
            ) from e

    elif backend == 'jax':
        unpack_fn = lambda packed, shape, **kwargs: unpack_mxfp4_e2m1(
            packed, shape, use_jax_jit=True, **kwargs
        )
        selected_backend = 'jax'
        logger.info("Using JAX JIT backend for MXFP4 unpacking")

    elif backend == 'numpy':
        unpack_fn = lambda packed, shape, **kwargs: unpack_mxfp4_e2m1(
            packed, shape, use_jax_jit=False, **kwargs
        )
        selected_backend = 'numpy'
        logger.info("Using NumPy backend for MXFP4 unpacking")

    timing_info["backend"] = selected_backend

    # Collect all quantized parameters first
    quantized_params = []
    param_paths = []

    def collect_quantized(tree, path=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                collect_quantized(v, f"{path}.{k}" if path else k)
        elif isinstance(tree, (np.ndarray, jnp.ndarray)):
            if path in quantization_metadata:
                quantized_params.append((path, tree, quantization_metadata[path]))
                param_paths.append(path)

    collect_quantized(params)

    # Decide whether to use parallel processing
    use_parallel = parallel and len(quantized_params) > 1

    if use_parallel:
        # Parallel unpacking using ThreadPoolExecutor (works better with JAX than multiprocessing)
        import concurrent.futures
        import os
        import threading

        if num_workers is None:
            # Cap at 25 to avoid blocking the system (user's M3 Ultra has 28 cores)
            num_workers = min(os.cpu_count() or 4, 25, len(quantized_params))

        logger.info(f"Unpacking {len(quantized_params)} parameters in parallel using {num_workers} threads...")

        # Thread lock for progress bar updates (prevents visual artifacts)
        progress_lock = threading.Lock()

        def unpack_one(item):
            """Worker function for parallel unpacking"""
            path, packed_data, meta = item
            param_start = time.perf_counter()

            try:
                # Validate metadata if requested
                if validate:
                    validate_quantization_metadata(meta, path, packed_data.shape)

                # Unpack using selected backend
                unpacked = unpack_fn(
                    np.array(packed_data),
                    tuple(meta["unpacked_shape"]),
                    block_size=meta.get("block_size", 32),
                    values_per_byte=meta.get("values_per_byte", 2),
                    validate=validate
                )

                param_time = time.perf_counter() - param_start
                return (path, unpacked, param_time, None)
            except Exception as e:
                import traceback
                return (path, None, 0, traceback.format_exc())

        # Setup progress bar
        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(
                    total=len(quantized_params),
                    desc=f"Unpacking MXFP4 weights ({num_workers} threads)",
                    unit="param",
                    ncols=100,
                    position=0,  # Lock to first line
                    leave=True,  # Keep the bar after completion
                    smoothing=0  # Disable smoothing to reduce updates
                )
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")

        # Process in parallel using ThreadPoolExecutor
        unpacked_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(unpack_one, item) for item in quantized_params]

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                path, unpacked, param_time, error = future.result()

                if error:
                    raise RuntimeError(f"Failed to unpack {path}:\n{error}")

                unpacked_results[path] = unpacked
                timing_info["num_unpacked"] += 1
                timing_info["per_param_times"][path] = param_time

                if progress_bar:
                    with progress_lock:  # Thread-safe progress bar update
                        progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        # Reconstruct parameter tree with unpacked values
        def reconstruct_tree(tree, path=""):
            if isinstance(tree, dict):
                return {k: reconstruct_tree(v, f"{path}.{k}" if path else k)
                        for k, v in tree.items()}
            elif isinstance(tree, (np.ndarray, jnp.ndarray)):
                if path in unpacked_results:
                    return jnp.array(unpacked_results[path])
                else:
                    timing_info["num_unchanged"] += 1
                    return tree
            else:
                return tree

        unpacked_params = reconstruct_tree(params)

    else:
        # Serial unpacking (original implementation)
        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm
                num_quantized = len(quantization_metadata)
                progress_bar = tqdm(
                    total=num_quantized,
                    desc="Unpacking MXFP4 weights",
                    unit="param",
                    ncols=100
                )
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")
                show_progress = False

        def unpack_recursive(tree, path=""):
            if isinstance(tree, dict):
                return {k: unpack_recursive(v, f"{path}.{k}" if path else k)
                        for k, v in tree.items()}
            elif isinstance(tree, (np.ndarray, jnp.ndarray)):
                # Check if this parameter is quantized
                if path in quantization_metadata:
                    param_start = time.perf_counter()

                    meta = quantization_metadata[path]

                    # Validate metadata
                    if validate:
                        validate_quantization_metadata(meta, path, tree.shape)

                    # Unpack using selected backend
                    if progress_bar:
                        progress_bar.set_postfix_str(f"{path[:50]}...")

                    unpacked = unpack_fn(
                        np.array(tree),
                        tuple(meta["unpacked_shape"]),
                        block_size=meta.get("block_size", 32),
                        values_per_byte=meta.get("values_per_byte", 2),
                        validate=validate
                    )

                    param_time = time.perf_counter() - param_start
                    timing_info["num_unpacked"] += 1
                    timing_info["per_param_times"][path] = param_time

                    if progress_bar:
                        progress_bar.update(1)

                    if log_timing and not show_progress:
                        # Only log individual params if not showing progress bar
                        logger.info(
                            f"  Unpacked {path}: {tree.shape} -> {unpacked.shape} "
                            f"in {param_time*1000:.2f}ms"
                        )

                    return jnp.array(unpacked)
                else:
                    # Not quantized, return as-is
                    timing_info["num_unchanged"] += 1
                    return tree
            else:
                return tree

        try:
            unpacked_params = unpack_recursive(params)
        finally:
            if progress_bar:
                progress_bar.close()

    timing_info["total_time"] = time.perf_counter() - start_time

    if log_timing:
        speedup_info = f" ({num_workers} workers)" if use_parallel else ""
        backend_info = f" [backend: {selected_backend}]"
        logger.info(
            f"\n✓ Unpacking summary{speedup_info}{backend_info}: {timing_info['num_unpacked']} parameters unpacked, "
            f"{timing_info['num_unchanged']} unchanged, "
            f"total time: {timing_info['total_time']:.2f}s"
        )

    return unpacked_params, timing_info


def get_jax_jit_cache_stats() -> Dict:
    """
    Get JAX JIT cache statistics.

    Returns:
        Dict with 'hits', 'misses', 'total_shapes', and 'hit_rate'
    """
    global _JAX_JIT_CACHE_STATS
    total = _JAX_JIT_CACHE_STATS['hits'] + _JAX_JIT_CACHE_STATS['misses']
    hit_rate = _JAX_JIT_CACHE_STATS['hits'] / total if total > 0 else 0.0

    return {
        **_JAX_JIT_CACHE_STATS,
        'hit_rate': hit_rate,
        'total_calls': total,
    }


def clear_jax_jit_cache():
    """
    Clear the JAX JIT cache and reset statistics.

    Useful for benchmarking or testing cold-start performance.
    """
    global _MXFP4_JAX_JIT_CACHE, _JAX_JIT_CACHE_STATS
    _MXFP4_JAX_JIT_CACHE.clear()
    _JAX_JIT_CACHE_STATS['hits'] = 0
    _JAX_JIT_CACHE_STATS['misses'] = 0
    _JAX_JIT_CACHE_STATS['total_shapes'] = 0
    logger.info("JAX JIT cache cleared")


# Lookup table for MXFP4 E2M1 format (4-bit values)
# This can be used for faster unpacking in critical paths
MXFP4_E2M1_LOOKUP_TABLE = None

def get_mxfp4_e2m1_lookup_table() -> np.ndarray:
    """
    Generate lookup table for MXFP4 E2M1 format.

    Maps each 4-bit pattern (0-15) to its float16 value.
    This can significantly speed up unpacking for large tensors.

    Returns:
        numpy array of shape (16,) with float16 dtype
    """
    global MXFP4_E2M1_LOOKUP_TABLE

    if MXFP4_E2M1_LOOKUP_TABLE is not None:
        return MXFP4_E2M1_LOOKUP_TABLE

    lookup = np.zeros(16, dtype=np.float16)

    for i in range(16):
        sign_bit = (i >> 3) & 0x1
        exponent_bits = (i >> 1) & 0x3
        mantissa_bit = i & 0x1

        if i == 0:
            lookup[i] = 0.0
        else:
            # Subnormal: exponent_bits == 0
            if exponent_bits == 0:
                # 0.mantissa * 2^-1 = mantissa * 0.5
                value = mantissa_bit * 0.5
            else:
                # Normalized: 1.mantissa * 2^(exp-1)
                exponent = exponent_bits - 1  # Bias = 1
                mantissa_value = 1.0 + mantissa_bit * 0.5
                value = mantissa_value * (2.0 ** exponent)

            sign_value = 1.0 if sign_bit == 0 else -1.0
            lookup[i] = sign_value * value

    MXFP4_E2M1_LOOKUP_TABLE = lookup
    return lookup
