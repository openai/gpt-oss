import os
import torch
import torch.distributed as dist


def suppress_output(rank: int) -> None:
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed() -> torch.device:
    """Initialize the model for distributed inference."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure CUDA is installed and accessible."
        )
    
    # Initialize distributed inference
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Validate rank against available devices
    if rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"Rank {rank} exceeds available CUDA devices ({torch.cuda.device_count()}). "
            f"Please set RANK to a value between 0 and {torch.cuda.device_count() - 1}."
        )
    
    try:
        if world_size > 1:
            dist.init_process_group(
                backend="nccl", init_method="env://", world_size=world_size, rank=rank
            )
        
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        # Test device accessibility
        try:
            torch.cuda.get_device_properties(device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to access CUDA device {rank}: {e}. "
                "Please check device availability and permissions."
            ) from e

        # Warm up NCCL to avoid first-time latency
        if world_size > 1:
            try:
                x = torch.ones(1, device=device)
                dist.all_reduce(x)
                torch.cuda.synchronize(device)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to initialize distributed communication on device {rank}: {e}"
                ) from e

        suppress_output(rank)
        return device
        
    except Exception as e:
        # Clean up distributed process group if initialization failed
        if world_size > 1 and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass  # Ignore cleanup errors
        raise
