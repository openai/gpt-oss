import os
import torch
import torch.distributed as dist


def suppress_output(rank):
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
    """Initialize the model for distributed inference.
    
    Returns:
        torch.device: The device to use for computation (CUDA if available, else CPU)
    """
    # Initialize distributed inference
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    if world_size > 1:
        if cuda_available:
            dist.init_process_group(
                backend="nccl", init_method="env://", world_size=world_size, rank=rank
            )
        else:
            dist.init_process_group(
                backend="gloo", init_method="env://", world_size=world_size, rank=rank
            )
    
    # Set device based on CUDA availability
    if cuda_available:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        # Warm up NCCL to avoid first-time latency in distributed training
        if world_size > 1:
            try:
                x = torch.ones(1, device=device)
                dist.all_reduce(x)
                torch.cuda.synchronize(device)
            except Exception as e:
                print(f"Warning: NCCL warmup failed: {e}")
    else:
        device = torch.device(f"cuda:{rank}")
        
        # Warm up NCCL to avoid first-time latency (only for CUDA)
        if world_size > 1:
            x = torch.ones(1, device=device)
            dist.all_reduce(x)
            torch.cuda.synchronize(device)
    else:
        device = torch.device("cpu")
        print("Warning: CUDA is not available. Using CPU instead.")

    suppress_output(rank)
    return device
